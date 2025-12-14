"""
Sentiment-Score RL Benchmark.

RL agent using factor + sentiment score features:
State = {F_t, z^score_t}

This is an intermediate between Factor-only RL and full STAIR-RL.
It uses pre-computed sentiment scores instead of raw semantic tokens.

The key difference from STAIR-RL:
- Uses aggregated sentiment scores (z^score), NOT semantic tokens
- No gating mechanism
- No TERC

Reference:
- STAIR-RL paper, Section 4.2 (Benchmarks)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from pathlib import Path
import torch
import logging

from .base_benchmark import BaseBenchmark, BenchmarkConfig
from .factor_only_rl import FactorOnlyRLBenchmark

logger = logging.getLogger(__name__)


class SentimentScoreRLBenchmark(FactorOnlyRLBenchmark):
    """
    Sentiment-Score RL Benchmark.

    Extends Factor-only RL by adding sentiment score features:
    - Per-asset sentiment from GDELT/Nostr (aggregated score)
    - Market-wide sentiment (average across assets)

    State = {F_t, z^score_t}

    where z^score_t is a scalar sentiment score per asset,
    NOT the raw semantic tokens used in full STAIR-RL.

    Sentiment scores are pre-computed from:
    - FinBERT on GDELT news articles
    - CryptoBERT on Nostr posts
    - Weighted by engagement (mentions/zaps)
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        model_path: Optional[Path] = None,
        device: str = 'cuda',
        use_pca: bool = True,
        n_pca_components: int = 20,
        sentiment_sources: str = 'both',  # 'gdelt', 'nostr', 'both'
    ):
        """
        Initialize Sentiment-Score RL benchmark.

        Args:
            config: Benchmark configuration
            model_path: Path to trained model weights
            device: Device for inference
            use_pca: Whether to use PCA-compressed factors
            n_pca_components: Number of PCA components
            sentiment_sources: Which sentiment sources to use
        """
        super().__init__(
            config=config,
            model_path=model_path,
            device=device,
            use_pca=use_pca,
            n_pca_components=n_pca_components,
        )
        self.name = "Sentiment-score RL"
        self.sentiment_sources = sentiment_sources

        # Additional state dimensions for sentiment
        self.sentiment_dim = 1  # Single sentiment score per asset
        self.private_dim = 2  # weight + sentiment (now populated)

        # Recalculate local feature dim
        self.local_feature_dim = (
            self.n_pca_components +  # PCA factors
            self.risk_factor_dim +   # Risk factors
            self.micro_dim +         # Momentum, vol, liquidity
            self.private_dim         # Weight + sentiment
        )

        # Update global dim to include market sentiment
        self.global_dim = 7  # CMKT, CSMB, CMOM, VIX, yield_spread, rf, market_sentiment

    def _build_state(
        self,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Build state vector for Sentiment-Score RL.

        Same as Factor-only RL but with sentiment scores added:
        - Per-asset sentiment in private state
        - Market-wide sentiment in global features
        """
        n_assets = self.config.n_assets

        # ========== Local Features ==========
        # PCA factors
        pca_factors = features.get('pca_factors', np.zeros((n_assets, self.n_pca_components)))
        if pca_factors.shape[0] < n_assets:
            pca_factors = np.vstack([
                pca_factors,
                np.zeros((n_assets - pca_factors.shape[0], self.n_pca_components))
            ])

        # Risk factors
        risk_factors = features.get('risk_factors', np.zeros((n_assets, self.risk_factor_dim)))
        if risk_factors.shape[0] < n_assets:
            risk_factors = np.vstack([
                risk_factors,
                np.zeros((n_assets - risk_factors.shape[0], self.risk_factor_dim))
            ])

        # Micro features
        micro_features = features.get('micro_features', None)
        if micro_features is None:
            returns = features.get('returns', np.zeros((20, n_assets)))
            momentum = np.sum(returns[-20:], axis=0) if len(returns) >= 20 else np.zeros(n_assets)
            volatility = np.std(returns[-20:], axis=0) if len(returns) >= 20 else np.zeros(n_assets)
            liquidity = np.zeros(n_assets)
            micro_features = np.column_stack([momentum, volatility, liquidity])

        if micro_features.shape[0] < n_assets:
            micro_features = np.vstack([
                micro_features,
                np.zeros((n_assets - micro_features.shape[0], self.micro_dim))
            ])

        # ========== Sentiment Scores ==========
        sentiment_scores = self._get_sentiment_scores(features, n_assets)

        # Private state: [weight, sentiment]
        if current_weights is None:
            current_weights = np.zeros(n_assets)
        private_state = np.column_stack([
            current_weights,
            sentiment_scores
        ])

        # Combine local features
        local_features = np.hstack([
            pca_factors[:n_assets, :self.n_pca_components],
            risk_factors[:n_assets, :self.risk_factor_dim],
            micro_features[:n_assets, :self.micro_dim],
            private_state[:n_assets, :self.private_dim],
        ])
        local_state = local_features.flatten()

        # ========== Global Features ==========
        global_factors = features.get('global_factors', np.zeros(6))
        if len(global_factors) < 6:
            global_factors = np.pad(global_factors, (0, 6 - len(global_factors)))

        # Add market-wide sentiment
        market_sentiment = np.mean(sentiment_scores)
        global_state = np.concatenate([global_factors, [market_sentiment]])

        # ========== Portfolio State ==========
        leverage_ratio = np.abs(current_weights).sum() / self.config.target_leverage
        cash_ratio = max(0, 1.0 - leverage_ratio)
        portfolio_state = np.concatenate([
            current_weights,
            [leverage_ratio, cash_ratio]
        ])

        # ========== Combine All ==========
        state = np.concatenate([local_state, global_state, portfolio_state])

        return state.astype(np.float32)

    def _get_sentiment_scores(
        self,
        features: Dict[str, np.ndarray],
        n_assets: int,
    ) -> np.ndarray:
        """
        Extract or compute sentiment scores.

        Sentiment score z^score_i is computed as:
        z^score = weighted_bullish - weighted_bearish

        where weights come from engagement (mentions/zaps).

        Sources:
        - GDELT: FinBERT scores weighted by num_mentions
        - Nostr: CryptoBERT scores weighted by zap_amount
        """
        sentiment = np.zeros(n_assets)

        # GDELT sentiment (FinBERT)
        if self.sentiment_sources in ['gdelt', 'both']:
            gdelt_sentiment = features.get('gdelt_sentiment', None)
            if gdelt_sentiment is not None:
                # Expected shape: (n_assets,) or (n_assets, 3) for [neg, neu, pos]
                if gdelt_sentiment.ndim == 2:
                    # Convert [neg, neu, pos] to single score
                    gdelt_score = gdelt_sentiment[:, 2] - gdelt_sentiment[:, 0]
                else:
                    gdelt_score = gdelt_sentiment

                if len(gdelt_score) >= n_assets:
                    sentiment += gdelt_score[:n_assets]
                else:
                    sentiment[:len(gdelt_score)] += gdelt_score

        # Nostr sentiment (CryptoBERT)
        if self.sentiment_sources in ['nostr', 'both']:
            nostr_sentiment = features.get('nostr_sentiment', None)
            if nostr_sentiment is not None:
                # Expected shape: (n_assets,) or (n_assets, 3) for [bear, neu, bull]
                if nostr_sentiment.ndim == 2:
                    # Convert [bear, neu, bull] to single score
                    nostr_score = nostr_sentiment[:, 2] - nostr_sentiment[:, 0]
                else:
                    nostr_score = nostr_sentiment

                if len(nostr_score) >= n_assets:
                    sentiment += nostr_score[:n_assets]
                else:
                    sentiment[:len(nostr_score)] += nostr_score

        # Average if using both sources
        if self.sentiment_sources == 'both':
            sentiment /= 2.0

        # Normalize to [-1, 1] range
        sentiment = np.clip(sentiment, -1.0, 1.0)

        return sentiment


class SentimentMomentumRLBenchmark(SentimentScoreRLBenchmark):
    """
    Sentiment + Momentum RL Benchmark.

    Combines momentum signals with sentiment scores.
    State = {Momentum, z^score_t}

    A simpler variant that doesn't use full factor model,
    just momentum and sentiment.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.name = "Sentiment-Momentum RL"
        self.use_pca = False
        self.n_pca_components = 0

        # Simplified features
        self.local_feature_dim = (
            3 +  # Short, medium, long momentum
            1 +  # Volatility
            2    # Weight + sentiment
        )

    def _build_state(
        self,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build simplified momentum + sentiment state."""
        n_assets = self.config.n_assets

        returns = features.get('returns', np.zeros((60, n_assets)))
        if returns.shape[0] < 60:
            padding = np.zeros((60 - returns.shape[0], returns.shape[1]))
            returns = np.vstack([padding, returns])

        # Momentum features
        local_features = []
        for i in range(n_assets):
            asset_returns = returns[:, i]
            mom_short = np.sum(asset_returns[-5:])
            mom_medium = np.sum(asset_returns[-20:])
            mom_long = np.sum(asset_returns[-60:])
            vol = np.std(asset_returns[-20:])
            local_features.extend([mom_short, mom_medium, mom_long, vol])

        # Sentiment
        sentiment_scores = self._get_sentiment_scores(features, n_assets)

        # Weights
        if current_weights is None:
            current_weights = np.zeros(n_assets)

        # Combine per-asset
        for i in range(n_assets):
            local_features.extend([current_weights[i], sentiment_scores[i]])

        local_state = np.array(local_features)

        # Global: market momentum + market sentiment
        market_mom = np.mean(returns[-20:])
        market_sentiment = np.mean(sentiment_scores)
        global_state = np.array([market_mom, market_sentiment])

        # Portfolio
        leverage_ratio = np.abs(current_weights).sum() / self.config.target_leverage
        portfolio_state = np.array([leverage_ratio])

        state = np.concatenate([local_state, global_state, portfolio_state])
        return state.astype(np.float32)
