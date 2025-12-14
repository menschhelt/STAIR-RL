"""
FinGPT + Mean-Variance Optimization (MVO) Benchmark.

Combines LLM-based market analysis with traditional portfolio optimization.
Uses vLLM for efficient local inference of FinGPT (Llama-3-8B based).

This benchmark represents LLM-enhanced portfolio construction:
1. FinGPT generates sentiment/outlook for each asset
2. Sentiment adjusts expected returns in MVO
3. Markowitz optimization with adjusted returns

Reference:
- FinGPT paper: https://arxiv.org/abs/2306.06031
- vLLM: https://github.com/vllm-project/vllm
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any
from pathlib import Path
import logging
import json
import time
from dataclasses import dataclass

from .base_benchmark import BaseBenchmark, BenchmarkConfig

logger = logging.getLogger(__name__)


@dataclass
class FinGPTConfig:
    """Configuration for FinGPT inference."""
    model_name: str = "FinGPT/fingpt-forecaster-sz-llama3-8b"
    vllm_base_url: str = "http://localhost:8000/v1"
    api_key: str = "dummy"  # vLLM doesn't require real key
    max_tokens: int = 256
    temperature: float = 0.1  # Low temp for consistent outputs
    request_timeout: float = 30.0
    batch_size: int = 5  # Assets per batch to avoid timeout
    cache_enabled: bool = True
    cache_ttl_hours: int = 24  # Cache LLM responses for 24h


class FinGPTClient:
    """
    Client for FinGPT inference via vLLM.

    vLLM Setup (run on GPU 1):
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
        --model FinGPT/fingpt-forecaster-sz-llama3-8b \
        --port 8000 \
        --tensor-parallel-size 1
    ```
    """

    def __init__(self, config: Optional[FinGPTConfig] = None):
        self.config = config or FinGPTConfig()
        self._client = None
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.config.vllm_base_url,
                    api_key=self.config.api_key,
                    timeout=self.config.request_timeout,
                )
            except ImportError:
                raise ImportError(
                    "openai package required for FinGPT. "
                    "Install with: pip install openai"
                )
        return self._client

    def _get_cache_key(self, symbol: str, date: str, context: str) -> str:
        """Generate cache key for LLM response."""
        return f"{symbol}_{date}_{hash(context) % 10000}"

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached response is still valid."""
        if not self.config.cache_enabled:
            return False
        if key not in self._cache:
            return False
        timestamp = self._cache_timestamps.get(key, 0)
        age_hours = (time.time() - timestamp) / 3600
        return age_hours < self.config.cache_ttl_hours

    def get_sentiment(
        self,
        symbol: str,
        date: str,
        price_context: str,
        news_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get FinGPT sentiment analysis for an asset.

        Args:
            symbol: Asset symbol (e.g., 'BTC', 'ETH')
            date: Analysis date (YYYY-MM-DD)
            price_context: Recent price/return information
            news_context: Optional news headlines

        Returns:
            Dict with 'sentiment' (-1 to 1), 'confidence' (0 to 1),
            'reasoning' (text explanation)
        """
        # Build prompt
        prompt = self._build_prompt(symbol, date, price_context, news_context)

        # Check cache
        cache_key = self._get_cache_key(symbol, date, prompt)
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        # Query LLM
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            # Parse response
            result = self._parse_response(response.choices[0].message.content)

            # Cache result
            if self.config.cache_enabled:
                self._cache[cache_key] = result
                self._cache_timestamps[cache_key] = time.time()

            return result

        except Exception as e:
            logger.warning(f"FinGPT query failed for {symbol}: {e}")
            return self._get_default_sentiment()

    def get_batch_sentiment(
        self,
        symbols: List[str],
        date: str,
        price_contexts: Dict[str, str],
        news_contexts: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get sentiment for multiple assets efficiently.

        Args:
            symbols: List of asset symbols
            date: Analysis date
            price_contexts: Dict of symbol -> price context
            news_contexts: Optional dict of symbol -> news context

        Returns:
            Dict of symbol -> sentiment result
        """
        results = {}
        news_contexts = news_contexts or {}

        # Process in batches to avoid timeout
        for i in range(0, len(symbols), self.config.batch_size):
            batch = symbols[i:i + self.config.batch_size]

            for symbol in batch:
                price_ctx = price_contexts.get(symbol, "No price data available.")
                news_ctx = news_contexts.get(symbol)

                results[symbol] = self.get_sentiment(
                    symbol=symbol,
                    date=date,
                    price_context=price_ctx,
                    news_context=news_ctx,
                )

            # Small delay between batches
            if i + self.config.batch_size < len(symbols):
                time.sleep(0.5)

        return results

    def _get_system_prompt(self) -> str:
        """System prompt for FinGPT."""
        return """You are a cryptocurrency market analyst. Analyze the given market data and provide a sentiment assessment.

Your response must be a valid JSON object with exactly these fields:
- "sentiment": a number between -1 (very bearish) and 1 (very bullish)
- "confidence": a number between 0 (uncertain) and 1 (very confident)
- "reasoning": a brief explanation (1-2 sentences)

Example response:
{"sentiment": 0.3, "confidence": 0.7, "reasoning": "Moderate bullish momentum with increasing volume."}"""

    def _build_prompt(
        self,
        symbol: str,
        date: str,
        price_context: str,
        news_context: Optional[str],
    ) -> str:
        """Build analysis prompt."""
        prompt = f"""Analyze {symbol} for {date}:

PRICE DATA:
{price_context}
"""
        if news_context:
            prompt += f"""
NEWS/SOCIAL:
{news_context}
"""
        prompt += "\nProvide your sentiment analysis as JSON:"
        return prompt

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract sentiment."""
        try:
            # Try to extract JSON from response
            # Handle case where response has extra text
            text = response_text.strip()

            # Find JSON object in response
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                result = json.loads(json_str)

                # Validate and normalize
                sentiment = float(result.get('sentiment', 0))
                sentiment = max(-1.0, min(1.0, sentiment))

                confidence = float(result.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))

                reasoning = str(result.get('reasoning', 'No reasoning provided.'))

                return {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'reasoning': reasoning,
                }

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(f"Failed to parse FinGPT response: {e}")

        return self._get_default_sentiment()

    def _get_default_sentiment(self) -> Dict[str, Any]:
        """Return neutral sentiment as default."""
        return {
            'sentiment': 0.0,
            'confidence': 0.0,
            'reasoning': 'Unable to generate analysis.',
        }


class FinGPTMVOBenchmark(BaseBenchmark):
    """
    FinGPT + Mean-Variance Optimization Benchmark.

    Combines LLM-based market analysis with Markowitz portfolio optimization.

    Algorithm:
    1. For each asset, query FinGPT for sentiment analysis
    2. Adjust expected returns based on sentiment:
       μ_adjusted = μ_historical + α × sentiment × confidence
    3. Run Markowitz MVO with adjusted returns
    4. Apply position constraints

    This represents a "human-in-the-loop" style portfolio construction
    where LLM provides qualitative insights that modify quantitative signals.
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        fingpt_config: Optional[FinGPTConfig] = None,
        risk_aversion: float = 2.0,
        sentiment_weight: float = 0.1,  # How much sentiment affects returns
        shrinkage_target: str = 'ledoit_wolf',
        lookback_days: int = 60,
        use_news: bool = False,  # Whether to include news context
    ):
        """
        Initialize FinGPT+MVO benchmark.

        Args:
            config: Benchmark configuration
            fingpt_config: FinGPT client configuration
            risk_aversion: Risk aversion parameter for MVO
            sentiment_weight: Weight of sentiment adjustment (α)
            shrinkage_target: Covariance shrinkage method
            lookback_days: Days for return estimation
            use_news: Whether to query news for context
        """
        super().__init__(config)
        self.name = "FinGPT+MVO"

        self.fingpt_client = FinGPTClient(fingpt_config)
        self.risk_aversion = risk_aversion
        self.sentiment_weight = sentiment_weight
        self.shrinkage_target = shrinkage_target
        self.lookback_days = lookback_days
        self.use_news = use_news

        # Cache for optimization
        self._cov_cache = None
        self._last_cov_date = None

    def compute_weights(
        self,
        timestamp: pd.Timestamp,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute portfolio weights using FinGPT + MVO.

        Steps:
        1. Estimate historical returns and covariance
        2. Query FinGPT for sentiment on each asset
        3. Adjust expected returns with sentiment
        4. Solve MVO optimization
        """
        n_assets = self.config.n_assets

        # Get historical returns
        returns = features.get('returns', None)
        if returns is None or len(returns) < 20:
            # Fallback to equal weight if insufficient data
            return np.ones(n_assets) * self.config.target_leverage / n_assets

        # Use lookback window
        returns = returns[-self.lookback_days:]

        # Estimate expected returns (historical mean)
        mu_historical = np.mean(returns, axis=0)
        if len(mu_historical) < n_assets:
            mu_historical = np.pad(mu_historical, (0, n_assets - len(mu_historical)))

        # Estimate covariance with shrinkage
        cov = self._estimate_covariance(returns)

        # Get FinGPT sentiment for each asset
        sentiments = self._get_sentiments(timestamp, prices, features)

        # Adjust expected returns with sentiment
        mu_adjusted = self._adjust_returns(mu_historical, sentiments)

        # Solve MVO
        weights = self._solve_mvo(mu_adjusted[:n_assets], cov[:n_assets, :n_assets])

        # Scale to target leverage
        weights = self._scale_to_leverage(weights)

        return weights

    def _get_sentiments(
        self,
        timestamp: pd.Timestamp,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Get FinGPT sentiment for each asset."""
        n_assets = self.config.n_assets
        sentiments = np.zeros(n_assets)

        # Get symbol mapping
        symbols = features.get('symbols', None)
        if symbols is None:
            symbols = [f"ASSET_{i}" for i in range(n_assets)]

        # Build price context for each asset
        returns = features.get('returns', np.zeros((20, n_assets)))
        price_contexts = {}

        for i, symbol in enumerate(symbols[:n_assets]):
            if i < returns.shape[1]:
                recent_returns = returns[-5:, i] if len(returns) >= 5 else returns[:, i]
                avg_return = np.mean(recent_returns) * 100  # Convert to percentage
                volatility = np.std(recent_returns) * 100

                price_contexts[symbol] = (
                    f"Recent 5-day avg return: {avg_return:.2f}%\n"
                    f"Recent volatility: {volatility:.2f}%\n"
                    f"Current price: ${prices[i]:.2f}"
                )
            else:
                price_contexts[symbol] = "No price data available."

        # News context (optional)
        news_contexts = None
        if self.use_news:
            news_contexts = features.get('news_headlines', {})

        # Query FinGPT
        try:
            date_str = timestamp.strftime('%Y-%m-%d')
            results = self.fingpt_client.get_batch_sentiment(
                symbols=list(symbols[:n_assets]),
                date=date_str,
                price_contexts=price_contexts,
                news_contexts=news_contexts,
            )

            # Extract sentiments
            for i, symbol in enumerate(symbols[:n_assets]):
                if symbol in results:
                    result = results[symbol]
                    # Weight sentiment by confidence
                    sentiments[i] = result['sentiment'] * result['confidence']

        except Exception as e:
            logger.warning(f"FinGPT sentiment query failed: {e}")

        return sentiments

    def _adjust_returns(
        self,
        mu_historical: np.ndarray,
        sentiments: np.ndarray,
    ) -> np.ndarray:
        """
        Adjust expected returns with sentiment.

        μ_adjusted = μ_historical + α × sentiment

        where α is the sentiment_weight parameter.
        """
        # Annualize historical returns (assuming daily data)
        mu_annual = mu_historical * 252

        # Sentiment adjustment (scaled to reasonable magnitude)
        # sentiment is in [-1, 1], we scale it to affect returns
        sentiment_adjustment = self.sentiment_weight * sentiments

        mu_adjusted = mu_annual + sentiment_adjustment

        return mu_adjusted

    def _estimate_covariance(self, returns: np.ndarray) -> np.ndarray:
        """Estimate covariance matrix with shrinkage."""
        n_assets = returns.shape[1]

        try:
            if self.shrinkage_target == 'ledoit_wolf':
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                lw.fit(returns)
                cov = lw.covariance_
            else:
                cov = np.cov(returns.T)
        except Exception:
            cov = np.cov(returns.T)

        # Ensure positive semi-definite
        cov = self._ensure_psd(cov)

        # Annualize (assuming daily data)
        cov = cov * 252

        # Pad if needed
        if cov.shape[0] < self.config.n_assets:
            new_cov = np.eye(self.config.n_assets) * 0.04  # 20% annual vol default
            new_cov[:cov.shape[0], :cov.shape[1]] = cov
            cov = new_cov

        return cov

    def _ensure_psd(self, cov: np.ndarray) -> np.ndarray:
        """Ensure covariance matrix is positive semi-definite."""
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Set negative eigenvalues to small positive
        eigenvalues = np.maximum(eigenvalues, 1e-8)

        # Reconstruct
        cov_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        return cov_psd

    def _solve_mvo(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
    ) -> np.ndarray:
        """
        Solve Mean-Variance Optimization.

        max: μ'w - (γ/2) w'Σw
        s.t.: Σ|w_i| ≤ L (target leverage)
              |w_i| ≤ w_max (max single weight)
        """
        n = len(mu)

        try:
            import cvxpy as cp

            w = cp.Variable(n)

            # Objective: maximize risk-adjusted return
            portfolio_return = mu @ w
            portfolio_variance = cp.quad_form(w, cov)
            objective = cp.Maximize(portfolio_return - (self.risk_aversion / 2) * portfolio_variance)

            # Constraints
            constraints = [
                cp.norm(w, 1) <= self.config.target_leverage,  # Leverage constraint
                w >= -self.config.max_single_weight,  # Min weight
                w <= self.config.max_single_weight,   # Max weight
            ]

            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS)

            if problem.status in ['optimal', 'optimal_inaccurate']:
                return w.value
            else:
                logger.warning(f"MVO failed with status: {problem.status}")
                return np.ones(n) * self.config.target_leverage / n

        except ImportError:
            logger.warning("cvxpy not installed, using analytical solution")
            return self._analytical_mvo(mu, cov)
        except Exception as e:
            logger.warning(f"MVO optimization failed: {e}")
            return np.ones(n) * self.config.target_leverage / n

    def _analytical_mvo(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Analytical MVO solution (unconstrained)."""
        try:
            # w* = (1/γ) Σ^(-1) μ
            cov_inv = np.linalg.inv(cov + np.eye(len(cov)) * 1e-6)
            weights = cov_inv @ mu / self.risk_aversion

            # Clip to reasonable range
            weights = np.clip(weights, -self.config.max_single_weight, self.config.max_single_weight)

            return weights
        except np.linalg.LinAlgError:
            return np.ones(len(mu)) * self.config.target_leverage / len(mu)

    def _scale_to_leverage(self, weights: np.ndarray) -> np.ndarray:
        """Scale weights to target leverage."""
        gross_exposure = np.abs(weights).sum()

        if gross_exposure > self.config.target_leverage:
            scale = self.config.target_leverage / gross_exposure
            weights = weights * scale

        return weights


class FinGPTMomentumBenchmark(FinGPTMVOBenchmark):
    """
    FinGPT + Momentum Strategy Benchmark.

    Simpler variant that uses FinGPT sentiment to filter momentum signals.

    Algorithm:
    1. Compute momentum scores for each asset
    2. Query FinGPT for sentiment
    3. Combine: final_score = momentum × (1 + sentiment)
    4. Long top N, short bottom N based on combined score
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        fingpt_config: Optional[FinGPTConfig] = None,
        momentum_window: int = 20,
        long_pct: float = 0.3,
        short_pct: float = 0.3,
    ):
        super().__init__(config, fingpt_config)
        self.name = "FinGPT+Momentum"
        self.momentum_window = momentum_window
        self.long_pct = long_pct
        self.short_pct = short_pct

    def compute_weights(
        self,
        timestamp: pd.Timestamp,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute weights using FinGPT-filtered momentum."""
        n_assets = self.config.n_assets

        # Get returns
        returns = features.get('returns', None)
        if returns is None or len(returns) < self.momentum_window:
            return np.zeros(n_assets)

        # Compute momentum
        momentum = np.sum(returns[-self.momentum_window:], axis=0)
        if len(momentum) < n_assets:
            momentum = np.pad(momentum, (0, n_assets - len(momentum)))

        # Get sentiments
        sentiments = self._get_sentiments(timestamp, prices, features)

        # Combine momentum and sentiment
        # Positive sentiment amplifies momentum signal
        combined_score = momentum[:n_assets] * (1 + sentiments[:n_assets])

        # Rank assets
        n_long = int(n_assets * self.long_pct)
        n_short = int(n_assets * self.short_pct)

        sorted_idx = np.argsort(combined_score)

        weights = np.zeros(n_assets)

        # Short bottom performers
        if n_short > 0:
            short_idx = sorted_idx[:n_short]
            weights[short_idx] = -self.config.target_leverage / (2 * n_short)

        # Long top performers
        if n_long > 0:
            long_idx = sorted_idx[-n_long:]
            weights[long_idx] = self.config.target_leverage / (2 * n_long)

        return weights
