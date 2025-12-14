"""
Capitalization-Weighted Benchmark - Market cap weighted portfolio.

Allocate weights proportional to market capitalization.
w_i ∝ MarketCap_i

This is the standard benchmark for most equity indices
(S&P 500, MSCI World, etc.)

For crypto, we use 24h quote volume or market cap as proxy.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

from .base_benchmark import BaseBenchmark, BenchmarkConfig

logger = logging.getLogger(__name__)


class CapWeightBenchmark(BaseBenchmark):
    """
    Capitalization-Weighted Portfolio Strategy.

    Weights are proportional to market capitalization:
    w_i = MarketCap_i / Σ MarketCap_j × L

    where L is the target leverage.

    For crypto assets, we often use 24h quote volume as a proxy
    for market cap when actual market cap data is unavailable.
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        use_volume_as_proxy: bool = True,
        cap_single_weight: bool = True,
    ):
        """
        Initialize Cap-Weight benchmark.

        Args:
            config: Benchmark configuration
            use_volume_as_proxy: If True, use volume when market cap unavailable
            cap_single_weight: If True, cap individual weights at max_single_weight
        """
        super().__init__(config)
        self.name = "Cap-Weight (CW)"
        self.use_volume_as_proxy = use_volume_as_proxy
        self.cap_single_weight = cap_single_weight

    def compute_weights(
        self,
        timestamp: pd.Timestamp,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute market-cap weighted portfolio.

        Args:
            timestamp: Current timestamp
            prices: Current asset prices (n_assets,)
            features: Dictionary with 'market_caps' or 'volumes' key
            current_weights: Current portfolio weights (not used)

        Returns:
            weights: Cap-weighted portfolio (n_assets,)
        """
        n_assets = self.config.n_assets

        # Get market caps
        market_caps = features.get('market_caps', None)

        if market_caps is None and self.use_volume_as_proxy:
            market_caps = features.get('volumes', None)

        if market_caps is None:
            # Fall back to equal weight if no cap data
            logger.warning(f"No market cap data at {timestamp}, using equal weight")
            return np.ones(n_assets) * self.config.target_leverage / n_assets

        # Ensure correct shape
        if len(market_caps) < n_assets:
            market_caps = np.pad(market_caps, (0, n_assets - len(market_caps)), constant_values=0)
        elif len(market_caps) > n_assets:
            market_caps = market_caps[:n_assets]

        # Filter out zero/negative caps
        market_caps = np.maximum(market_caps, 0)
        total_cap = market_caps.sum()

        if total_cap <= 0:
            logger.warning(f"Total market cap is zero at {timestamp}")
            return np.zeros(n_assets)

        # Compute raw weights
        weights = market_caps / total_cap * self.config.target_leverage

        # Cap individual weights if requested
        if self.cap_single_weight:
            excess = np.maximum(weights - self.config.max_single_weight, 0)
            weights = np.minimum(weights, self.config.max_single_weight)

            # Redistribute excess proportionally to uncapped assets
            uncapped_mask = weights < self.config.max_single_weight
            if uncapped_mask.any() and excess.sum() > 0:
                uncapped_weights = weights[uncapped_mask]
                if uncapped_weights.sum() > 0:
                    redistribution = excess.sum() * (uncapped_weights / uncapped_weights.sum())
                    weights[uncapped_mask] += redistribution

        return weights


class VolumeWeightBenchmark(CapWeightBenchmark):
    """
    Volume-Weighted Portfolio Strategy.

    Weights are proportional to 24h trading volume (quote volume).
    w_i = Volume_i / Σ Volume_j × L

    This is often used in crypto as a liquidity-aware weighting scheme.
    High volume assets get higher weights, improving execution quality.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        super().__init__(config, use_volume_as_proxy=True)
        self.name = "Volume-Weight (VW)"

    def compute_weights(
        self,
        timestamp: pd.Timestamp,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute volume-weighted portfolio.

        Explicitly uses volume data instead of market cap.
        """
        # Force use of volume data
        if 'volumes' in features:
            features = {**features, 'market_caps': features['volumes']}

        return super().compute_weights(timestamp, prices, features, current_weights)


class SqrtCapWeightBenchmark(BaseBenchmark):
    """
    Square-Root Market Cap Weighted Portfolio.

    w_i ∝ √MarketCap_i

    This reduces concentration in mega-cap assets compared to
    pure cap-weighting, while still tilting toward larger assets.

    Often used as a compromise between cap-weight and equal-weight.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        super().__init__(config)
        self.name = "Sqrt-Cap-Weight"

    def compute_weights(
        self,
        timestamp: pd.Timestamp,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute square-root cap weighted portfolio.
        """
        n_assets = self.config.n_assets

        market_caps = features.get('market_caps', features.get('volumes', None))

        if market_caps is None:
            return np.ones(n_assets) * self.config.target_leverage / n_assets

        # Ensure correct shape
        if len(market_caps) < n_assets:
            market_caps = np.pad(market_caps, (0, n_assets - len(market_caps)), constant_values=0)

        # Square root transformation
        sqrt_caps = np.sqrt(np.maximum(market_caps, 0))
        total_sqrt_cap = sqrt_caps.sum()

        if total_sqrt_cap <= 0:
            return np.zeros(n_assets)

        weights = sqrt_caps / total_sqrt_cap * self.config.target_leverage

        # Clip to max single weight
        weights = np.clip(weights, 0, self.config.max_single_weight)

        return weights
