"""
Equal-Weight Benchmark - Simple 1/N portfolio allocation.

This is the simplest baseline: allocate equally to all assets.
w_i = 1/N for all i

References:
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009).
  "Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy?"
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

from .base_benchmark import BaseBenchmark, BenchmarkConfig

logger = logging.getLogger(__name__)


class EqualWeightBenchmark(BaseBenchmark):
    """
    Equal-Weight (1/N) Portfolio Strategy.

    The simplest diversification strategy: allocate equal weight
    to all assets in the universe.

    For N assets with target leverage L:
    w_i = L / N  (long-only version)

    This strategy is often surprisingly competitive with more
    sophisticated approaches, especially out-of-sample.
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        long_only: bool = True,
    ):
        """
        Initialize Equal-Weight benchmark.

        Args:
            config: Benchmark configuration
            long_only: If True, all weights positive (default)
                       If False, can go short on some assets
        """
        super().__init__(config)
        self.name = "Equal-Weight (EW)"
        self.long_only = long_only

    def compute_weights(
        self,
        timestamp: pd.Timestamp,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute equal weights for all assets.

        Args:
            timestamp: Current timestamp
            prices: Current asset prices (n_assets,)
            features: Dictionary of features (not used)
            current_weights: Current portfolio weights (not used)

        Returns:
            weights: Equal weights (n_assets,)
        """
        n_assets = self.config.n_assets

        # Filter out assets with zero or missing prices
        valid_mask = (prices > 0) & ~np.isnan(prices)
        n_valid = valid_mask.sum()

        if n_valid == 0:
            logger.warning(f"No valid assets at {timestamp}")
            return np.zeros(n_assets)

        # Equal weight for valid assets, scaled by target leverage
        weights = np.zeros(n_assets)

        if self.long_only:
            # Long-only: w_i = L / N for valid assets
            weight_per_asset = self.config.target_leverage / n_valid
            weights[valid_mask] = weight_per_asset
        else:
            # Long-short: alternate between +L/N and -L/N
            # This is mainly for testing purposes
            weight_per_asset = self.config.target_leverage / n_valid
            valid_indices = np.where(valid_mask)[0]
            for i, idx in enumerate(valid_indices):
                weights[idx] = weight_per_asset if i % 2 == 0 else -weight_per_asset

        return weights


class EqualRiskContributionBenchmark(BaseBenchmark):
    """
    Equal Risk Contribution (ERC) / Risk Parity Strategy.

    Allocate weights so that each asset contributes equally
    to the total portfolio risk.

    For portfolio variance σ²_p:
    RC_i = w_i × ∂σ_p/∂w_i = w_i × (Σw)_i / σ_p

    We want RC_i = RC_j for all i, j

    This is a simplified version that uses inverse volatility weighting
    as an approximation.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        super().__init__(config)
        self.name = "Equal Risk Contribution (ERC)"

    def compute_weights(
        self,
        timestamp: pd.Timestamp,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute inverse-volatility weighted portfolio.

        Args:
            timestamp: Current timestamp
            prices: Current asset prices (n_assets,)
            features: Dictionary with 'returns' key
            current_weights: Current portfolio weights (not used)

        Returns:
            weights: Inverse-volatility weights (n_assets,)
        """
        n_assets = self.config.n_assets

        # Get historical returns
        returns = features.get('returns', None)
        if returns is None or len(returns) < 20:
            # Fall back to equal weight if not enough history
            return np.ones(n_assets) * self.config.target_leverage / n_assets

        # Compute volatility for each asset
        volatilities = np.std(returns, axis=0)

        # Handle zero volatility
        volatilities[volatilities == 0] = np.inf

        # Inverse volatility weights
        inv_vol = 1.0 / volatilities

        # Normalize to sum to target leverage
        weights = inv_vol / inv_vol.sum() * self.config.target_leverage

        # Handle NaN
        weights = np.nan_to_num(weights, nan=0.0)

        return weights
