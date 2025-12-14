"""
Price-Only RL Benchmark.

RL agent using only price-based features:
State = {R_{t-K+1:t}, σ_{t-K+1:t}}

This is an ablation to test the value of adding factor features.

Reference:
- STAIR-RL paper, Section 4.2 (Benchmarks)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from pathlib import Path
import torch
import logging

from .base_benchmark import BaseBenchmark, BenchmarkConfig, BacktestResult

logger = logging.getLogger(__name__)


class PriceOnlyRLBenchmark(BaseBenchmark):
    """
    Price-Only RL Benchmark.

    Uses the same RL architecture as STAIR-RL but with
    a reduced state space containing only:
    - Historical returns: R_{t-K+1:t}
    - Historical volatility: σ_{t-K+1:t}
    - Portfolio state: weights, leverage, cash

    This tests whether price momentum/volatility alone
    is sufficient for portfolio optimization.
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        lookback: int = 20,
        model_path: Optional[Path] = None,
        device: str = 'cuda',
    ):
        """
        Initialize Price-Only RL benchmark.

        Args:
            config: Benchmark configuration
            lookback: Number of lookback periods for features
            model_path: Path to trained model weights
            device: Device for inference ('cuda' or 'cpu')
        """
        super().__init__(config)
        self.name = "Price-only RL"
        self.lookback = lookback
        self.model_path = model_path
        self.device = device

        # State dimensions
        self.returns_dim = lookback  # Historical returns
        self.vol_dim = lookback       # Historical volatility
        self.portfolio_dim = self.config.n_assets + 2  # weights + leverage + cash

        # Total state dim per asset
        self.state_dim_per_asset = self.returns_dim + self.vol_dim + 2  # + momentum, vol summary

        # Total global features
        self.global_dim = 2  # market return, market vol

        self.model = None
        self._load_model()

    def _load_model(self):
        """Load trained model if path provided."""
        if self.model_path is not None and self.model_path.exists():
            try:
                from agents.ppo_cvar import PPOCVaRAgent, PPOConfig

                config = PPOConfig(
                    state_dim=self._get_state_dim(),
                    action_dim=self.config.n_assets,
                )
                self.model = PPOCVaRAgent(config)
                self.model.load(self.model_path)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Loaded Price-only RL model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                self.model = None

    def _get_state_dim(self) -> int:
        """Calculate total state dimension."""
        local_dim = self.state_dim_per_asset * self.config.n_assets
        return local_dim + self.global_dim + self.portfolio_dim

    def compute_weights(
        self,
        timestamp: pd.Timestamp,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute portfolio weights using Price-only RL.

        Args:
            timestamp: Current timestamp
            prices: Current asset prices (n_assets,)
            features: Dictionary with 'returns' key
            current_weights: Current portfolio weights

        Returns:
            weights: Target portfolio weights (n_assets,)
        """
        n_assets = self.config.n_assets

        if self.model is None:
            # Fallback to equal weight if no model
            return np.ones(n_assets) * self.config.target_leverage / n_assets

        # Build state
        state = self._build_state(prices, features, current_weights)

        # Get action from model
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.model.select_action(state_tensor, deterministic=True)
            weights = action.cpu().numpy().squeeze()

        return weights

    def _build_state(
        self,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Build state vector for Price-only RL.

        State components:
        1. Per-asset features:
           - Recent returns (lookback periods)
           - Recent volatility (rolling std)
           - Momentum (cumulative return)
           - Volatility summary (current vol)

        2. Global features:
           - Market return (equal-weighted average)
           - Market volatility

        3. Portfolio state:
           - Current weights
           - Leverage ratio
           - Cash ratio
        """
        n_assets = self.config.n_assets
        returns = features.get('returns', np.zeros((self.lookback, n_assets)))

        # Pad if needed
        if returns.shape[0] < self.lookback:
            padding = np.zeros((self.lookback - returns.shape[0], returns.shape[1]))
            returns = np.vstack([padding, returns])

        # Ensure correct number of assets
        if returns.shape[1] < n_assets:
            padding = np.zeros((returns.shape[0], n_assets - returns.shape[1]))
            returns = np.hstack([returns, padding])

        # Use last lookback periods
        returns = returns[-self.lookback:]

        # Per-asset features
        local_features = []
        for i in range(n_assets):
            asset_returns = returns[:, i]

            # Rolling volatility
            vol = np.std(asset_returns) if len(asset_returns) > 1 else 0.0

            # Momentum (cumulative return)
            momentum = np.sum(asset_returns)

            # Combine: [returns..., momentum, vol]
            asset_state = np.concatenate([
                asset_returns,  # lookback returns
                [momentum, vol]  # summary stats
            ])
            local_features.append(asset_state)

        local_state = np.concatenate(local_features)

        # Global features
        market_return = np.mean(returns[-1]) if len(returns) > 0 else 0.0
        market_vol = np.std(np.mean(returns, axis=1)) if len(returns) > 1 else 0.0
        global_state = np.array([market_return, market_vol])

        # Portfolio state
        if current_weights is None:
            current_weights = np.zeros(n_assets)
        leverage_ratio = np.abs(current_weights).sum() / self.config.target_leverage
        cash_ratio = 1.0 - leverage_ratio

        portfolio_state = np.concatenate([
            current_weights,
            [leverage_ratio, cash_ratio]
        ])

        # Combine all
        state = np.concatenate([local_state, global_state, portfolio_state])

        return state.astype(np.float32)


class MomentumRLBenchmark(PriceOnlyRLBenchmark):
    """
    Momentum-based RL Benchmark.

    Simplified version using only momentum signals:
    - Short-term momentum (5-day)
    - Medium-term momentum (20-day)
    - Long-term momentum (60-day)

    This is a further ablation of Price-only RL.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None, **kwargs):
        super().__init__(config, lookback=60, **kwargs)
        self.name = "Momentum RL"
        self.short_window = 5
        self.medium_window = 20
        self.long_window = 60

    def _build_state(
        self,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Build momentum-based state.

        Per-asset features:
        - Short-term momentum (5-day return)
        - Medium-term momentum (20-day return)
        - Long-term momentum (60-day return)
        - Current volatility
        """
        n_assets = self.config.n_assets
        returns = features.get('returns', np.zeros((self.lookback, n_assets)))

        # Pad if needed
        if returns.shape[0] < self.lookback:
            padding = np.zeros((self.lookback - returns.shape[0], returns.shape[1]))
            returns = np.vstack([padding, returns])
        returns = returns[-self.lookback:]

        # Per-asset momentum features
        local_features = []
        for i in range(n_assets):
            asset_returns = returns[:, i]

            # Momentum at different horizons
            mom_short = np.sum(asset_returns[-self.short_window:])
            mom_medium = np.sum(asset_returns[-self.medium_window:])
            mom_long = np.sum(asset_returns)

            # Volatility
            vol = np.std(asset_returns[-20:]) if len(asset_returns) >= 20 else 0.0

            local_features.extend([mom_short, mom_medium, mom_long, vol])

        local_state = np.array(local_features)

        # Portfolio state
        if current_weights is None:
            current_weights = np.zeros(n_assets)
        leverage_ratio = np.abs(current_weights).sum() / self.config.target_leverage
        portfolio_state = np.concatenate([current_weights, [leverage_ratio]])

        state = np.concatenate([local_state, portfolio_state])
        return state.astype(np.float32)
