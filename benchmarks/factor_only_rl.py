"""
Factor-Only RL Benchmark (= STAIR-RL without LLM).

RL agent using factor-based features:
State = {F_t} where F_t includes:
- PCA-compressed alpha factors
- Risk factors (β_MKT, β_SMB, β_MOM)
- Residual alpha
- Market microstructure features

This is the actual STAIR-RL implementation without LLM semantic tokens.

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


class FactorOnlyRLBenchmark(BaseBenchmark):
    """
    Factor-Only RL Benchmark (STAIR-RL without LLM).

    Uses the full STAIR-RL architecture with:
    - PCA-compressed alpha factors (20 components)
    - Risk factor loadings (β_MKT, β_SMB, β_MOM, α_resid)
    - Market microstructure (momentum, volatility, liquidity)
    - Portfolio state (weights, leverage, cash)

    Missing compared to full STAIR-RL:
    - Semantic tokens from LLM
    - Gating mechanism for text
    - TERC (Text-Enhanced Risk Calibration)

    This represents our actual implementation since we're
    not using LLM API.
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        model_path: Optional[Path] = None,
        device: str = 'cuda',
        use_pca: bool = True,
        n_pca_components: int = 20,
    ):
        """
        Initialize Factor-Only RL benchmark.

        Args:
            config: Benchmark configuration
            model_path: Path to trained model weights
            device: Device for inference ('cuda' or 'cpu')
            use_pca: Whether to use PCA-compressed factors
            n_pca_components: Number of PCA components
        """
        super().__init__(config)
        self.name = "Factor-only RL (STAIR-RL)"
        self.model_path = model_path
        self.device = device
        self.use_pca = use_pca
        self.n_pca_components = n_pca_components

        # State dimensions (matching state_builder.py)
        self.pca_dim = n_pca_components if use_pca else 0
        self.risk_factor_dim = 4  # β_MKT, β_SMB, β_MOM, α_resid
        self.micro_dim = 3  # momentum, volatility, liquidity
        self.private_dim = 2  # weight, sentiment (set to 0 without LLM)

        # Per-asset local features
        self.local_feature_dim = self.pca_dim + self.risk_factor_dim + self.micro_dim + self.private_dim

        # Global features
        self.global_dim = 6  # CMKT, CSMB, CMOM, VIX, yield_spread, risk_free_rate

        # Portfolio state
        self.portfolio_dim = self.config.n_assets + 2  # weights + leverage + cash

        self.model = None
        self._load_model()

    def _load_model(self):
        """Load trained model if path provided."""
        if self.model_path is not None and self.model_path.exists():
            try:
                from agents.ppo_cvar import PPOCVaRAgent
                from config.settings import PPOCVaRConfig

                config = PPOCVaRConfig()
                self.model = PPOCVaRAgent(
                    state_dim=self._get_state_dim(),
                    action_dim=self.config.n_assets,
                    config=config,
                )
                self.model.load(self.model_path)
                self.model.actor.to(self.device)
                self.model.actor.eval()
                logger.info(f"Loaded Factor-only RL model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                self.model = None

    def _get_state_dim(self) -> int:
        """Calculate total state dimension."""
        # Local features: (n_assets, local_feature_dim) -> flattened
        local_total = self.config.n_assets * self.local_feature_dim
        return local_total + self.global_dim + self.portfolio_dim

    def compute_weights(
        self,
        timestamp: pd.Timestamp,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute portfolio weights using Factor-only RL.

        Args:
            timestamp: Current timestamp
            prices: Current asset prices (n_assets,)
            features: Dictionary containing:
                - 'pca_factors': PCA-compressed alpha factors (n_assets, n_components)
                - 'risk_factors': Risk factor loadings (n_assets, 4)
                - 'micro_features': Momentum, vol, liquidity (n_assets, 3)
                - 'global_factors': Market-wide features (6,)
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
            action, _, _ = self.model.actor.get_action(state_tensor, deterministic=True)
            weights = action.cpu().numpy().squeeze()

        return weights

    def _build_state(
        self,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Build state vector for Factor-only RL.

        State structure:
        1. Local features (per-asset, flattened):
           - PCA factors (20)
           - Risk factors (4): β_MKT, β_SMB, β_MOM, α_resid
           - Micro features (3): momentum, volatility, liquidity
           - Private state (2): weight, sentiment (0 without LLM)

        2. Global features (broadcast to all assets):
           - Factor returns: CMKT, CSMB, CMOM (3)
           - Macro: VIX, yield_spread, risk_free_rate (3)

        3. Portfolio state:
           - Current weights (n_assets)
           - Leverage ratio (1)
           - Cash ratio (1)
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

        # Micro features (from returns if not provided)
        micro_features = features.get('micro_features', None)
        if micro_features is None:
            returns = features.get('returns', np.zeros((20, n_assets)))
            momentum = np.sum(returns[-20:], axis=0) if len(returns) >= 20 else np.zeros(n_assets)
            volatility = np.std(returns[-20:], axis=0) if len(returns) >= 20 else np.zeros(n_assets)
            liquidity = np.zeros(n_assets)  # Would need volume data
            micro_features = np.column_stack([momentum, volatility, liquidity])

        if micro_features.shape[0] < n_assets:
            micro_features = np.vstack([
                micro_features,
                np.zeros((n_assets - micro_features.shape[0], self.micro_dim))
            ])

        # Private state (weight from current_weights, sentiment=0 without LLM)
        if current_weights is None:
            current_weights = np.zeros(n_assets)
        private_state = np.column_stack([
            current_weights,
            np.zeros(n_assets)  # sentiment placeholder
        ])

        # Combine local features per asset
        local_features = np.hstack([
            pca_factors[:n_assets, :self.n_pca_components],
            risk_factors[:n_assets, :self.risk_factor_dim],
            micro_features[:n_assets, :self.micro_dim],
            private_state[:n_assets, :self.private_dim],
        ])

        # Flatten to 1D
        local_state = local_features.flatten()

        # ========== Global Features ==========
        global_factors = features.get('global_factors', np.zeros(self.global_dim))
        if len(global_factors) < self.global_dim:
            global_factors = np.pad(global_factors, (0, self.global_dim - len(global_factors)))

        # ========== Portfolio State ==========
        leverage_ratio = np.abs(current_weights).sum() / self.config.target_leverage
        cash_ratio = max(0, 1.0 - leverage_ratio)
        portfolio_state = np.concatenate([
            current_weights,
            [leverage_ratio, cash_ratio]
        ])

        # ========== Combine All ==========
        state = np.concatenate([local_state, global_factors, portfolio_state])

        return state.astype(np.float32)

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        save_path: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the Factor-only RL model.

        This uses the existing training infrastructure from
        training/trainer.py with the appropriate state configuration.

        Args:
            train_data: Training data
            val_data: Validation data
            save_path: Path to save model
            **kwargs: Additional training arguments

        Returns:
            Training metrics
        """
        from training.trainer import Phase1Trainer, Phase2Trainer

        # Phase 1: CQL-SAC offline pre-training
        logger.info("Starting Phase 1: CQL-SAC offline pre-training")
        phase1_trainer = Phase1Trainer(
            state_dim=self._get_state_dim(),
            action_dim=self.config.n_assets,
        )
        phase1_metrics = phase1_trainer.train(train_data, **kwargs.get('phase1', {}))

        # Phase 2: PPO-CVaR online fine-tuning
        logger.info("Starting Phase 2: PPO-CVaR online fine-tuning")
        phase2_trainer = Phase2Trainer(
            state_dim=self._get_state_dim(),
            action_dim=self.config.n_assets,
            pretrained_actor=phase1_trainer.agent.actor,
        )
        phase2_metrics = phase2_trainer.train(val_data, **kwargs.get('phase2', {}))

        # Save model
        self.model = phase2_trainer.agent
        self.model.save(save_path)
        self.model_path = save_path

        return {
            'phase1': phase1_metrics,
            'phase2': phase2_metrics,
        }
