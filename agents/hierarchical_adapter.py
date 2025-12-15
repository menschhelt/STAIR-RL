"""
HierarchicalActorCritic Network Adapter

Provides backward-compatible interface for HierarchicalActorCritic to work with
existing CQL-SAC and PPO-CVaR agents.

This adapter wraps HierarchicalActorCritic and provides methods that match the
interface expected by the current agent implementations.
"""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn

from agents.networks import HierarchicalActorCritic
from agents.hierarchical_state_builder import HierarchicalStateBuilder


class HierarchicalActorCriticAdapter(nn.Module):
    """
    Adapter to make HierarchicalActorCritic compatible with existing agent interfaces.

    This adapter:
    1. Converts (market_state, portfolio_state) → state_dict internally
    2. Provides backward-compatible methods (encode_state, get_action, get_value)
    3. Returns dual outputs (pooled/unpooled) for hierarchical policy
    4. Handles Meta Head and Portfolio Head outputs

    Usage:
        hierarchical_model = HierarchicalActorCritic(...)
        adapter = HierarchicalActorCriticAdapter(hierarchical_model)

        # Encode state (dual outputs)
        z_pooled, z_unpooled = adapter.encode_state(market_state, portfolio_state)

        # Get action
        weights, trade_prob = adapter.get_action(z_pooled, z_unpooled)

        # Get value
        value = adapter.get_value(z_pooled)

        # Get CVaR quantiles
        quantiles = adapter.get_quantiles(z_pooled)
    """

    def __init__(
        self,
        hierarchical_model: HierarchicalActorCritic,
        gdelt_embeddings_path: Optional[str] = None,
        nostr_embeddings_path: Optional[str] = None,
        macro_data_dir: Optional[str] = None,
        device: str = 'cpu',
    ):
        """
        Initialize adapter.

        Args:
            hierarchical_model: Instance of HierarchicalActorCritic
            gdelt_embeddings_path: Path to GDELT HDF5 file (optional)
            nostr_embeddings_path: Path to Nostr HDF5 file (optional)
            macro_data_dir: Path to macro data directory (optional)
            device: Device for tensors
        """
        super().__init__()
        self.model = hierarchical_model
        self.state_builder = HierarchicalStateBuilder(
            n_assets=hierarchical_model.n_assets,
            n_alphas=101,  # Alpha101 only (alpha_000 ~ alpha_100)
            temporal_window=20,
            gdelt_embeddings_path=gdelt_embeddings_path,
            nostr_embeddings_path=nostr_embeddings_path,
            macro_data_dir=macro_data_dir,
            use_normalized_alphas=True,  # Use L1-normalized alphas
            device=device,
        )

    def encode_state(
        self,
        market_state: torch.Tensor,
        portfolio_state: torch.Tensor,
        timestamps: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode state using HierarchicalFeatureEncoder.

        This method:
        1. Converts inputs to state_dict format
        2. Encodes using HierarchicalFeatureEncoder
        3. Returns dual outputs (pooled/unpooled)

        Args:
            market_state: (B, N, state_dim) - Market features
            portfolio_state: (B, portfolio_dim) - Portfolio state
            timestamps: Optional list of ISO timestamps for embedding lookup

        Returns:
            z_pooled: (B, 176) - Pooled representation for Critic
            z_unpooled: (B, N, 176) - Unpooled representation for Actor
        """
        # Convert to state_dict format
        state_dict = self.state_builder.build_state_dict(
            market_state, portfolio_state, timestamps=timestamps
        )

        # Encode using HierarchicalFeatureEncoder
        z_pooled, z_unpooled = self.model.encoder(state_dict)

        return z_pooled, z_unpooled

    def get_action(
        self,
        z_pooled: torch.Tensor,
        z_unpooled: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from HierarchicalActor.

        This method calls the HierarchicalActor which has two heads:
        - Meta Head: Global trade/hold decision (uses z_pooled)
        - Portfolio Head: Per-asset weight allocation (uses z_unpooled)

        Args:
            z_pooled: (B, 176) - Pooled representation
            z_unpooled: (B, N, 176) - Unpooled representation
            deterministic: If True, use deterministic policy (for evaluation)

        Returns:
            weights: (B, N) - Portfolio weights (from Portfolio Head)
            trade_prob: (B, 1) - Trade probability (from Meta Head)

        Notes:
            - trade_prob is currently ignored in training (future enhancement)
            - When trade_prob < 0.5, could set weights to zero (hold)
        """
        # Forward through HierarchicalActor
        # Returns: (trade_prob, weights)
        trade_prob, weights = self.model.actor(z_pooled, z_unpooled)

        # Note: In current implementation, we ignore trade_prob and just use weights
        # Future enhancement: Apply trade_prob threshold to weights
        # if trade_prob < 0.5: weights = zeros

        return weights, trade_prob

    def get_value(self, z_pooled: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate from HierarchicalCritic.

        Args:
            z_pooled: (B, 176) - Pooled representation

        Returns:
            value: (B, 1) - State value estimate
        """
        # Forward through HierarchicalCritic
        # Returns: (value, quantiles)
        value, _ = self.model.critic(z_pooled)
        return value

    def get_quantiles(self, z_pooled: torch.Tensor) -> torch.Tensor:
        """
        Get CVaR quantiles from HierarchicalCritic.

        Args:
            z_pooled: (B, 176) - Pooled representation

        Returns:
            quantiles: (B, n_quantiles) - CVaR quantile estimates

        Notes:
            - Used by PPO-CVaR for risk-sensitive policy optimization
            - Quantiles estimate the distribution of returns
        """
        # Forward through HierarchicalCritic
        # Returns: (value, quantiles)
        _, quantiles = self.model.critic(z_pooled)
        return quantiles

    def forward(
        self,
        market_state: torch.Tensor,
        portfolio_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → act → value.

        This is a convenience method that combines all steps.

        Args:
            market_state: (B, N, state_dim) - Market features
            portfolio_state: (B, portfolio_dim) - Portfolio state

        Returns:
            weights: (B, N) - Portfolio weights
            trade_prob: (B, 1) - Trade probability
            value: (B, 1) - Value estimate
            quantiles: (B, n_quantiles) - CVaR quantiles
        """
        # Encode
        z_pooled, z_unpooled = self.encode_state(market_state, portfolio_state)

        # Act
        weights, trade_prob = self.get_action(z_pooled, z_unpooled)

        # Value
        value = self.get_value(z_pooled)
        quantiles = self.get_quantiles(z_pooled)

        return weights, trade_prob, value, quantiles

    @property
    def device(self) -> torch.device:
        """Get device of the model."""
        return next(self.model.parameters()).device

    def to(self, device: torch.device):
        """Move adapter to device."""
        self.model = self.model.to(device)
        return self

    def train(self, mode: bool = True):
        """Set training mode."""
        self.model.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        return self

    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()

    def state_dict(self):
        """Return model state dict."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """Load model state dict."""
        return self.model.load_state_dict(state_dict)
