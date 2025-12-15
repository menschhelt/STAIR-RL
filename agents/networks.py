"""
Neural Networks for Portfolio RL Agents.

Architecture based on the thesis design:
- FeatureEncoder: 1D-CNN + GRU for local features, MLP for global/portfolio
- Actor: Policy network with Tanh output for continuous weights [-1, 1]
- Critic: Value function (Q for SAC, V for PPO)

Input State:
- market_state: (N_assets=20, D_features=35)
- portfolio_state: (22,) = [weights(20), leverage_ratio, cash_ratio]

Output Action:
- weights: (N_assets,) in [-1, 1] via Tanh activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

# TERC (Transfer Entropy with Recursive Conditioning) for semantic filtering
from .terc import TERCFilter, SemanticGate, TERCConfig, TERCModule


class FeatureEncoder(nn.Module):
    """
    Feature encoder for market and portfolio states.

    Architecture:
    1. Local Features (per asset):
       - 1D-CNN: (35,) -> (64,) with kernel=3
       - GRU: (64,) -> (128,) hidden state
       - Output: h_local (N_assets, 128)

    2. Global Features (shared):
       - MLP: (6,) -> (32,) -> (32,)
       - Output: h_global (32,)

    3. Portfolio State:
       - MLP: (22,) -> (64,) -> (32,) -> (16,)
       - Output: x_port (16,)

    Final embedding: z = [avg(h_local), h_global, x_port]
    Dimension: 128 + 32 + 16 = 176
    """

    def __init__(
        self,
        n_assets: int = 20,
        local_feature_dim: int = 29,  # Per-asset features
        global_feature_dim: int = 6,   # Market-wide features
        portfolio_state_dim: int = 22, # weights + leverage_ratio + cash_ratio
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.n_assets = n_assets
        self.local_feature_dim = local_feature_dim
        self.global_feature_dim = global_feature_dim
        self.portfolio_state_dim = portfolio_state_dim
        self.hidden_dim = hidden_dim

        # Total state dimension per asset
        total_feature_dim = local_feature_dim + global_feature_dim  # 35

        # ========== Local Feature Encoder ==========
        # 1D-CNN for spatial patterns across features
        self.local_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # GRU for sequential processing
        self.local_gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # ========== Global Feature Encoder ==========
        self.global_mlp = nn.Sequential(
            nn.Linear(global_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # ========== Portfolio State Encoder ==========
        self.portfolio_mlp = nn.Sequential(
            nn.Linear(portfolio_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Output dimension: hidden_dim + 32 + 16 = 176
        self.output_dim = hidden_dim + 32 + 16

    def forward(
        self,
        market_state: torch.Tensor,
        portfolio_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode market and portfolio states.

        Args:
            market_state: (batch, n_assets, 35) market features
            portfolio_state: (batch, 22) portfolio state

        Returns:
            z: (batch, 176) encoded state embedding
        """
        batch_size = market_state.shape[0]

        # ========== Process Local Features ==========
        # Reshape for CNN: (batch * n_assets, 1, 35)
        local_features = market_state.view(batch_size * self.n_assets, 1, -1)

        # CNN: (batch * n_assets, 64, 35)
        cnn_out = self.local_cnn(local_features)

        # Transpose for GRU: (batch * n_assets, 35, 64)
        cnn_out = cnn_out.transpose(1, 2)

        # GRU: take last hidden state
        _, h_local = self.local_gru(cnn_out)  # h_local: (1, batch*n_assets, hidden_dim)
        h_local = h_local.squeeze(0)  # (batch*n_assets, hidden_dim)

        # Reshape and average pool across assets
        h_local = h_local.view(batch_size, self.n_assets, self.hidden_dim)
        h_local_pooled = h_local.mean(dim=1)  # (batch, hidden_dim)

        # ========== Process Global Features ==========
        # Extract global features from market state (last 6 dims, same for all assets)
        global_features = market_state[:, 0, -self.global_feature_dim:]  # (batch, 6)
        h_global = self.global_mlp(global_features)  # (batch, 32)

        # ========== Process Portfolio State ==========
        x_port = self.portfolio_mlp(portfolio_state)  # (batch, 16)

        # ========== Concatenate ==========
        z = torch.cat([h_local_pooled, h_global, x_port], dim=-1)  # (batch, 176)

        return z


class Actor(nn.Module):
    """
    Policy network for continuous action space.

    Input: encoded state (176-dim)
    Output: portfolio weights in [-1, 1] via Tanh activation

    Architecture:
    - Linear(176, 512) -> ReLU -> LayerNorm
    - Linear(512, 256) -> ReLU -> LayerNorm
    - Linear(256, 128) -> ReLU
    - Linear(128, n_assets) -> Tanh
    """

    def __init__(
        self,
        input_dim: int = 176,
        n_assets: int = 20,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
    ):
        super().__init__()

        self.n_assets = n_assets

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:  # No LayerNorm on last hidden
                layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, n_assets)

        # Initialize output layer with small weights for stable training
        nn.init.uniform_(self.output_layer.weight, -0.003, 0.003)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute policy output.

        Args:
            z: (batch, 176) encoded state

        Returns:
            action: (batch, n_assets) portfolio weights in [-1, 1]
        """
        x = self.mlp(z)
        action = torch.tanh(self.output_layer(x))
        return action

    def get_action(
        self,
        z: torch.Tensor,
        deterministic: bool = False,
        noise_scale: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action with optional exploration noise.

        Args:
            z: encoded state
            deterministic: if True, return mean action without noise
            noise_scale: std of Gaussian noise for exploration

        Returns:
            Tuple of (action, log_prob)
        """
        mean_action = self.forward(z)

        if deterministic:
            return mean_action, torch.zeros_like(mean_action[:, 0])

        # Add exploration noise
        noise = torch.randn_like(mean_action) * noise_scale
        action = torch.tanh(torch.atanh(mean_action.clamp(-0.999, 0.999)) + noise)

        # Approximate log probability (for entropy bonus)
        log_prob = -0.5 * (noise ** 2).sum(dim=-1)

        return action, log_prob


class Critic(nn.Module):
    """
    Value function network.

    For SAC (Q-function):
    - Input: state (176-dim) + action (n_assets)
    - Output: Q(s, a) scalar

    For PPO (V-function):
    - Input: state (176-dim)
    - Output: V(s) scalar

    Architecture:
    - Linear(input_dim, 512) -> ReLU -> LayerNorm
    - Linear(512, 256) -> ReLU -> LayerNorm
    - Linear(256, 128) -> ReLU
    - Linear(128, 1)
    """

    def __init__(
        self,
        state_dim: int = 176,
        action_dim: int = 20,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        q_function: bool = True,  # True for Q(s,a), False for V(s)
    ):
        super().__init__()

        self.q_function = q_function
        input_dim = state_dim + action_dim if q_function else state_dim

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)

        # Initialize output layer
        nn.init.uniform_(self.output_layer.weight, -0.003, 0.003)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute value.

        Args:
            state: (batch, 176) encoded state
            action: (batch, n_assets) action (for Q-function only)

        Returns:
            value: (batch, 1) value estimate
        """
        if self.q_function:
            if action is None:
                raise ValueError("Q-function requires action input")
            # Debug: print shapes if they don't match expected
            if state.shape[-1] + action.shape[-1] != self.mlp[0].in_features:
                raise ValueError(
                    f"Shape mismatch in Critic! state: {state.shape}, action: {action.shape}, "
                    f"expected input: {self.mlp[0].in_features}"
                )
            x = torch.cat([state, action], dim=-1)
        else:
            x = state

        x = self.mlp(x)
        value = self.output_layer(x)
        return value


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network with shared feature encoder.

    Used for PPO where both policy and value share the backbone.
    """

    def __init__(
        self,
        n_assets: int = 20,
        local_feature_dim: int = 29,
        global_feature_dim: int = 6,
        portfolio_state_dim: int = 22,
        hidden_dim: int = 128,
        actor_hidden_dims: Tuple[int, ...] = (512, 256, 128),
        critic_hidden_dims: Tuple[int, ...] = (512, 256, 128),
    ):
        super().__init__()

        self.n_assets = n_assets

        # Shared feature encoder
        self.encoder = FeatureEncoder(
            n_assets=n_assets,
            local_feature_dim=local_feature_dim,
            global_feature_dim=global_feature_dim,
            portfolio_state_dim=portfolio_state_dim,
            hidden_dim=hidden_dim,
        )

        # Actor (policy)
        self.actor = Actor(
            input_dim=self.encoder.output_dim,
            n_assets=n_assets,
            hidden_dims=actor_hidden_dims,
        )

        # Critic (value function)
        self.critic = Critic(
            state_dim=self.encoder.output_dim,
            action_dim=n_assets,
            hidden_dims=critic_hidden_dims,
            q_function=False,  # V-function for PPO
        )

    def forward(
        self,
        market_state: torch.Tensor,
        portfolio_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both actor and critic.

        Args:
            market_state: (batch, n_assets, 35)
            portfolio_state: (batch, 22)

        Returns:
            Tuple of (action, value)
        """
        z = self.encoder(market_state, portfolio_state)
        action = self.actor(z)
        value = self.critic(z)
        return action, value

    def get_action_and_value(
        self,
        market_state: torch.Tensor,
        portfolio_state: torch.Tensor,
        deterministic: bool = False,
        noise_scale: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value.

        Args:
            market_state: (batch, n_assets, 35)
            portfolio_state: (batch, 22)
            deterministic: if True, return mean action
            noise_scale: exploration noise scale

        Returns:
            Tuple of (action, log_prob, value)
        """
        z = self.encoder(market_state, portfolio_state)
        action, log_prob = self.actor.get_action(z, deterministic, noise_scale)
        value = self.critic(z)
        return action, log_prob, value.squeeze(-1)

    def get_value(
        self,
        market_state: torch.Tensor,
        portfolio_state: torch.Tensor,
    ) -> torch.Tensor:
        """Get value estimate only."""
        z = self.encoder(market_state, portfolio_state)
        return self.critic(z).squeeze(-1)


class TwinCritic(nn.Module):
    """
    Twin Q-networks for SAC (clipped double Q-learning).

    Uses two independent Q-networks and takes the minimum
    to reduce overestimation bias.
    """

    def __init__(
        self,
        state_dim: int = 176,
        action_dim: int = 20,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
    ):
        super().__init__()

        self.q1 = Critic(state_dim, action_dim, hidden_dims, q_function=True)
        self.q2 = Critic(state_dim, action_dim, hidden_dims, q_function=True)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both Q-values.

        Returns:
            Tuple of (q1, q2)
        """
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2

    def q_min(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute minimum of Q1 and Q2 (for conservative estimate)."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


# ========== Utility Functions ==========

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """
    Soft update target network parameters.

    target = tau * source + (1 - tau) * target
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def hard_update(target: nn.Module, source: nn.Module):
    """Hard update: copy source parameters to target."""
    target.load_state_dict(source.state_dict())


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# Hierarchical Architecture (v2) - Attention-based with unpooled Portfolio Head
# ==============================================================================

class CrossAlphaAttention(nn.Module):
    """
    Cross-Alpha Attention for learnable alpha compression.

    Compresses 292 alpha factors into 64-dimensional representations
    while learning which alphas are important for the current market state.

    Architecture:
    - Input projection: (B, T, N, 292) -> (B, T, N, 64)
    - Self-attention across feature dimension
    - FFN with residual connection

    Note: Since we project first then attend, the attention learns
    which projected features are important, not raw alpha importance.
    """

    def __init__(
        self,
        n_alphas: int = 101,
        d_model: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_alphas = n_alphas

        # Project alphas to d_model dimension
        self.input_proj = nn.Linear(n_alphas, d_model)

        # Layer norm before attention
        self.norm1 = nn.LayerNorm(d_model)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, alphas: torch.Tensor) -> torch.Tensor:
        """
        Process alpha factors through attention.

        Args:
            alphas: (B, T, N, 292) raw alpha factors
                B = batch size
                T = lookback window (time steps)
                N = number of assets (20)
                292 = alpha features (101 + 191)

        Returns:
            h_alpha: (B, T, N, 64) compressed alpha representations
        """
        B, T, N, A = alphas.shape

        # Reshape for processing: (B*T*N, 1, 292)
        x = alphas.reshape(B * T * N, 1, A)

        # Project to d_model: (B*T*N, 1, 64)
        x = self.input_proj(x)

        # Self-attention (with single token, this mainly adds nonlinearity)
        # In practice, we could extend to attend across time or assets
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x)
        x = residual + attn_out

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        # Reshape back: (B, T, N, 64)
        return x.reshape(B, T, N, self.d_model)


class CrossAssetAttention(nn.Module):
    """
    Cross-Asset Attention for learning inter-asset correlations.

    Allows each asset to attend to all other assets, learning
    dynamic correlations like "when BTC moves, how does ETH respond?"

    Architecture:
    - Self-attention across N=20 assets
    - FFN with residual connection
    """

    def __init__(
        self,
        d_model: int = 192,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-asset attention.

        Args:
            x: (B, T, N, d_model) features per asset

        Returns:
            x: (B, T, N, d_model) attended features
        """
        B, T, N, D = x.shape

        # Reshape: (B*T, N, D) - attend across assets
        x = x.reshape(B * T, N, D)

        # Self-attention across assets
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = residual + attn_out

        # FFN
        residual = x
        x = residual + self.ffn(self.norm2(x))

        # Reshape back: (B, T, N, D)
        return x.reshape(B, T, N, D)


class TextProjection(nn.Module):
    """
    Projects BERT embeddings (768-dim) to lower dimension (64-dim).

    Used for both FinBERT (news) and CryptoBERT (social) embeddings.
    Includes optional masking via has_signal flag.
    """

    def __init__(
        self,
        bert_dim: int = 768,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.proj = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        has_signal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project BERT embeddings.

        Args:
            x: (B, T, 768) OR (B, T, N, 768) BERT embeddings (market-wide or per-asset)
            has_signal: (B, T, 1) OR (B, T, N, 1) binary mask indicating data availability

        Returns:
            (B, T, 64) OR (B, T, N, 64) projected embeddings (masked if has_signal provided)
        """
        out = self.proj(x)

        # Mask out embeddings where no text data exists
        if has_signal is not None:
            out = out * has_signal

        return out


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.

    Adds sinusoidal position embeddings to the input sequence.
    Standard implementation from "Attention is All You Need".
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create position encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) or (B*N, T, d_model)

        Returns:
            (B, T, d_model) or (B*N, T, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class PriceTransformerEncoder(nn.Module):
    """
    Transformer-based encoder for 5-minute OHLCV sequences.

    Processes raw price sequences into compact embeddings using self-attention.
    This captures temporal patterns in price movements and volume dynamics.

    Architecture:
    1. Linear embedding: (5) -> (64)  [OHLCV -> hidden dim]
    2. Positional encoding: Add position information
    3. Transformer encoder: 2 layers, 4 heads, 128 feedforward dim
    4. Temporal pooling: Mean over sequence -> (64)

    Args:
        d_model: Hidden dimension (default: 64)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 2)
        dim_feedforward: Feedforward dimension (default: 128)
        dropout: Dropout rate (default: 0.1)

    Input:
        ohlcv_seq: (B, T, N, seq_len, 5) where seq_len=288 (1 day of 5-min candles)
            T: temporal window (e.g., 20 timesteps)
            N: number of assets (e.g., 20)
            seq_len: sequence length (e.g., 288 = 24h * 12/h)
            5: [open, high, low, close, volume] (normalized)

    Output:
        (B, T, N, 64) price embeddings
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Embedding layer: 5 (OHLCV) -> d_model
        self.embedding = nn.Linear(5, d_model)

        # Positional encoding
        self.positional = PositionalEncoding(d_model, max_len=512)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, ohlcv_seq: torch.Tensor) -> torch.Tensor:
        """
        Encode OHLCV sequences.

        Args:
            ohlcv_seq: (B, T, N, seq_len, 5)
                B: batch size
                T: temporal window (e.g., 20)
                N: number of assets (e.g., 20)
                seq_len: sequence length (e.g., 288)
                5: [open, high, low, close, volume]

        Returns:
            (B, T, N, 64) price embeddings
        """
        B, T, N, seq_len, _ = ohlcv_seq.shape

        # Reshape: (B*T*N, seq_len, 5)
        x = ohlcv_seq.reshape(B * T * N, seq_len, 5)

        # Embedding: (B*T*N, seq_len, 5) -> (B*T*N, seq_len, d_model)
        x = self.embedding(x)

        # Positional encoding: (B*T*N, seq_len, d_model)
        x = self.positional(x)

        # Transformer: (B*T*N, seq_len, d_model)
        x = self.transformer(x)

        # Temporal pooling: mean over sequence
        # (B*T*N, seq_len, d_model) -> (B*T*N, d_model)
        x = x.mean(dim=1)

        # Reshape back: (B, T, N, d_model)
        x = x.reshape(B, T, N, self.d_model)

        return x


class HierarchicalFeatureEncoder(nn.Module):
    """
    Hierarchical Feature Encoder with separate pooled/unpooled outputs.

    This is the core architectural innovation:
    - Returns z_pooled (B, 176) for Meta Head and Critic
    - Returns z_unpooled (B, N, 176) for Portfolio Head

    The Portfolio Head receives per-asset features, preserving individual
    asset information for weight allocation.

    Pipeline:
    1. Alpha attention: (B, T, N, 292) -> (B, T, N, 64)
    2. Text projection: (B, T, N, 768) -> (B, T, N, 64) for news & social
    3. Price encoding: (B, T, N, 288, 5) -> (B, T, N, 64) via Transformer
    4. Concat: (B, T, N, 256) = 64 + 64 + 64 + 64
    5. Cross-asset attention: (B, T, N, 256)
    6. GRU per asset: (B, N, 128) temporal encoding
    7. Global/Portfolio MLPs: (B, 32), (B, 16)
    8. Split outputs:
       - z_pooled = [pool(h_assets), h_global, h_port] = (B, 176)
       - z_unpooled = [h_assets, broadcast(h_global), broadcast(h_port)] = (B, N, 176)
    """

    def __init__(
        self,
        n_alphas: int = 101,
        n_assets: int = 20,
        d_alpha: int = 64,
        d_text: int = 64,
        d_price: int = 64,
        d_temporal: int = 128,
        d_global: int = 32,
        d_portfolio: int = 16,
        n_alpha_heads: int = 8,
        n_asset_heads: int = 8,
        dropout: float = 0.1,
        # TERC (Transfer Entropy with Recursive Conditioning) parameters
        use_terc: bool = True,
        terc_tau: float = 0.15,
        terc_use_learned: bool = True,
    ):
        super().__init__()

        self.n_assets = n_assets
        self.d_temporal = d_temporal
        self.d_global = d_global
        self.d_portfolio = d_portfolio
        self.use_terc = use_terc

        # Gate activation storage for logging (Paper Line 758)
        self._last_gate_values = None
        self._last_terc_importance = None

        # Alpha processing
        self.alpha_attention = CrossAlphaAttention(
            n_alphas=n_alphas,
            d_model=d_alpha,
            n_heads=n_alpha_heads,
            dropout=dropout,
        )

        # TERC Filter for semantic embeddings (논문 Line 756: h^sem ← TERC-Filter)
        if use_terc:
            terc_config = TERCConfig(
                embedding_dim=768,
                tau=terc_tau,
                use_learned_importance=terc_use_learned,
            )
            self.terc_filter = TERCFilter(terc_config)
            # Semantic Gate (논문 Line 758: g ← σ(W_g z + b_g))
            # Gate input: h_alpha (64) + h_price (64) = 128
            self.semantic_gate = SemanticGate(
                numeric_dim=d_alpha + d_price,  # 128
                semantic_dim=d_text,  # 64
            )

        # Text processing
        self.news_proj = TextProjection(768, d_text, dropout)
        self.social_proj = TextProjection(768, d_text, dropout)

        # Price processing (Transformer encoder for OHLCV sequences)
        self.price_encoder = PriceTransformerEncoder(
            d_model=d_price,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=dropout,
        )

        # Cross-asset attention
        d_concat = d_alpha + d_text + d_text + d_price  # 64 + 64 + 64 + 64 = 256
        self.cross_asset_attn = CrossAssetAttention(
            d_model=d_concat,
            n_heads=n_asset_heads,
            dropout=dropout,
        )

        # Temporal processing (GRU per asset)
        self.gru = nn.GRU(
            input_size=d_concat,
            hidden_size=d_temporal,
            num_layers=1,
            batch_first=True,
        )

        # Global features MLP (23 macro indicators + 5 Fama-French factors = 28)
        self.global_mlp = nn.Sequential(
            nn.Linear(28, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_global),
        )

        # Portfolio state MLP (current weights, leverage, cash)
        self.portfolio_mlp = nn.Sequential(
            nn.Linear(22, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_portfolio),
        )

        # Output dimensions
        self.output_dim = d_temporal + d_global + d_portfolio  # 128 + 32 + 16 = 176

    def forward(
        self,
        state_dict: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode state into pooled and unpooled representations.

        Args:
            state_dict: Dictionary containing:
                # Local Features (per-asset, with N dimension):
                - 'alphas': (B, T, N, 101) raw alpha factors (Alpha101 only)
                - 'ohlcv_seq': (B, T, N, 288, 5) 5-minute OHLCV sequences
                - 'portfolio_state': (B, 22) current portfolio state

                # Global Features (market-wide, NO N dimension):
                - 'news_embedding': (B, T, 768) GDELT market-wide embeddings
                - 'social_embedding': (B, T, 768) Nostr market-wide embeddings
                - 'has_social_signal': (B, T, 1) social data availability
                - 'global_features': (B, T, 28) 23 macro + 5 Fama-French factors

        Returns:
            z_pooled: (B, 176) for Meta Head and Critic
            z_unpooled: (B, N, 176) for Portfolio Head
        """
        alphas = state_dict['alphas']
        news_emb = state_dict['news_embedding']         # (B, T, 768) - market-wide
        social_emb = state_dict['social_embedding']     # (B, T, 768) - market-wide
        has_social = state_dict.get('has_social_signal', None)  # (B, T, 1) - market-wide
        ohlcv_seq = state_dict.get('ohlcv_seq', None)
        global_feat = state_dict['global_features']
        portfolio = state_dict['portfolio_state']
        terc_target = state_dict.get('terc_target', None)  # Optional target for TE calc

        B, T, N, _ = alphas.shape

        # 1. Alpha attention: (B, T, N, 101) -> (B, T, N, 64)
        h_alpha = self.alpha_attention(alphas)

        # 2. TERC Filtering (논문 Line 756: h^sem ← TERC-Filter(h^sem, F_t, τ))
        # Apply before text projection to filter on 768-dim embeddings
        if self.use_terc:
            news_emb_filtered, _news_importance = self.terc_filter(
                news_emb, target=terc_target
            )
            social_emb_filtered, _social_importance = self.terc_filter(
                social_emb, target=terc_target
            )
        else:
            news_emb_filtered = news_emb
            social_emb_filtered = social_emb

        # 3. Text projection - Market-Wide: (B, T, 768) -> (B, T, 64)
        h_news_global = self.news_proj(news_emb_filtered)        # (B, T, 64)
        h_social_global = self.social_proj(social_emb_filtered, has_social)  # (B, T, 64)

        # 4. Price encoding: (B, T, N, 288, 5) -> (B, T, N, 64)
        if ohlcv_seq is not None:
            h_price = self.price_encoder(ohlcv_seq)
        else:
            # Fallback to zeros if OHLCV not provided (backward compatibility)
            h_price = torch.zeros(B, T, N, 64, device=alphas.device)

        # 5. Semantic Gate (논문 Line 758: g ← σ(W_g z + b_g))
        # Gate semantic features based on numeric context (alpha + price)
        if self.use_terc:
            # Combine alpha and price for gate input: (B, T, N, 128)
            h_numeric = torch.cat([h_alpha, h_price], dim=-1)

            # Broadcast text to per-asset: (B, T, 64) -> (B, T, N, 64)
            h_news_broadcast = h_news_global.unsqueeze(2).expand(B, T, N, -1)
            h_social_broadcast = h_social_global.unsqueeze(2).expand(B, T, N, -1)

            # Apply semantic gate (논문 Line 759: z̃ ← [h^num; g ⊙ h^sem; x^port])
            h_news, gate_news = self.semantic_gate(h_numeric, h_news_broadcast)
            h_social, gate_social = self.semantic_gate(h_numeric, h_social_broadcast)

            # Store gate activations for logging
            self._last_gate_values = (gate_news + gate_social) / 2  # Average of both gates
        else:
            # No gating: simple broadcast
            h_news = h_news_global.unsqueeze(2).expand(B, T, N, -1)
            h_social = h_social_global.unsqueeze(2).expand(B, T, N, -1)

        # 6. Concat per asset: (B, T, N, 256)
        h_concat = torch.cat([h_alpha, h_news, h_social, h_price], dim=-1)

        # 5. Cross-asset attention: (B, T, N, 256)
        h_cross = self.cross_asset_attn(h_concat)

        # 6. Temporal GRU per asset
        # Reshape: (B*N, T, 256)
        h_temporal_in = h_cross.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        h_temporal, _ = self.gru(h_temporal_in)
        # Take last timestep: (B*N, 128) -> (B, N, 128)
        h_assets = h_temporal[:, -1, :].reshape(B, N, self.d_temporal)

        # 6. Global features: (B, T, 6) -> (B, 6) -> (B, 32)
        h_global = self.global_mlp(global_feat[:, -1, :])

        # 7. Portfolio state: (B, 22) -> (B, 16)
        h_port = self.portfolio_mlp(portfolio)

        # ========== Key: Generate both pooled and unpooled outputs ==========

        # z_pooled: For Meta Head and Critic (B, 176)
        h_pooled = h_assets.mean(dim=1)  # (B, 128)
        z_pooled = torch.cat([h_pooled, h_global, h_port], dim=-1)  # (B, 176)

        # z_unpooled: For Portfolio Head (B, N, 176)
        # Broadcast global and portfolio features to match asset dimension
        h_global_exp = h_global.unsqueeze(1).expand(-1, N, -1)  # (B, N, 32)
        h_port_exp = h_port.unsqueeze(1).expand(-1, N, -1)      # (B, N, 16)
        z_unpooled = torch.cat([h_assets, h_global_exp, h_port_exp], dim=-1)  # (B, N, 176)

        return z_pooled, z_unpooled

    def get_gate_statistics(self) -> Dict[str, float]:
        """
        Get gate activation statistics for TensorBoard logging.

        Returns:
            Dict with gate statistics:
            - gate/mean_activation: Average gate value
            - gate/std_activation: Standard deviation of gate values
            - gate/active_ratio: Ratio of gates > 0.5 (actively passing semantic info)
        """
        if self._last_gate_values is None:
            return {}

        gate = self._last_gate_values.detach()

        # Flatten gate values for statistics
        gate_flat = gate.reshape(-1)

        return {
            'mean_activation': float(gate_flat.mean().item()),
            'std_activation': float(gate_flat.std().item()),
            'active_ratio': float((gate_flat > 0.5).float().mean().item()),
        }

    def get_terc_metrics(self) -> Dict[str, float]:
        """
        Get TERC filter metrics for TensorBoard logging.

        Returns:
            Dict with terc/* metrics if TERC is enabled, empty dict otherwise.
        """
        if not self.use_terc or not hasattr(self, 'terc_filter'):
            return {}

        return self.terc_filter.get_terc_metrics_for_tensorboard()

    def get_all_tensorboard_metrics(self) -> Dict[str, float]:
        """
        Get all encoder metrics for TensorBoard logging.

        Returns:
            Combined dict with gate/* and terc/* metrics.
        """
        metrics = {}

        # Gate statistics
        gate_stats = self.get_gate_statistics()
        for key, value in gate_stats.items():
            metrics[f'gate/{key}'] = value

        # TERC metrics
        terc_stats = self.get_terc_metrics()
        metrics.update(terc_stats)

        return metrics


class HierarchicalActor(nn.Module):
    """
    Simplified Hierarchical Actor - Lazy Agent Architecture.

    Key design principles:
    - NO Meta Head (no explicit trade/hold decisions)
    - Portfolio Head directly outputs target weights
    - Lazy behavior emerges from:
      1. Current portfolio weights in state (agent sees current positions)
      2. Transaction cost penalties in reward (agent learns to avoid churning)
      3. Deadband filter in PositionSizer (automatic 2% threshold)

    This avoids 2^N complexity of explicit trade/hold decisions while
    achieving the same lazy trading behavior through cost-aware learning.
    """

    def __init__(
        self,
        d_model: int = 176,
        n_assets: int = 20,
        portfolio_hidden_dims: Tuple[int, ...] = (128, 64),
    ):
        super().__init__()
        self.n_assets = n_assets

        # Portfolio Head: Per-asset weight decision
        # Input: z_unpooled (B, N, 176) - processes each asset independently
        # Output: weights (B, N) in [-1, 1]
        portfolio_layers = []
        prev_dim = d_model
        for hidden_dim in portfolio_hidden_dims:
            portfolio_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ])
            prev_dim = hidden_dim
        portfolio_layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Tanh(),
        ])
        self.portfolio_head = nn.Sequential(*portfolio_layers)

        # Initialize final layer with small weights for stable early training
        final_linear = self.portfolio_head[-2]
        if isinstance(final_linear, nn.Linear):
            nn.init.uniform_(final_linear.weight, -0.003, 0.003)
            nn.init.zeros_(final_linear.bias)

    def forward(
        self,
        z_pooled: torch.Tensor,
        z_unpooled: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute portfolio weights (Lazy Agent - no explicit trade/hold decision).

        Args:
            z_pooled: (B, 176) pooled features (unused, kept for interface compatibility)
            z_unpooled: (B, N, 176) per-asset features

        Returns:
            weights: (B, N) portfolio weights in [-1, 1]
        """
        # Portfolio weights (per-asset)
        # (B, N, 176) -> (B, N, 1) -> (B, N)
        weights = self.portfolio_head(z_unpooled).squeeze(-1)  # (B, N)

        return weights

    def get_action(
        self,
        z_pooled: torch.Tensor,
        z_unpooled: torch.Tensor,
        deterministic: bool = False,
        portfolio_noise_scale: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action with optional exploration noise (Lazy Agent).

        Args:
            z_pooled: (B, 176) pooled features
            z_unpooled: (B, N, 176) per-asset features
            deterministic: If True, return deterministic action without noise
            portfolio_noise_scale: Scale of exploration noise

        Returns:
            weights: (B, N) portfolio weights
            log_prob: (B,) log probability for PPO
        """
        weights = self.forward(z_pooled, z_unpooled)

        if deterministic:
            log_prob = torch.zeros(z_pooled.shape[0], device=z_pooled.device)
            return weights, log_prob

        # Add exploration noise to portfolio weights
        noise = torch.randn_like(weights) * portfolio_noise_scale
        noisy_weights = torch.tanh(
            torch.atanh(weights.clamp(-0.999, 0.999)) + noise
        )

        # Simple log probability approximation (Gaussian in logit space)
        log_prob = -0.5 * (noise ** 2).sum(dim=-1)

        return noisy_weights, log_prob

    def evaluate_actions(
        self,
        z_pooled: torch.Tensor,
        z_unpooled: torch.Tensor,
        old_actions: torch.Tensor,
        portfolio_noise_scale: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability of given actions under current policy.

        This is needed for PPO to compute the ratio π_θ(a|s) / π_θ_old(a|s)
        where we need π_θ(a|s) for the OLD actions under the CURRENT policy.

        Args:
            z_pooled: (B, 176) pooled features
            z_unpooled: (B, N, 176) per-asset features
            old_actions: (B, N) actions to evaluate (from buffer)
            portfolio_noise_scale: Scale of exploration noise (must match collection)

        Returns:
            log_prob: (B,) log probability of old_actions under current policy
            entropy: (B,) entropy of the policy (for entropy bonus)
        """
        # Get current policy mean (deterministic output)
        mean_weights = self.forward(z_pooled, z_unpooled)

        # Convert actions and means to logit space
        # Clamp to avoid numerical issues at boundaries
        old_logits = torch.atanh(old_actions.clamp(-0.999, 0.999))
        mean_logits = torch.atanh(mean_weights.clamp(-0.999, 0.999))

        # Compute implied noise (what noise would have produced old_actions from current mean)
        implied_noise = (old_logits - mean_logits) / portfolio_noise_scale

        # Log probability under Gaussian: -0.5 * ||noise||^2 - 0.5 * N * log(2π) - N * log(σ)
        # Simplified (ignoring constants): -0.5 * ||noise||^2
        log_prob = -0.5 * (implied_noise ** 2).sum(dim=-1)

        # Entropy of Gaussian: 0.5 * N * (1 + log(2π)) + N * log(σ)
        # Simplified approximation
        n_actions = old_actions.shape[-1]
        entropy = 0.5 * n_actions * (1 + 2.837)  # 2.837 ≈ log(2π)
        entropy = torch.full((old_actions.shape[0],), entropy, device=old_actions.device)

        return log_prob, entropy


class HierarchicalCritic(nn.Module):
    """
    Critic for Hierarchical policy.

    Uses pooled features only, as value is a function of the global state.
    Includes optional CVaR quantile estimation for risk-sensitive RL.
    """

    def __init__(
        self,
        d_model: int = 176,
        hidden_dims: Tuple[int, ...] = (256, 128),
        n_quantiles: int = 32,
    ):
        super().__init__()

        # Value head
        value_layers = []
        prev_dim = d_model
        for hidden_dim in hidden_dims:
            value_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ])
            prev_dim = hidden_dim
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_head = nn.Sequential(*value_layers)

        # Quantile head for CVaR estimation
        self.quantile_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, n_quantiles),
        )

        # Initialize
        nn.init.uniform_(self.value_head[-1].weight, -0.003, 0.003)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(
        self,
        z_pooled: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute value and quantiles.

        Args:
            z_pooled: (B, 176) pooled features

        Returns:
            value: (B, 1) state value
            quantiles: (B, n_quantiles) for CVaR computation
        """
        value = self.value_head(z_pooled)
        quantiles = self.quantile_head(z_pooled)
        return value, quantiles


class HierarchicalActorCritic(nn.Module):
    """
    Combined Hierarchical Actor-Critic with shared encoder.

    This is the main model for STAIR-RL training.
    """

    def __init__(
        self,
        n_alphas: int = 101,
        n_assets: int = 20,
        d_alpha: int = 64,
        d_text: int = 64,
        d_temporal: int = 128,
        d_global: int = 32,
        d_portfolio: int = 16,
        actor_hidden_dims: Tuple[int, ...] = (128, 64),
        critic_hidden_dims: Tuple[int, ...] = (256, 128),
        n_quantiles: int = 32,
        dropout: float = 0.1,
        # TERC (Transfer Entropy with Recursive Conditioning) - Paper Line 756
        use_terc: bool = False,
        terc_tau: float = 0.15,
        terc_use_learned: bool = True,
    ):
        super().__init__()

        self.n_assets = n_assets
        self.use_terc = use_terc

        # Shared encoder
        self.encoder = HierarchicalFeatureEncoder(
            n_alphas=n_alphas,
            n_assets=n_assets,
            d_alpha=d_alpha,
            d_text=d_text,
            d_temporal=d_temporal,
            d_global=d_global,
            d_portfolio=d_portfolio,
            dropout=dropout,
            use_terc=use_terc,
            terc_tau=terc_tau,
            terc_use_learned=terc_use_learned,
        )

        d_model = self.encoder.output_dim  # 176

        # Hierarchical actor (Lazy Agent - no Meta Head)
        self.actor = HierarchicalActor(
            d_model=d_model,
            n_assets=n_assets,
            portfolio_hidden_dims=actor_hidden_dims,
        )

        # Critic
        self.critic = HierarchicalCritic(
            d_model=d_model,
            hidden_dims=critic_hidden_dims,
            n_quantiles=n_quantiles,
        )

    def forward(
        self,
        state_dict: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass (Lazy Agent).

        Args:
            state_dict: State dictionary (see HierarchicalFeatureEncoder)

        Returns:
            weights: (B, N) per-asset portfolio weights
            value: (B, 1) state value
            quantiles: (B, n_quantiles) CVaR quantiles
        """
        z_pooled, z_unpooled = self.encoder(state_dict)
        weights = self.actor(z_pooled, z_unpooled)
        value, quantiles = self.critic(z_pooled)
        return weights, value, quantiles

    def get_action_and_value(
        self,
        state_dict: dict,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value (Lazy Agent).

        Returns:
            weights: (B, N) per-asset portfolio weights
            log_prob: (B,) log probability
            value: (B, 1) state value
        """
        z_pooled, z_unpooled = self.encoder(state_dict)
        weights, log_prob = self.actor.get_action(
            z_pooled, z_unpooled, deterministic=deterministic
        )
        value, _ = self.critic(z_pooled)
        return weights, log_prob, value

    def get_value(self, state_dict: dict) -> torch.Tensor:
        """Get value estimate only."""
        z_pooled, _ = self.encoder(state_dict)
        value, _ = self.critic(z_pooled)
        return value.squeeze(-1)

    def get_gate_statistics(self) -> Dict[str, float]:
        """
        Get gate activation statistics for TensorBoard logging.

        Delegates to encoder's gate statistics.
        """
        return self.encoder.get_gate_statistics()

    def get_all_tensorboard_metrics(self) -> Dict[str, float]:
        """
        Get all model metrics for TensorBoard logging.

        Delegates to encoder's combined tensorboard metrics.
        Returns metrics prefixed with gate/ and terc/.
        """
        return self.encoder.get_all_tensorboard_metrics()


# ========== Standalone Testing ==========

if __name__ == '__main__':
    # Test dimensions
    batch_size = 32
    n_assets = 20
    total_features = 35
    portfolio_dim = 22

    # Create dummy inputs
    market_state = torch.randn(batch_size, n_assets, total_features)
    portfolio_state = torch.randn(batch_size, portfolio_dim)

    # Test FeatureEncoder
    encoder = FeatureEncoder(n_assets=n_assets)
    z = encoder(market_state, portfolio_state)
    print(f"FeatureEncoder output shape: {z.shape}")  # Should be (32, 176)
    print(f"FeatureEncoder parameters: {count_parameters(encoder):,}")

    # Test Actor
    actor = Actor(input_dim=encoder.output_dim, n_assets=n_assets)
    action = actor(z)
    print(f"Actor output shape: {action.shape}")  # Should be (32, 20)
    print(f"Actor output range: [{action.min():.3f}, {action.max():.3f}]")
    print(f"Actor parameters: {count_parameters(actor):,}")

    # Test Critic (Q-function)
    q_critic = Critic(state_dim=encoder.output_dim, action_dim=n_assets, q_function=True)
    q_value = q_critic(z, action)
    print(f"Q-Critic output shape: {q_value.shape}")  # Should be (32, 1)
    print(f"Q-Critic parameters: {count_parameters(q_critic):,}")

    # Test Critic (V-function)
    v_critic = Critic(state_dim=encoder.output_dim, q_function=False)
    v_value = v_critic(z)
    print(f"V-Critic output shape: {v_value.shape}")  # Should be (32, 1)
    print(f"V-Critic parameters: {count_parameters(v_critic):,}")

    # Test ActorCritic
    ac = ActorCritic(n_assets=n_assets)
    action, value = ac(market_state, portfolio_state)
    print(f"\nActorCritic:")
    print(f"  Action shape: {action.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Total parameters: {count_parameters(ac):,}")

    # Test TwinCritic
    twin_q = TwinCritic(state_dim=encoder.output_dim, action_dim=n_assets)
    q1, q2 = twin_q(z, action)
    print(f"\nTwinCritic:")
    print(f"  Q1 shape: {q1.shape}")
    print(f"  Q2 shape: {q2.shape}")
    print(f"  Parameters: {count_parameters(twin_q):,}")

    # ========== Test Hierarchical Architecture (v2) ==========
    print("\n" + "=" * 60)
    print("Testing Hierarchical Architecture (v2)")
    print("=" * 60)

    # Test parameters
    B = 4  # batch size
    T = 24  # lookback (2 hours @ 5-min)
    N = 20  # assets
    n_alphas = 101  # Alpha101 only

    # Create dummy state dictionary
    state_dict = {
        'alphas': torch.randn(B, T, N, n_alphas),
        'news_embedding': torch.randn(B, T, 768),  # Market-wide (NO N)
        'social_embedding': torch.randn(B, T, 768),  # Market-wide (NO N)
        'has_social_signal': torch.ones(B, T, 1),  # Market-wide (NO N)
        'global_features': torch.randn(B, T, 6),
        'portfolio_state': torch.randn(B, 22),
    }

    # Test CrossAlphaAttention
    print("\n1. CrossAlphaAttention:")
    alpha_attn = CrossAlphaAttention(n_alphas=n_alphas, d_model=64)
    h_alpha = alpha_attn(state_dict['alphas'])
    print(f"   Input: {state_dict['alphas'].shape}")
    print(f"   Output: {h_alpha.shape}")
    print(f"   Parameters: {count_parameters(alpha_attn):,}")
    assert h_alpha.shape == (B, T, N, 64), "CrossAlphaAttention output shape mismatch!"

    # Test CrossAssetAttention
    print("\n2. CrossAssetAttention:")
    asset_attn = CrossAssetAttention(d_model=192)
    h_concat = torch.randn(B, T, N, 192)
    h_cross = asset_attn(h_concat)
    print(f"   Input: {h_concat.shape}")
    print(f"   Output: {h_cross.shape}")
    print(f"   Parameters: {count_parameters(asset_attn):,}")
    assert h_cross.shape == (B, T, N, 192), "CrossAssetAttention output shape mismatch!"

    # Test TextProjection
    print("\n3. TextProjection:")
    text_proj = TextProjection(bert_dim=768, output_dim=64)
    h_text = text_proj(state_dict['news_embedding'])
    print(f"   Input: {state_dict['news_embedding'].shape}")
    print(f"   Output: {h_text.shape}")
    print(f"   Parameters: {count_parameters(text_proj):,}")
    assert h_text.shape == (B, T, N, 64), "TextProjection output shape mismatch!"

    # Test HierarchicalFeatureEncoder
    print("\n4. HierarchicalFeatureEncoder:")
    h_encoder = HierarchicalFeatureEncoder(n_alphas=n_alphas)
    z_pooled, z_unpooled = h_encoder(state_dict)
    print(f"   z_pooled shape: {z_pooled.shape} (for Meta Head, Critic)")
    print(f"   z_unpooled shape: {z_unpooled.shape} (for Portfolio Head)")
    print(f"   Parameters: {count_parameters(h_encoder):,}")
    assert z_pooled.shape == (B, 176), "z_pooled shape mismatch!"
    assert z_unpooled.shape == (B, N, 176), "z_unpooled shape mismatch!"

    # Test HierarchicalActor (Lazy Agent)
    print("\n5. HierarchicalActor (Lazy Agent):")
    h_actor = HierarchicalActor(d_model=176, n_assets=N)
    weights = h_actor(z_pooled, z_unpooled)
    print(f"   weights shape: {weights.shape}, range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"   Parameters: {count_parameters(h_actor):,}")
    assert weights.shape == (B, N), "weights shape mismatch!"
    assert (weights >= -1).all() and (weights <= 1).all(), "weights out of range!"

    # Test HierarchicalCritic
    print("\n6. HierarchicalCritic:")
    h_critic = HierarchicalCritic(d_model=176)
    value, quantiles = h_critic(z_pooled)
    print(f"   value shape: {value.shape}")
    print(f"   quantiles shape: {quantiles.shape}")
    print(f"   Parameters: {count_parameters(h_critic):,}")
    assert value.shape == (B, 1), "value shape mismatch!"
    assert quantiles.shape == (B, 32), "quantiles shape mismatch!"

    # Test HierarchicalActorCritic (full model - Lazy Agent)
    print("\n7. HierarchicalActorCritic (Full Model - Lazy Agent):")
    hac = HierarchicalActorCritic(n_alphas=n_alphas, n_assets=N)
    weights, value, quantiles = hac(state_dict)
    print(f"   weights: {weights.shape}")
    print(f"   value: {value.shape}")
    print(f"   quantiles: {quantiles.shape}")
    print(f"   Total parameters: {count_parameters(hac):,}")
    assert weights.shape == (B, N), "weights shape mismatch!"

    # Verify per-asset differentiation (key test!)
    print("\n8. Portfolio Head Per-Asset Differentiation Test:")
    # Create state where assets have different features
    state_diff = {
        'alphas': torch.zeros(B, T, N, n_alphas),
        'news_embedding': torch.zeros(B, T, 768),  # Market-wide (NO N)
        'social_embedding': torch.zeros(B, T, 768),  # Market-wide (NO N)
        'has_social_signal': torch.ones(B, T, 1),  # Market-wide (NO N)
        'global_features': torch.zeros(B, T, 6),
        'portfolio_state': torch.zeros(B, 22),
    }
    # Make one asset distinctive
    state_diff['alphas'][:, :, 5, :] = 10.0  # Asset 5 has high alpha values
    state_diff['alphas'][:, :, 15, :] = -10.0  # Asset 15 has low alpha values

    weights_diff = hac.actor(*hac.encoder(state_diff))
    weight_std = weights_diff.std(dim=1).mean()  # Std across assets
    print(f"   Weight std across assets: {weight_std:.4f}")
    print(f"   Asset 5 weight: {weights_diff[0, 5]:.4f}")
    print(f"   Asset 15 weight: {weights_diff[0, 15]:.4f}")
    if weight_std > 0.01:
        print("   ✓ Portfolio Head differentiates between assets!")
    else:
        print("   ✗ Warning: Portfolio Head may not be differentiating assets properly")

    print("\n" + "=" * 60)
    print("All Hierarchical Architecture tests passed!")
    print("=" * 60)
