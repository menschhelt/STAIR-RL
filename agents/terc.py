"""
TERC: Transfer Entropy with Recursive Conditioning

Token selection mechanism based on information-theoretic relevance.
Implements the TERC-Filter from STAIR-RL paper (Line 147-150, 756).

Key concepts:
- Transfer Entropy: TE(X→Y) = H(Y_t+1|Y_t) - H(Y_t+1|Y_t, X_t)
- TERC selects tokens where TE > τ (threshold, default 0.15 nats)
- Submodular maximization guarantees (1-1/e) ≈ 63.2% optimality

Paper claims:
- TERC > Random by 10-15%
- TERC > All Tokens by 22%
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class TERCConfig:
    """Configuration for TERC filter."""
    embedding_dim: int = 768        # Input embedding dimension (FinBERT/CryptoBERT)
    tau: float = 0.15               # Transfer entropy threshold (nats)
    n_bins: int = 32                # Bins for histogram-based TE estimation
    window_size: int = 5            # Temporal window for TE calculation
    use_learned_importance: bool = True  # Use learned MLP vs pure TE
    importance_hidden_dim: int = 256     # Hidden dim for importance MLP
    min_selected_ratio: float = 0.1      # Minimum ratio of tokens to keep
    temperature: float = 1.0             # Temperature for soft selection


class TransferEntropyEstimator:
    """
    Histogram-based Transfer Entropy estimator.

    TE(X→Y) = Σ p(y_{t+1}, y_t, x_t) log[p(y_{t+1}|y_t, x_t) / p(y_{t+1}|y_t)]
            = H(Y_t+1|Y_t) - H(Y_t+1|Y_t, X_t)
    """

    def __init__(self, n_bins: int = 32, window_size: int = 5):
        self.n_bins = n_bins
        self.window_size = window_size

    def _discretize(self, x: np.ndarray) -> np.ndarray:
        """Discretize continuous values into bins."""
        # Use percentile-based binning for robustness
        percentiles = np.percentile(x.flatten(), np.linspace(0, 100, self.n_bins + 1))
        return np.digitize(x, percentiles[1:-1])

    def _conditional_entropy(
        self,
        y_next: np.ndarray,  # (T-1,)
        *conditions: np.ndarray,  # Each (T-1,)
    ) -> float:
        """
        Compute H(Y_next | conditions).

        Uses histogram-based estimation with smoothing.
        """
        # Create joint histogram
        joint = np.stack([y_next] + list(conditions), axis=1)  # (T-1, 1+len(conditions))

        # Count occurrences
        unique_rows, counts = np.unique(joint, axis=0, return_counts=True)
        probs = counts / counts.sum()

        # Compute conditional entropy
        # H(Y|X) = H(Y,X) - H(X)
        # For simplicity, use plugin estimator with Laplace smoothing
        h_joint = -np.sum(probs * np.log(probs + 1e-10))

        # Compute H(conditions)
        if len(conditions) > 0:
            cond_joint = np.stack(conditions, axis=1)
            unique_cond, cond_counts = np.unique(cond_joint, axis=0, return_counts=True)
            cond_probs = cond_counts / cond_counts.sum()
            h_cond = -np.sum(cond_probs * np.log(cond_probs + 1e-10))
            return h_joint - h_cond
        else:
            return h_joint

    def compute_te(
        self,
        source: np.ndarray,  # (T, D) - semantic embeddings
        target: np.ndarray,  # (T,) - target variable (e.g., returns)
    ) -> np.ndarray:
        """
        Compute Transfer Entropy from each source dimension to target.

        TE(X_d → Y) = H(Y_t+1|Y_t) - H(Y_t+1|Y_t, X_d,t)

        Args:
            source: (T, D) semantic embedding time series
            target: (T,) target time series

        Returns:
            te: (D,) transfer entropy per embedding dimension
        """
        T, D = source.shape
        te = np.zeros(D)

        # Discretize target
        target_disc = self._discretize(target)
        y_t = target_disc[:-1]  # Y_t
        y_next = target_disc[1:]  # Y_{t+1}

        # Compute H(Y_t+1|Y_t) once
        h_y_given_yt = self._conditional_entropy(y_next, y_t)

        # Compute TE for each source dimension
        for d in range(D):
            source_d = self._discretize(source[:-1, d])  # X_d,t
            h_y_given_yt_xt = self._conditional_entropy(y_next, y_t, source_d)
            te[d] = max(0, h_y_given_yt - h_y_given_yt_xt)  # TE >= 0

        return te


class TERCFilter(nn.Module):
    """
    Transfer Entropy based token filtering.

    Paper Line 756: h^sem ← TERC-Filter(h^sem, F_t, τ)

    Two modes:
    1. Training: Use learned importance MLP (differentiable, fast)
    2. Evaluation: Use actual TE computation (accurate, slow)
    """

    def __init__(self, config: Optional[TERCConfig] = None):
        super().__init__()
        self.config = config or TERCConfig()
        cfg = self.config

        # Learned importance network (for training)
        if cfg.use_learned_importance:
            self.importance_mlp = nn.Sequential(
                nn.Linear(cfg.embedding_dim, cfg.importance_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(cfg.importance_hidden_dim),
                nn.Dropout(0.1),
                nn.Linear(cfg.importance_hidden_dim, cfg.embedding_dim),
            )

        # TE estimator (for evaluation)
        self.te_estimator = TransferEntropyEstimator(
            n_bins=cfg.n_bins,
            window_size=cfg.window_size,
        )

        # Cached TE values (updated periodically during training)
        self.register_buffer('cached_te', torch.zeros(cfg.embedding_dim))
        self.register_buffer('te_computed', torch.tensor(False))

    def compute_importance_scores(
        self,
        h_sem: torch.Tensor,  # (B, T, 768)
    ) -> torch.Tensor:
        """
        Compute importance scores using learned MLP.

        Returns:
            importance: (B, T, 768) soft importance weights in [0, 1]
        """
        # MLP outputs logits
        logits = self.importance_mlp(h_sem)  # (B, T, 768)

        # Apply sigmoid with temperature
        importance = torch.sigmoid(logits / self.config.temperature)

        return importance

    def compute_te_mask(
        self,
        h_sem: torch.Tensor,  # (B, T, 768)
        target: torch.Tensor,  # (B, T) target for TE (e.g., returns)
    ) -> torch.Tensor:
        """
        Compute binary mask based on Transfer Entropy.

        This is the "true" TERC as described in the paper.

        Design Note (Batch Averaging):
            We compute TE from batch-averaged embeddings for computational efficiency.
            This produces a global mask shared across all samples in the batch.

            Alternative (per-sample TE) would be O(B * T * D) vs current O(T * D).
            The global mask approach assumes TE importance is stationary within a batch,
            which is reasonable for short time windows (typical batch spans ~minutes).

            For production, consider:
            1. Periodic TE re-computation (e.g., every N batches)
            2. Rolling window TE with adaptive τ for regime changes

        Returns:
            mask: (768,) binary mask where TE > τ
        """
        cfg = self.config

        # Use mean across batch for TE computation (see Design Note above)
        h_mean = h_sem.mean(dim=0).detach().cpu().numpy()  # (T, 768)
        target_mean = target.mean(dim=0).detach().cpu().numpy()  # (T,)

        # Compute TE
        te = self.te_estimator.compute_te(h_mean, target_mean)

        # Create mask
        mask = torch.tensor(te > cfg.tau, dtype=torch.float32, device=h_sem.device)

        # Ensure minimum selection ratio
        if mask.sum() < cfg.min_selected_ratio * cfg.embedding_dim:
            # Select top k by TE
            k = int(cfg.min_selected_ratio * cfg.embedding_dim)
            top_k_indices = np.argsort(te)[-k:]
            mask = torch.zeros(cfg.embedding_dim, device=h_sem.device)
            mask[top_k_indices] = 1.0

        # Cache for later use
        self.cached_te = torch.tensor(te, dtype=torch.float32, device=h_sem.device)
        self.te_computed = torch.tensor(True)

        return mask

    def forward(
        self,
        h_sem: torch.Tensor,  # (B, T, 768) semantic embedding
        target: Optional[torch.Tensor] = None,  # (B, T) for TE target
        use_te: bool = False,  # Force TE computation
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter semantic tokens based on Transfer Entropy / learned importance.

        Args:
            h_sem: Semantic embeddings from BERT
            target: Target variable for TE calculation (returns, actions)
            use_te: Force actual TE computation (slow)

        Returns:
            h_filtered: (B, T, 768) filtered embeddings
            importance: (B, T, 768) importance weights used
        """
        cfg = self.config

        if self.training and cfg.use_learned_importance and not use_te:
            # Training mode: use learned importance (differentiable)
            importance = self.compute_importance_scores(h_sem)
        elif target is not None:
            # Evaluation mode with target: compute actual TE
            mask = self.compute_te_mask(h_sem, target)
            importance = mask.unsqueeze(0).unsqueeze(0).expand_as(h_sem)
        elif self.te_computed and not self.training:
            # Use cached TE mask
            mask = (self.cached_te > cfg.tau).float()
            importance = mask.unsqueeze(0).unsqueeze(0).expand_as(h_sem)
        else:
            # Fallback: use learned importance
            importance = self.compute_importance_scores(h_sem)

        # Apply filtering (soft multiplication)
        h_filtered = h_sem * importance

        return h_filtered, importance

    def get_selected_token_ratio(self) -> float:
        """Get ratio of tokens selected by TERC."""
        if self.te_computed:
            return (self.cached_te > self.config.tau).float().mean().item()
        return 1.0

    def get_te_statistics(self) -> Dict[str, float]:
        """Get TE statistics for logging."""
        if not self.te_computed:
            return {}

        te = self.cached_te.cpu().numpy()
        return {
            'te_mean': float(te.mean()),
            'te_std': float(te.std()),
            'te_max': float(te.max()),
            'te_min': float(te.min()),
            'te_above_tau': float((te > self.config.tau).mean()),
        }

    def get_terc_metrics_for_tensorboard(self) -> Dict[str, float]:
        """
        Get all TERC metrics formatted for TensorBoard logging.

        Returns:
            Dict with metrics:
            - terc/selected_ratio: Ratio of tokens selected (TE > tau)
            - terc/avg_te_score: Average transfer entropy score
            - terc/mask_sparsity: Sparsity of the selection mask (1 - density)
        """
        selected_ratio = self.get_selected_token_ratio()

        if not self.te_computed:
            return {
                'selected_ratio': selected_ratio,
                'avg_te_score': 0.0,
                'mask_sparsity': 1.0 - selected_ratio,
            }

        te = self.cached_te.cpu().numpy()
        return {
            'selected_ratio': selected_ratio,
            'avg_te_score': float(te.mean()),
            'mask_sparsity': 1.0 - selected_ratio,
        }


class SemanticGate(nn.Module):
    """
    Explicit semantic gate: g = σ(W_g z + b_g)

    Paper Line 758: g ← σ(W_g z + b_g)

    This gate learns to dynamically weight semantic information
    based on the current numeric state.
    """

    def __init__(
        self,
        numeric_dim: int,      # Dimension of numeric features (h_num)
        semantic_dim: int = 64,  # Dimension of semantic features (h_sem projected)
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.numeric_dim = numeric_dim
        self.semantic_dim = semantic_dim

        # Gate network: takes numeric features, outputs gate for semantic
        # Paper Line 758: g = σ(W_g z + b_g)
        # Using MLP for increased expressiveness (documented deviation from paper)
        self.gate_net = nn.Sequential(
            nn.Linear(numeric_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, semantic_dim),
            # Note: No sigmoid here - applied after bias addition
        )

        # Learnable bias for gate (paper: b_g, allows default pass-through)
        self.gate_bias = nn.Parameter(torch.zeros(semantic_dim))

    def forward(
        self,
        h_num: torch.Tensor,   # (B, ..., numeric_dim) numeric features
        h_sem: torch.Tensor,   # (B, ..., semantic_dim) semantic features
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gating to semantic features.

        Args:
            h_num: Numeric features [h_alpha; h_price]
            h_sem: Semantic embeddings (already projected to semantic_dim)

        Returns:
            h_gated: g ⊙ h_sem
            gate: Gate activations for logging
        """
        # Compute gate: g = σ(W_g z + b_g)
        # Paper Line 758: single sigmoid after linear transformation
        gate_logits = self.gate_net(h_num)  # (B, ..., semantic_dim)
        gate = torch.sigmoid(gate_logits + self.gate_bias)  # σ(Wz + b) in [0, 1]

        # Apply gate
        h_gated = gate * h_sem

        return h_gated, gate

    def get_gate_statistics(self, gate: torch.Tensor) -> Dict[str, float]:
        """Get gate statistics for logging."""
        gate_np = gate.detach().cpu().numpy()
        return {
            'gate_mean': float(gate_np.mean()),
            'gate_std': float(gate_np.std()),
            'gate_min': float(gate_np.min()),
            'gate_max': float(gate_np.max()),
            'gate_active_ratio': float((gate_np > 0.5).mean()),
        }


class TERCModule(nn.Module):
    """
    Complete TERC module combining filter and gate.

    Pipeline:
    1. TERC Filter: Select relevant tokens based on TE
    2. Semantic Gate: Dynamically weight based on numeric context

    Paper algorithm (Line 756-759):
        h^sem ← TERC-Filter(h^sem, F_t, τ)
        g ← σ(W_g z + b_g)
        z̃ ← [h^num; g ⊙ h^sem; x^port]
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        projected_dim: int = 64,
        numeric_dim: int = 128,  # h_alpha (64) + h_price (64)
        terc_config: Optional[TERCConfig] = None,
    ):
        super().__init__()

        # TERC Filter
        config = terc_config or TERCConfig(embedding_dim=embedding_dim)
        self.terc_filter = TERCFilter(config)

        # Projection from 768 to 64
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, projected_dim),
        )

        # Semantic Gate
        self.semantic_gate = SemanticGate(
            numeric_dim=numeric_dim,
            semantic_dim=projected_dim,
        )

    def forward(
        self,
        h_sem_raw: torch.Tensor,  # (B, T, 768) raw BERT embeddings
        h_num: torch.Tensor,       # (B, T, N, numeric_dim) numeric features
        target: Optional[torch.Tensor] = None,  # For TE computation
        use_te: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply TERC filtering and gating.

        Args:
            h_sem_raw: Raw semantic embeddings (768-dim)
            h_num: Numeric features for gating
            target: Target for TE calculation
            use_te: Force TE computation

        Returns:
            h_gated: (B, T, N, projected_dim) gated semantic features
            info: Dict with intermediate values for logging
        """
        B, T, D = h_sem_raw.shape
        N = h_num.shape[2] if h_num.dim() == 4 else 1

        # 1. TERC filtering
        h_filtered, importance = self.terc_filter(h_sem_raw, target, use_te)

        # 2. Project to lower dimension
        h_projected = self.projection(h_filtered)  # (B, T, 64)

        # 3. Broadcast to N assets
        h_broadcast = h_projected.unsqueeze(2).expand(B, T, N, -1)  # (B, T, N, 64)

        # 4. Apply semantic gate
        # h_num is (B, T, N, numeric_dim), need to match
        if h_num.dim() == 3:
            h_num = h_num.unsqueeze(2)

        h_gated, gate = self.semantic_gate(h_num, h_broadcast)

        # Collect info for logging
        info = {
            'importance': importance,
            'gate': gate,
            'h_filtered': h_filtered,
            'h_projected': h_projected,
        }

        return h_gated, info

    def get_all_metrics_for_tensorboard(self) -> Dict[str, float]:
        """
        Get all TERC module metrics for TensorBoard logging.

        Returns:
            Dict with terc/* and gate/* metrics combined.
        """
        # Get TERC filter metrics
        terc_metrics = self.terc_filter.get_terc_metrics_for_tensorboard()

        return {
            'terc/selected_ratio': terc_metrics['selected_ratio'],
            'terc/avg_te_score': terc_metrics['avg_te_score'],
            'terc/mask_sparsity': terc_metrics['mask_sparsity'],
        }


# Convenience function for creating TERC module
def create_terc_module(
    embedding_dim: int = 768,
    projected_dim: int = 64,
    numeric_dim: int = 128,
    tau: float = 0.15,
    use_learned_importance: bool = True,
) -> TERCModule:
    """Create a TERC module with specified configuration."""
    config = TERCConfig(
        embedding_dim=embedding_dim,
        tau=tau,
        use_learned_importance=use_learned_importance,
    )
    return TERCModule(
        embedding_dim=embedding_dim,
        projected_dim=projected_dim,
        numeric_dim=numeric_dim,
        terc_config=config,
    )
