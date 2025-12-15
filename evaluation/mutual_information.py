"""
Mutual Information Estimation for √I Scaling Validation (Theorem 1).

Theorem 1 (Value of Semantic Information):
    V*_M'(S_t) - V*_M(F_t) ≥ C_V · √I(A*_t; H*_t | F_t)

This module provides:
1. MINE (Mutual Information Neural Estimation) - Belghazi et al., 2018
2. InfoNCE bound - van den Oord et al., 2018
3. Validation of √I scaling via linear regression

Paper claims:
- ΔV ∝ √I (linear relationship between value improvement and √MI)
- Semantic information provides guaranteed value improvement
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class MIEstimationResult:
    """Result of mutual information estimation."""
    mi_estimate: float           # Estimated mutual information (nats)
    mi_std: float               # Standard deviation (if available)
    method: str                 # Estimation method used
    n_samples: int              # Number of samples used


class MINEEstimator(nn.Module):
    """
    MINE: Mutual Information Neural Estimation.

    Based on: Belghazi et al., "Mutual Information Neural Estimation", ICML 2018

    I(X;Y) ≥ E_P[T(x,y)] - log(E_Q[exp(T(x,y'))])

    where T is a statistics network and P is joint, Q is marginal (y' independent).

    This provides a lower bound on I(X;Y) that can be optimized.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        """
        Initialize MINE estimator.

        Args:
            x_dim: Dimension of X (e.g., actions)
            y_dim: Dimension of Y (e.g., semantic embeddings)
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
        """
        super().__init__()

        layers = []
        input_dim = x_dim + y_dim

        for i in range(n_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        layers.append(nn.Linear(hidden_dim, 1))

        self.statistics_net = nn.Sequential(*layers)

        # Exponential moving average for stable training
        self.register_buffer('ema', torch.tensor(1.0))
        self.ema_decay = 0.99

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        update_ema: bool = True,
    ) -> torch.Tensor:
        """
        Compute MINE lower bound on I(X;Y).

        Args:
            x: (B, D_x) - e.g., actions or policy outputs
            y: (B, D_y) - e.g., semantic embeddings
            update_ema: Whether to update EMA (during training)

        Returns:
            mi_estimate: Scalar estimate of I(X;Y)
        """
        batch_size = x.shape[0]

        # Joint samples: (x_i, y_i)
        joint = torch.cat([x, y], dim=-1)
        t_joint = self.statistics_net(joint).squeeze(-1)  # (B,)

        # Marginal samples: (x_i, y_j) where j ≠ i (shuffle y)
        idx = torch.randperm(batch_size, device=y.device)
        y_shuffled = y[idx]
        marginal = torch.cat([x, y_shuffled], dim=-1)
        t_marginal = self.statistics_net(marginal).squeeze(-1)  # (B,)

        # MINE bound with EMA for stability
        # I(X;Y) ≥ E[T(x,y)] - log(E[exp(T(x,y'))])
        joint_term = t_joint.mean()

        # Use logsumexp for numerical stability
        # log(mean(exp(t))) = logsumexp(t) - log(N)
        marginal_term = torch.logsumexp(t_marginal, dim=0) - np.log(batch_size)

        # EMA for stability monitoring (optional)
        if update_ema and self.training:
            with torch.no_grad():
                exp_marginal = torch.exp(t_marginal)
                self.ema = self.ema_decay * self.ema + (1 - self.ema_decay) * exp_marginal.mean()

        mi_estimate = joint_term - marginal_term

        return mi_estimate

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Single training step for MINE.

        Args:
            x: (B, D_x) input samples
            y: (B, D_y) input samples
            optimizer: PyTorch optimizer

        Returns:
            MI estimate (negated loss)
        """
        optimizer.zero_grad()

        # Maximize MI = minimize negative MI
        mi = self.forward(x, y)
        loss = -mi

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        return mi.item()


class InfoNCEEstimator(nn.Module):
    """
    InfoNCE bound for Mutual Information estimation.

    Based on: van den Oord et al., "Representation Learning with Contrastive Predictive Coding", 2018

    I(X;Y) ≥ log(K) - L_NCE

    where L_NCE is the NCE loss with K negative samples.

    This is more stable than MINE for high-dimensional data.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int = 256,
        temperature: float = 0.1,
    ):
        """
        Initialize InfoNCE estimator.

        Args:
            x_dim: Dimension of X
            y_dim: Dimension of Y
            hidden_dim: Hidden dimension for projection
            temperature: Temperature for softmax
        """
        super().__init__()

        self.temperature = temperature

        # Project both to same dimension for similarity
        self.x_proj = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.y_proj = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute InfoNCE bound on I(X;Y).

        Args:
            x: (B, D_x) input samples
            y: (B, D_y) input samples

        Returns:
            mi_lower_bound: Lower bound on I(X;Y)
            nce_loss: NCE loss for training
        """
        batch_size = x.shape[0]

        # Project to common space
        x_proj = F.normalize(self.x_proj(x), dim=-1)  # (B, H)
        y_proj = F.normalize(self.y_proj(y), dim=-1)  # (B, H)

        # Compute similarity matrix
        # sim[i,j] = x_i · y_j
        sim = torch.mm(x_proj, y_proj.t()) / self.temperature  # (B, B)

        # NCE loss: classify positive pair among B negatives
        labels = torch.arange(batch_size, device=x.device)
        nce_loss = F.cross_entropy(sim, labels)

        # MI lower bound: log(K) - L_NCE
        # K = batch_size (number of negative samples + 1)
        mi_lower_bound = np.log(batch_size) - nce_loss

        return mi_lower_bound, nce_loss

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Single training step for InfoNCE.

        Returns:
            MI estimate
        """
        optimizer.zero_grad()

        mi, nce_loss = self.forward(x, y)

        # Minimize NCE loss = maximize MI bound
        nce_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        return mi.item()


def estimate_conditional_mi(
    actions: np.ndarray,
    semantic: np.ndarray,
    numeric: np.ndarray,
    estimator_type: str = 'mine',
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = 'cuda',
) -> MIEstimationResult:
    """
    Estimate I(A; H | F) - conditional mutual information.

    For Theorem 1: I(A*_t; H*_t | F_t)

    Uses the chain rule:
        I(A; H | F) = I(A; H, F) - I(A; F)

    Args:
        actions: (N, D_a) action samples
        semantic: (N, D_h) semantic embeddings
        numeric: (N, D_f) numeric features
        estimator_type: 'mine' or 'infonce'
        n_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Torch device

    Returns:
        MIEstimationResult with conditional MI estimate
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Convert to tensors
    actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
    semantic_t = torch.tensor(semantic, dtype=torch.float32, device=device)
    numeric_t = torch.tensor(numeric, dtype=torch.float32, device=device)

    n_samples = len(actions)
    d_a, d_h, d_f = actions.shape[1], semantic.shape[1], numeric.shape[1]

    # Create combined features
    combined_t = torch.cat([semantic_t, numeric_t], dim=-1)  # (N, D_h + D_f)

    # Create estimators
    if estimator_type == 'mine':
        # I(A; H, F)
        est_combined = MINEEstimator(d_a, d_h + d_f).to(device)
        # I(A; F)
        est_numeric = MINEEstimator(d_a, d_f).to(device)
    else:
        est_combined = InfoNCEEstimator(d_a, d_h + d_f).to(device)
        est_numeric = InfoNCEEstimator(d_a, d_f).to(device)

    opt_combined = torch.optim.Adam(est_combined.parameters(), lr=lr)
    opt_numeric = torch.optim.Adam(est_numeric.parameters(), lr=lr)

    # Training loop
    mi_combined_history = []
    mi_numeric_history = []

    for epoch in range(n_epochs):
        # Shuffle data
        idx = torch.randperm(n_samples)

        epoch_mi_combined = []
        epoch_mi_numeric = []

        for i in range(0, n_samples, batch_size):
            batch_idx = idx[i:i+batch_size]
            if len(batch_idx) < 2:
                continue

            a_batch = actions_t[batch_idx]
            comb_batch = combined_t[batch_idx]
            num_batch = numeric_t[batch_idx]

            mi_c = est_combined.train_step(a_batch, comb_batch, opt_combined)
            mi_n = est_numeric.train_step(a_batch, num_batch, opt_numeric)

            epoch_mi_combined.append(mi_c)
            epoch_mi_numeric.append(mi_n)

        mi_combined_history.append(np.mean(epoch_mi_combined))
        mi_numeric_history.append(np.mean(epoch_mi_numeric))

    # Final estimates (last 10 epochs average)
    mi_combined = np.mean(mi_combined_history[-10:])
    mi_numeric = np.mean(mi_numeric_history[-10:])

    # Conditional MI via chain rule
    conditional_mi = max(0, mi_combined - mi_numeric)

    # Standard deviation from last 10 epochs
    mi_std = np.std([mi_combined_history[-10:][i] - mi_numeric_history[-10:][i]
                     for i in range(10)])

    return MIEstimationResult(
        mi_estimate=conditional_mi,
        mi_std=mi_std,
        method=f'{estimator_type}_conditional',
        n_samples=n_samples,
    )


def validate_sqrt_scaling(
    delta_V_list: List[float],
    I_list: List[float],
) -> Dict[str, float]:
    """
    Validate √I scaling from Theorem 1.

    Theorem 1 claims:
        ΔV ≥ C_V · √I(A*; H* | F)

    This should show as a linear relationship between ΔV and √I.

    Args:
        delta_V_list: List of value improvements at different MI levels
        I_list: List of corresponding MI estimates

    Returns:
        Validation results including C_V estimate and R²
    """
    try:
        from scipy.stats import linregress
    except ImportError:
        raise ImportError("scipy required for √I scaling validation. "
                         "Install with: pip install scipy")

    # Convert to numpy
    delta_V = np.array(delta_V_list)
    I = np.array(I_list)
    sqrt_I = np.sqrt(I)

    # Linear regression: ΔV = C_V · √I + intercept
    slope, intercept, r_value, p_value, std_err = linregress(sqrt_I, delta_V)

    # Check if scaling is valid (good fit)
    r_squared = r_value ** 2
    scaling_valid = r_squared > 0.7  # Threshold for "good" fit

    # Check if C_V is positive (as theorem requires ΔV ≥ 0)
    c_v_positive = slope > 0

    return {
        'C_V_estimated': slope,
        'C_V_std_err': std_err,
        'intercept': intercept,
        'r_squared': r_squared,
        'p_value': p_value,
        'scaling_valid': scaling_valid,
        'c_v_positive': c_v_positive,
        'theorem_1_holds': scaling_valid and c_v_positive,
        'n_points': len(delta_V_list),
    }


class SemanticValueAnalyzer:
    """
    Analyze the value of semantic information for Theorem 1 validation.

    Runs ablation experiments to measure ΔV at different MI levels.
    """

    def __init__(
        self,
        model_with_semantic,
        model_without_semantic,
        env,
        device: str = 'cuda',
    ):
        """
        Initialize analyzer.

        Args:
            model_with_semantic: Full STAIR-RL model
            model_without_semantic: Ablated model (numeric only)
            env: Trading environment
            device: Torch device
        """
        self.model_full = model_with_semantic
        self.model_numeric = model_without_semantic
        self.env = env
        self.device = device

    def run_ablation(
        self,
        n_episodes: int = 100,
    ) -> Dict[str, float]:
        """
        Run ablation to measure value difference.

        Returns:
            Dict with V_full, V_numeric, delta_V
        """
        # Evaluate full model
        v_full = self._evaluate_model(self.model_full, n_episodes)

        # Evaluate numeric-only model
        v_numeric = self._evaluate_model(self.model_numeric, n_episodes)

        delta_v = v_full - v_numeric

        return {
            'V_full': v_full,
            'V_numeric': v_numeric,
            'delta_V': delta_v,
            'n_episodes': n_episodes,
        }

    def _evaluate_model(self, model, n_episodes: int) -> float:
        """Evaluate model and return average return."""
        returns = []

        for _ in range(n_episodes):
            state, _ = self.env.reset()
            episode_return = 0
            done = False

            while not done:
                with torch.no_grad():
                    action = model.get_action_deterministic(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_return += reward
                done = terminated or truncated

            returns.append(episode_return)

        return np.mean(returns)
