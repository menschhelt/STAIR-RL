"""
H-Divergence measurement for PAC Bound validation (Theorem 2).

The H-divergence measures distribution shift between offline and online data,
which is crucial for validating the PAC bound in offline-to-online transfer.

Theorem 2 (PAC Bound):
    P(CVaR_α(-R) ≤ κ + ε(δ, n)) ≥ 1 - δ

    where ε(δ, n) = C₁·ε_PPO + C₂·√(d_H + log(2/δ)/((1-α)n))

Paper claims (Line 3437-3446):
    - d_H^baseline = 0.47 ± 0.05
    - d_H^CQL = 0.31 ± 0.04 (33% reduction)
    - 96.2% confidence that CVaR ≤ 5%
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import torch

try:
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class HdivergenceResult:
    """Result of H-divergence computation."""
    d_H: float                    # H-divergence value
    epsilon_star: float           # Optimal classifier error
    accuracy: float               # Classifier accuracy
    accuracy_std: float           # Accuracy std across CV folds
    n_offline: int                # Number of offline samples
    n_online: int                 # Number of online samples


def compute_h_divergence(
    offline_states: np.ndarray,
    online_states: np.ndarray,
    n_splits: int = 5,
    normalize: bool = True,
    kernel: str = 'rbf',
    C: float = 1.0,
    gamma: str = 'scale',
    random_state: int = 42,
) -> HdivergenceResult:
    """
    Compute H-divergence between offline and online state distributions.

    H-divergence (Ben-David et al., 2010):
        d_H(D₁, D₂) = 2 * sup_{H ∈ H} |P_D₁[H] - P_D₂[H]|
                    = 2(1 - 2ε*)

    where ε* is the optimal error of a classifier distinguishing D₁ from D₂.

    Args:
        offline_states: (N_offline, D) states from offline dataset
        online_states: (N_online, D) states from online rollouts
        n_splits: Number of cross-validation folds
        normalize: Whether to standardize features
        kernel: SVM kernel ('rbf', 'linear', 'poly')
        C: SVM regularization parameter
        gamma: SVM gamma parameter
        random_state: Random seed for reproducibility

    Returns:
        HdivergenceResult with d_H, accuracy, etc.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required for H-divergence computation. "
                         "Install with: pip install scikit-learn")

    # Prepare data
    X = np.vstack([offline_states, online_states])
    y = np.concatenate([
        np.zeros(len(offline_states)),  # Label 0 for offline
        np.ones(len(online_states))      # Label 1 for online
    ])

    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Train SVM classifier with cross-validation
    clf = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    accuracy = scores.mean()
    accuracy_std = scores.std()

    # H-divergence calculation
    # If classifier is random (50% accuracy), d_H = 0
    # If classifier is perfect (100% accuracy), d_H = 2
    epsilon_star = 1 - accuracy
    d_H = 2 * (1 - 2 * epsilon_star)

    # Clamp to valid range [0, 2]
    d_H = max(0, min(2, d_H))

    return HdivergenceResult(
        d_H=d_H,
        epsilon_star=epsilon_star,
        accuracy=accuracy,
        accuracy_std=accuracy_std,
        n_offline=len(offline_states),
        n_online=len(online_states),
    )


def validate_pac_bound(
    d_H: float,
    n_online: int,
    alpha: float = 0.95,
    delta: float = 0.05,
    epsilon_ppo: float = 0.0002,
    kappa: float = 0.05,
    R_max: float = 0.10,
) -> Dict[str, float]:
    """
    Validate PAC bound from Theorem 2.

    ε(δ, n) = C₁·ε_PPO + C₂·√(d_H + log(2/δ)/((1-α)n))

    The bound guarantees:
        P(CVaR_α(-R) ≤ κ + ε) ≥ 1 - δ

    Args:
        d_H: H-divergence between offline and online distributions
        n_online: Number of online samples
        alpha: CVaR confidence level (default 0.95)
        delta: PAC bound failure probability (default 0.05)
        epsilon_ppo: PPO optimization error bound
        kappa: CVaR constraint threshold (default 0.05 = 5%)
        R_max: Maximum absolute return (for scaling constants)

    Returns:
        Dict with epsilon bound, constants, and theoretical CVaR bound
    """
    # Scaling constants (derived from R_max and alpha)
    C1 = R_max / (1 - alpha)  # ≈ 2.0 for α=0.95, R_max=0.10
    C2 = np.sqrt(2) * R_max / np.sqrt(1 - alpha)  # ≈ 0.63

    # Compute epsilon bound
    log_term = np.log(2 / delta)
    sample_term = log_term / ((1 - alpha) * n_online)

    epsilon = C1 * epsilon_ppo + C2 * np.sqrt(d_H + sample_term)

    # Theoretical CVaR bound
    cvar_bound = kappa + epsilon

    # Confidence level
    confidence = 1 - delta

    return {
        'epsilon': epsilon,
        'C1': C1,
        'C2': C2,
        'd_H': d_H,
        'n_online': n_online,
        'alpha': alpha,
        'delta': delta,
        'kappa': kappa,
        'theoretical_cvar_bound': cvar_bound,
        'confidence': confidence,
        'sample_term': sample_term,
        'ppo_contribution': C1 * epsilon_ppo,
        'distribution_contribution': C2 * np.sqrt(d_H),
    }


class HdivergenceMonitor:
    """
    Monitor H-divergence during online training.

    Tracks distribution shift between offline replay buffer and
    online rollouts to validate PAC bound assumptions.
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        compute_interval: int = 1000,
        alpha: float = 0.95,
        kappa: float = 0.05,
    ):
        """
        Initialize H-divergence monitor.

        Args:
            buffer_size: Size of state buffer for each distribution
            compute_interval: Steps between H-divergence computations
            alpha: CVaR confidence level
            kappa: CVaR constraint threshold
        """
        self.buffer_size = buffer_size
        self.compute_interval = compute_interval
        self.alpha = alpha
        self.kappa = kappa

        self.offline_buffer: List[np.ndarray] = []
        self.online_buffer: List[np.ndarray] = []
        self.history: List[Dict] = []
        self.step_count = 0

    def add_offline_states(self, states: np.ndarray):
        """Add states from offline dataset."""
        if len(states.shape) == 1:
            states = states.reshape(1, -1)

        for state in states:
            self.offline_buffer.append(state)
            if len(self.offline_buffer) > self.buffer_size:
                self.offline_buffer.pop(0)

    def add_online_states(self, states: np.ndarray):
        """Add states from online rollouts."""
        if len(states.shape) == 1:
            states = states.reshape(1, -1)

        for state in states:
            self.online_buffer.append(state)
            if len(self.online_buffer) > self.buffer_size:
                self.online_buffer.pop(0)

        self.step_count += len(states)

        # Check if we should compute H-divergence
        if (self.step_count % self.compute_interval == 0 and
            len(self.offline_buffer) >= 100 and
            len(self.online_buffer) >= 100):
            self._compute_and_log()

    def _compute_and_log(self):
        """Compute H-divergence and log results."""
        offline_arr = np.array(self.offline_buffer)
        online_arr = np.array(self.online_buffer)

        result = compute_h_divergence(offline_arr, online_arr)
        pac_result = validate_pac_bound(
            d_H=result.d_H,
            n_online=len(online_arr),
            alpha=self.alpha,
            kappa=self.kappa,
        )

        self.history.append({
            'step': self.step_count,
            'd_H': result.d_H,
            'accuracy': result.accuracy,
            'epsilon': pac_result['epsilon'],
            'cvar_bound': pac_result['theoretical_cvar_bound'],
        })

    def get_current_d_H(self) -> Optional[float]:
        """Get most recent H-divergence value."""
        if not self.history:
            return None
        return self.history[-1]['d_H']

    def get_statistics(self) -> Dict[str, float]:
        """Get summary statistics."""
        if not self.history:
            return {}

        d_H_values = [h['d_H'] for h in self.history]
        return {
            'd_H_mean': np.mean(d_H_values),
            'd_H_std': np.std(d_H_values),
            'd_H_min': np.min(d_H_values),
            'd_H_max': np.max(d_H_values),
            'd_H_latest': d_H_values[-1],
            'n_measurements': len(self.history),
        }


def compare_with_paper_claims(d_H: float, method: str = 'CQL') -> Dict[str, any]:
    """
    Compare measured H-divergence with paper claims.

    Paper claims (Line 3437-3446):
        - d_H^baseline = 0.47 ± 0.05
        - d_H^CQL = 0.31 ± 0.04

    Args:
        d_H: Measured H-divergence
        method: 'baseline' or 'CQL'

    Returns:
        Comparison results
    """
    claims = {
        'baseline': {'mean': 0.47, 'std': 0.05},
        'CQL': {'mean': 0.31, 'std': 0.04},
    }

    if method not in claims:
        method = 'CQL'

    claim = claims[method]
    z_score = (d_H - claim['mean']) / claim['std']
    within_1std = abs(z_score) <= 1
    within_2std = abs(z_score) <= 2

    # Reduction from baseline
    reduction = (claims['baseline']['mean'] - d_H) / claims['baseline']['mean'] * 100

    return {
        'measured_d_H': d_H,
        'paper_claim_mean': claim['mean'],
        'paper_claim_std': claim['std'],
        'z_score': z_score,
        'within_1std': within_1std,
        'within_2std': within_2std,
        'reduction_from_baseline_pct': reduction,
        'paper_reduction_claim_pct': 33.0,  # Paper claims 33% reduction
    }
