"""
Shapley-Gate Alignment Analysis for Interpretability.

Paper Line 1375-1390:
    L_align = -λ_align · Corr(g_t, Φ_{s,t})

where:
- g_t: Semantic gate activations
- Φ_{s,t}: Shapley values for semantic features

This module computes:
1. Shapley values via KernelSHAP
2. Correlation between gate activations and Shapley values
3. Interpretability metrics for the semantic gate
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class ShapleyGateResult:
    """Result of Shapley-Gate alignment analysis."""
    correlation: float              # Corr(gate, shapley)
    p_value: float                  # Statistical significance
    alignment_loss: float           # -correlation (for training)
    gate_mean: float               # Mean gate activation
    gate_std: float                # Std of gate activation
    shapley_mean: float            # Mean absolute Shapley value
    n_samples: int                 # Number of samples analyzed


def compute_shapley_alignment(
    model_fn: Callable,
    states: np.ndarray,
    gate_activations: np.ndarray,
    background_states: np.ndarray,
    n_samples: int = 500,
    semantic_indices: Optional[List[int]] = None,
) -> ShapleyGateResult:
    """
    Compute alignment between gate activations and Shapley values.

    Paper Line 1375-1390:
        L_align = -λ_align · Corr(g_t, Φ_{s,t})

    High positive correlation indicates the gate learns to weight
    features proportionally to their true importance (Shapley value).

    Args:
        model_fn: Function that takes states and returns values/Q-values
        states: (N, D) state samples
        gate_activations: (N, D_sem) gate activations for semantic features
        background_states: (M, D) background samples for SHAP
        n_samples: Number of samples for SHAP computation
        semantic_indices: Indices of semantic features in state (if not all)

    Returns:
        ShapleyGateResult with correlation and statistics
    """
    if not HAS_SHAP:
        raise ImportError("SHAP library required for Shapley analysis. "
                         "Install with: pip install shap")

    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        raise ImportError("scipy required for correlation computation. "
                         "Install with: pip install scipy")

    # Limit samples for computational efficiency
    n_samples = min(n_samples, len(states))
    sample_idx = np.random.choice(len(states), n_samples, replace=False)

    states_sample = states[sample_idx]
    gate_sample = gate_activations[sample_idx]

    # Create SHAP explainer with background data
    background_sample = background_states[:min(100, len(background_states))]

    explainer = shap.KernelExplainer(
        model=model_fn,
        data=background_sample,
    )

    # Compute Shapley values
    shap_values = explainer.shap_values(states_sample, nsamples=100)

    # If multi-output, take mean
    if isinstance(shap_values, list):
        shap_values = np.mean(shap_values, axis=0)

    # Extract Shapley values for semantic features
    if semantic_indices is not None:
        shap_semantic = shap_values[:, semantic_indices]
    else:
        # Assume semantic features are at the end
        shap_semantic = shap_values[:, -gate_sample.shape[1]:]

    # Ensure shapes match
    if shap_semantic.shape != gate_sample.shape:
        # Flatten for correlation
        shap_flat = np.abs(shap_semantic).mean(axis=1)  # (N,)
        gate_flat = gate_sample.mean(axis=1)  # (N,)
    else:
        shap_flat = np.abs(shap_semantic).flatten()
        gate_flat = gate_sample.flatten()

    # Compute correlation
    correlation, p_value = pearsonr(gate_flat, shap_flat)

    return ShapleyGateResult(
        correlation=correlation,
        p_value=p_value,
        alignment_loss=-correlation,
        gate_mean=float(gate_sample.mean()),
        gate_std=float(gate_sample.std()),
        shapley_mean=float(np.abs(shap_semantic).mean()),
        n_samples=n_samples,
    )


class ShapleyGateAnalyzer:
    """
    Analyze gate-Shapley alignment during training.

    Periodically computes Shapley values and checks if the semantic gate
    learns to align with true feature importance.
    """

    def __init__(
        self,
        model,
        compute_interval: int = 10000,
        n_background: int = 500,
        n_eval_samples: int = 200,
    ):
        """
        Initialize analyzer.

        Args:
            model: STAIR-RL model with semantic gate
            compute_interval: Steps between Shapley computations
            n_background: Number of background samples for SHAP
            n_eval_samples: Number of samples for each evaluation
        """
        self.model = model
        self.compute_interval = compute_interval
        self.n_background = n_background
        self.n_eval_samples = n_eval_samples

        self.background_states: List[np.ndarray] = []
        self.history: List[Dict] = []
        self.step_count = 0

    def add_states(self, states: np.ndarray, gate_activations: np.ndarray):
        """
        Add states for background and potential evaluation.

        Args:
            states: (B, D) state batch
            gate_activations: (B, D_sem) gate activations
        """
        # Store for background
        for state in states:
            self.background_states.append(state)
            if len(self.background_states) > self.n_background:
                self.background_states.pop(0)

        self.step_count += len(states)

    def maybe_compute(
        self,
        states: np.ndarray,
        gate_activations: np.ndarray,
        force: bool = False,
    ) -> Optional[ShapleyGateResult]:
        """
        Compute alignment if interval reached.

        Args:
            states: Current states batch
            gate_activations: Current gate activations
            force: Force computation regardless of interval

        Returns:
            Result if computed, None otherwise
        """
        if not force and self.step_count % self.compute_interval != 0:
            return None

        if len(self.background_states) < 50:
            return None

        if not HAS_SHAP:
            return None

        try:
            def model_fn(x):
                with torch.no_grad():
                    x_t = torch.tensor(x, dtype=torch.float32)
                    if hasattr(self.model, 'critic'):
                        # Get Q-value or V-value
                        z, _ = self.model.encoder(x_t)
                        return self.model.critic(z).numpy()
                    else:
                        return self.model(x_t).numpy()

            background = np.array(self.background_states)

            result = compute_shapley_alignment(
                model_fn=model_fn,
                states=states,
                gate_activations=gate_activations,
                background_states=background,
                n_samples=self.n_eval_samples,
            )

            self.history.append({
                'step': self.step_count,
                'correlation': result.correlation,
                'p_value': result.p_value,
                'gate_mean': result.gate_mean,
            })

            return result

        except Exception as e:
            print(f"Shapley computation failed: {e}")
            return None

    def get_statistics(self) -> Dict[str, float]:
        """Get summary statistics."""
        if not self.history:
            return {}

        correlations = [h['correlation'] for h in self.history]
        return {
            'correlation_mean': np.mean(correlations),
            'correlation_std': np.std(correlations),
            'correlation_latest': correlations[-1],
            'correlation_trend': correlations[-1] - correlations[0] if len(correlations) > 1 else 0,
            'n_computations': len(self.history),
        }


class FastShapleyApproximator:
    """
    Fast Shapley value approximation using gradient-based methods.

    For large-scale analysis where KernelSHAP is too slow.
    Uses Integrated Gradients as an approximation to Shapley values.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize approximator.

        Args:
            model: PyTorch model (must be differentiable)
        """
        self.model = model

    def compute_integrated_gradients(
        self,
        inputs: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        target_idx: int = 0,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients as Shapley approximation.

        IG(x)_i = (x_i - x'_i) * ∫_{α=0}^{1} ∂F(x' + α(x-x')) / ∂x_i dα

        Args:
            inputs: (B, D) input tensor
            baseline: (D,) baseline (default: zeros)
            n_steps: Number of integration steps
            target_idx: Output index for gradients

        Returns:
            attributions: (B, D) feature attributions
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs[0])

        # Scale inputs along path from baseline
        scaled_inputs = []
        for alpha in np.linspace(0, 1, n_steps):
            scaled = baseline + alpha * (inputs - baseline)
            scaled_inputs.append(scaled)

        scaled_inputs = torch.stack(scaled_inputs, dim=1)  # (B, n_steps, D)

        # Compute gradients
        scaled_inputs.requires_grad_(True)

        B, S, D = scaled_inputs.shape
        outputs = self.model(scaled_inputs.reshape(B * S, D))

        if outputs.dim() > 1:
            outputs = outputs[:, target_idx]

        grads = torch.autograd.grad(
            outputs.sum(),
            scaled_inputs,
            create_graph=False,
        )[0]  # (B, n_steps, D)

        # Integrate
        avg_grads = grads.mean(dim=1)  # (B, D)
        attributions = (inputs - baseline) * avg_grads

        return attributions

    def compute_gate_alignment(
        self,
        inputs: torch.Tensor,
        gate_activations: torch.Tensor,
        semantic_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute alignment between gate and gradient-based attributions.

        Args:
            inputs: (B, D) input states
            gate_activations: (B, D_sem) gate activations
            semantic_mask: (D,) boolean mask for semantic features

        Returns:
            Alignment statistics
        """
        with torch.enable_grad():
            attributions = self.compute_integrated_gradients(inputs)

        # Extract semantic attributions
        if semantic_mask is not None:
            attr_semantic = attributions[:, semantic_mask]
        else:
            # Assume last D_sem features are semantic
            d_sem = gate_activations.shape[-1]
            attr_semantic = attributions[:, -d_sem:]

        # Normalize for comparison
        attr_norm = torch.abs(attr_semantic)
        attr_norm = attr_norm / (attr_norm.sum(dim=-1, keepdim=True) + 1e-8)

        gate_norm = gate_activations / (gate_activations.sum(dim=-1, keepdim=True) + 1e-8)

        # Cosine similarity as alignment metric
        cosine_sim = F.cosine_similarity(attr_norm, gate_norm, dim=-1).mean()

        # Pearson correlation
        attr_flat = attr_norm.flatten().detach().cpu().numpy()
        gate_flat = gate_norm.flatten().detach().cpu().numpy()

        try:
            from scipy.stats import pearsonr
            correlation, _ = pearsonr(attr_flat, gate_flat)
        except:
            correlation = float(cosine_sim)

        return {
            'cosine_similarity': float(cosine_sim),
            'correlation': correlation,
            'attr_entropy': float(-torch.sum(attr_norm * torch.log(attr_norm + 1e-8), dim=-1).mean()),
            'gate_entropy': float(-torch.sum(gate_norm * torch.log(gate_norm + 1e-8), dim=-1).mean()),
        }


def analyze_gate_by_regime(
    gate_activations: np.ndarray,
    regime_labels: np.ndarray,
    regime_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Analyze gate activations by market regime.

    Paper Table (Line 1340-1352) analyzes gate behavior across:
    - Bull market
    - Bear market
    - High volatility
    - Sideways/range-bound

    Args:
        gate_activations: (N, D_sem) gate activations
        regime_labels: (N,) regime labels (integers)
        regime_names: Optional mapping from label to name

    Returns:
        Statistics per regime
    """
    if regime_names is None:
        regime_names = {
            0: 'bull',
            1: 'bear',
            2: 'high_vol',
            3: 'sideways',
        }

    results = {}

    for label, name in regime_names.items():
        mask = regime_labels == label
        if mask.sum() == 0:
            continue

        gates = gate_activations[mask]

        results[name] = {
            'mean': float(gates.mean()),
            'std': float(gates.std()),
            'active_ratio': float((gates > 0.5).mean()),
            'min': float(gates.min()),
            'max': float(gates.max()),
            'n_samples': int(mask.sum()),
        }

    return results
