"""
Evaluation Module - Theory validation metrics for STAIR-RL.

Components:
- h_divergence: H-divergence measurement for PAC bound validation (Theorem 2)
- mutual_information: MI estimation for √I scaling validation (Theorem 1)
- shapley_gate: Shapley-Gate alignment for interpretability
"""

from .h_divergence import (
    compute_h_divergence,
    validate_pac_bound,
    HdivergenceMonitor,
)

from .mutual_information import (
    MINEEstimator,
    InfoNCEEstimator,
    validate_sqrt_scaling,
)

from .shapley_gate import (
    compute_shapley_alignment,
    ShapleyGateAnalyzer,
)

__all__ = [
    # H-divergence
    'compute_h_divergence',
    'validate_pac_bound',
    'HdivergenceMonitor',
    # Mutual Information
    'MINEEstimator',
    'InfoNCEEstimator',
    'validate_sqrt_scaling',
    # Shapley-Gate
    'compute_shapley_alignment',
    'ShapleyGateAnalyzer',
]
