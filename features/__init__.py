"""
Features Module - Feature engineering and state construction.

Components:
- HierarchicalStateBuilder: Hierarchical state dict for multi-modal RL (DEPRECATED: old StateBuilder removed)
- TensorBuilder: 3D Tensor (T, Slots, Features) for batch training
- AlphaCalculator: Pre-calculate alpha factors
- RankNormalizer: Cross-sectional rank normalization
"""

# Lazy imports to avoid import errors when dependencies are missing
__all__ = [
    'HierarchicalStateBuilder',
    'TensorBuilder',
    'RankNormalizer',
    'AlphaCalculator',
    'AlphaAdapter',
]


def __getattr__(name):
    """Lazy import to handle missing dependencies."""
    if name == 'StateBuilder':
        # Redirect to HierarchicalStateBuilder for backward compatibility
        from agents.hierarchical_state_builder import HierarchicalStateBuilder
        import warnings
        warnings.warn(
            "StateBuilder is deprecated. Use HierarchicalStateBuilder from agents module.",
            DeprecationWarning,
            stacklevel=2
        )
        return HierarchicalStateBuilder
    elif name == 'HierarchicalStateBuilder':
        from agents.hierarchical_state_builder import HierarchicalStateBuilder
        return HierarchicalStateBuilder
    elif name == 'TensorBuilder':
        from .tensor_builder import TensorBuilder
        return TensorBuilder
    elif name == 'RankNormalizer':
        from .tensor_builder import RankNormalizer
        return RankNormalizer
    elif name == 'AlphaCalculator':
        from .alpha_calculator import AlphaCalculator
        return AlphaCalculator
    elif name == 'AlphaAdapter':
        from .alpha_adapter import AlphaAdapter
        return AlphaAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
