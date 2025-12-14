"""
Features Module - Feature engineering and state construction.

Components:
- StateBuilder: 2D State Matrix (N assets, D features) for RL
- TensorBuilder: 3D Tensor (T, Slots, Features) for batch training
- AlphaCalculator: Pre-calculate alpha factors
- RankNormalizer: Cross-sectional rank normalization
- AlphaPCACompressor: Rolling PCA for alpha compression
"""

# Lazy imports to avoid import errors when dependencies are missing
__all__ = [
    'StateBuilder',
    'TensorBuilder',
    'RankNormalizer',
    'AlphaCalculator',
    'AlphaAdapter',
    'AlphaPCACompressor',
]


def __getattr__(name):
    """Lazy import to handle missing dependencies."""
    if name == 'StateBuilder':
        from .state_builder import StateBuilder
        return StateBuilder
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
    elif name == 'AlphaPCACompressor':
        from .pca_compressor import AlphaPCACompressor
        return AlphaPCACompressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
