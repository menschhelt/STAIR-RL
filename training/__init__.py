"""
Training Module - Training loops and utilities for RL agents.

Components:
- Trainer: Main training orchestrator
- Phase1Trainer: CQL-SAC offline pre-training
- Phase2Trainer: PPO-CVaR online fine-tuning
- EvaluationCallbacks: Evaluation metrics and logging
"""

__all__ = [
    'Trainer',
    'Phase1Trainer',
    'Phase2Trainer',
]


def __getattr__(name):
    """Lazy import to handle missing dependencies."""
    if name in ('Trainer', 'Phase1Trainer', 'Phase2Trainer'):
        from .trainer import Trainer, Phase1Trainer, Phase2Trainer
        return {'Trainer': Trainer, 'Phase1Trainer': Phase1Trainer, 'Phase2Trainer': Phase2Trainer}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
