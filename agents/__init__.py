"""
Agents Module - RL Agent implementations for portfolio management.

Components:
- Actor: Policy network with Tanh output for continuous action space
- Critic: Value function networks (Q-function for SAC, V-function for PPO)
- CQLSACAgent: Conservative Q-Learning + Soft Actor-Critic (Phase 1: Offline)
- PPOCVaRAgent: PPO with CVaR constraint (Phase 2: Online)
- EventDetector: Event-driven retraining trigger (Phase 3)
"""

__all__ = [
    'Actor',
    'Critic',
    'ActorCritic',
    'FeatureEncoder',
    'CQLSACAgent',
    'PPOCVaRAgent',
    'EventDetector',
]


def __getattr__(name):
    """Lazy import to handle missing dependencies."""
    if name in ('Actor', 'Critic', 'ActorCritic', 'FeatureEncoder'):
        from .networks import Actor, Critic, ActorCritic, FeatureEncoder
        return {'Actor': Actor, 'Critic': Critic, 'ActorCritic': ActorCritic, 'FeatureEncoder': FeatureEncoder}[name]
    elif name == 'CQLSACAgent':
        from .cql_sac import CQLSACAgent
        return CQLSACAgent
    elif name == 'PPOCVaRAgent':
        from .ppo_cvar import PPOCVaRAgent
        return PPOCVaRAgent
    elif name == 'EventDetector':
        from .event_detector import EventDetector
        return EventDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
