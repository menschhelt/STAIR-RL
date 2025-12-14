"""
Environments Module - Gym-style RL environments for trading.

Components:
- TradingEnv: Main trading environment with state/action/reward interface
- VectorizedTradingEnv: Parallel environments for faster training
- PositionSizer: NAV-based position sizing with leverage management
- PositionInfo: Container for position sizing results
- EnvConfig: Environment configuration dataclass
"""

__all__ = [
    'TradingEnv',
    'VectorizedTradingEnv',
    'PositionSizer',
    'PositionInfo',
    'EnvConfig',
    'MarginCalculator',
]


def __getattr__(name):
    """Lazy import to handle missing dependencies."""
    if name == 'TradingEnv':
        from .trading_env import TradingEnv
        return TradingEnv
    elif name == 'VectorizedTradingEnv':
        from .trading_env import VectorizedTradingEnv
        return VectorizedTradingEnv
    elif name == 'EnvConfig':
        from .trading_env import EnvConfig
        return EnvConfig
    elif name == 'PositionSizer':
        from .position_sizer import PositionSizer
        return PositionSizer
    elif name == 'PositionInfo':
        from .position_sizer import PositionInfo
        return PositionInfo
    elif name == 'MarginCalculator':
        from .position_sizer import MarginCalculator
        return MarginCalculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
