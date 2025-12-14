"""
Backtesting Module - Historical simulation and validation.

Components:
- BacktestEngine: Main simulation engine
- ParquetDataLoader: Load data from Parquet files
- DataValidator: Validate data integrity
- PerformanceMetrics: Calculate performance metrics
"""

__all__ = [
    'BacktestEngine',
    'Portfolio',
    'Position',
    'Trade',
    'ParquetDataLoader',
    'BacktestDataProvider',
    'DataValidator',
    'PerformanceMetrics',
]


def __getattr__(name):
    """Lazy import to handle missing dependencies."""
    if name in ('BacktestEngine', 'Portfolio', 'Position', 'Trade'):
        from .engine import BacktestEngine, Portfolio, Position, Trade
        return locals()[name]
    elif name in ('ParquetDataLoader', 'BacktestDataProvider'):
        from .data_loader import ParquetDataLoader, BacktestDataProvider
        return locals()[name]
    elif name == 'DataValidator':
        from .validator import DataValidator
        return DataValidator
    elif name == 'PerformanceMetrics':
        from .metrics import PerformanceMetrics
        return PerformanceMetrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
