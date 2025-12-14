"""
Benchmark Module - Portfolio benchmark strategies for comparison.

Components:
- BaseBenchmark: Abstract base class for all benchmarks
- BenchmarkConfig: Configuration dataclass
- BacktestResult: Results container

Traditional Strategies:
- EqualWeightBenchmark: Equal-weight (1/N) portfolio
- EqualRiskContributionBenchmark: Inverse-volatility weighting
- CapWeightBenchmark: Capitalization-weighted portfolio
- VolumeWeightBenchmark: Volume-weighted portfolio
- SqrtCapWeightBenchmark: Square-root cap weighting

Optimization-based:
- MarkowitzBenchmark: Mean-variance optimization
- MinVarianceBenchmark: Minimum variance portfolio
- MaxSharpeRatioBenchmark: Maximum Sharpe ratio (tangency)

RL-based (No LLM):
- PriceOnlyRLBenchmark: RL with price/volatility features only
- MomentumRLBenchmark: RL with momentum features
- FactorOnlyRLBenchmark: RL with factor features (= STAIR-RL)
- SentimentScoreRLBenchmark: RL with factor + sentiment scores
- SentimentMomentumRLBenchmark: Simplified momentum + sentiment

LLM-based (vLLM):
- FinGPTMVOBenchmark: FinGPT + Mean-Variance Optimization
- FinGPTMomentumBenchmark: FinGPT + Momentum

Utilities:
- BenchmarkRunner: Run all benchmarks and compare results
- BENCHMARK_REGISTRY: Dict of all available benchmarks
- BENCHMARK_SUITES: Predefined benchmark suites
"""

# Base classes
from .base_benchmark import (
    BaseBenchmark,
    BenchmarkConfig,
    BacktestResult,
)

# Traditional strategies
from .equal_weight import (
    EqualWeightBenchmark,
    EqualRiskContributionBenchmark,
)
from .cap_weight import (
    CapWeightBenchmark,
    VolumeWeightBenchmark,
    SqrtCapWeightBenchmark,
)

# Optimization-based
from .markowitz import (
    MarkowitzBenchmark,
    MinVarianceBenchmark,
    MaxSharpeRatioBenchmark,
)

# RL-based (no LLM)
from .price_only_rl import (
    PriceOnlyRLBenchmark,
    MomentumRLBenchmark,
)
from .factor_only_rl import (
    FactorOnlyRLBenchmark,
)
from .sentiment_score_rl import (
    SentimentScoreRLBenchmark,
    SentimentMomentumRLBenchmark,
)

# LLM-based (vLLM)
from .fingpt_mvo import (
    FinGPTMVOBenchmark,
    FinGPTMomentumBenchmark,
    FinGPTConfig,
    FinGPTClient,
)

# Runner and utilities
from .benchmark_runner import (
    BenchmarkRunner,
    BENCHMARK_REGISTRY,
    BENCHMARK_SUITES,
    BenchmarkSuite,
    run_benchmark_comparison,
)


__all__ = [
    # Base
    'BaseBenchmark',
    'BenchmarkConfig',
    'BacktestResult',

    # Traditional
    'EqualWeightBenchmark',
    'EqualRiskContributionBenchmark',
    'CapWeightBenchmark',
    'VolumeWeightBenchmark',
    'SqrtCapWeightBenchmark',

    # Optimization
    'MarkowitzBenchmark',
    'MinVarianceBenchmark',
    'MaxSharpeRatioBenchmark',

    # RL (no LLM)
    'PriceOnlyRLBenchmark',
    'MomentumRLBenchmark',
    'FactorOnlyRLBenchmark',
    'SentimentScoreRLBenchmark',
    'SentimentMomentumRLBenchmark',

    # LLM (vLLM)
    'FinGPTMVOBenchmark',
    'FinGPTMomentumBenchmark',
    'FinGPTConfig',
    'FinGPTClient',

    # Utilities
    'BenchmarkRunner',
    'BENCHMARK_REGISTRY',
    'BENCHMARK_SUITES',
    'BenchmarkSuite',
    'run_benchmark_comparison',
]
