#!/usr/bin/env python3
"""
Benchmark Comparison Script.

Runs all benchmark strategies and generates comparison report:
- Traditional: Equal-Weight, Cap-Weight
- Optimization: Markowitz MVO, Min-Variance, Max-Sharpe
- RL (no LLM): Price-only, Factor-only (STAIR-RL), Sentiment-score
- LLM: FinGPT+MVO (requires vLLM server)

Usage:
    python scripts/run_benchmark.py --suite paper --start 2024-01-01 --end 2024-06-30
    python scripts/run_benchmark.py --suite traditional
    python scripts/run_benchmark.py --strategies equal_weight,factor_only_rl,fingpt_mvo

Predefined Suites:
    - paper: Main comparison from STAIR-RL paper
    - traditional: Non-ML baseline strategies
    - rl_only: RL strategies without LLM
    - full: All available strategies
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config, DATA_DIR, BASE_DIR
from benchmarks import (
    BenchmarkRunner,
    BenchmarkConfig,
    BENCHMARK_REGISTRY,
    BENCHMARK_SUITES,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(config: Config, start_date: str, end_date: str) -> pd.DataFrame:
    """Load and prepare data for benchmarking."""
    from backtesting.data_loader import BacktestDataLoader

    logger.info("Loading data for benchmarking...")

    data_loader = BacktestDataLoader(
        data_dir=DATA_DIR,
        feature_dir=DATA_DIR / 'features',
    )

    data = data_loader.load_period(
        start_date=start_date,
        end_date=end_date,
    )

    logger.info(f"Loaded {len(data)} rows")
    return data


def run_benchmark_suite(
    suite_name: str,
    config: Config,
    start_date: str,
    end_date: str,
    output_dir: Path,
    initial_nav: float = 100_000.0,
):
    """Run a predefined benchmark suite."""
    logger.info("=" * 60)
    logger.info(f"Running Benchmark Suite: {suite_name}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    # Load data
    data = load_data(config, start_date, end_date)

    if len(data) == 0:
        logger.error("No data available for benchmarking")
        return None

    # Create benchmark config
    benchmark_config = BenchmarkConfig(
        n_assets=config.universe.top_n,
        rebalance_freq='daily',
        target_leverage=config.rl.target_leverage,
        transaction_cost=config.backtest.taker_fee,
        slippage=config.backtest.slippage,
    )

    # Create runner
    runner = BenchmarkRunner(
        data=data,
        config=benchmark_config,
        output_dir=output_dir,
    )

    # Run suite
    results = runner.run_suite(
        suite_name=suite_name,
        start_date=start_date,
        end_date=end_date,
        initial_nav=initial_nav,
    )

    # Generate outputs
    comparison = runner.compare_results(results)
    runner.generate_report(results)
    runner.save_results(results)

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON RESULTS")
    print("=" * 80)
    print(comparison.to_string())
    print("=" * 80)

    return results


def run_specific_strategies(
    strategies: list,
    config: Config,
    start_date: str,
    end_date: str,
    output_dir: Path,
    initial_nav: float = 100_000.0,
):
    """Run specific benchmark strategies."""
    logger.info("=" * 60)
    logger.info(f"Running Benchmarks: {', '.join(strategies)}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    # Validate strategies
    invalid = [s for s in strategies if s not in BENCHMARK_REGISTRY]
    if invalid:
        logger.error(f"Unknown strategies: {invalid}")
        logger.info(f"Available: {list(BENCHMARK_REGISTRY.keys())}")
        return None

    # Load data
    data = load_data(config, start_date, end_date)

    if len(data) == 0:
        logger.error("No data available for benchmarking")
        return None

    # Create benchmark config
    benchmark_config = BenchmarkConfig(
        n_assets=config.universe.top_n,
        rebalance_freq='daily',
        target_leverage=config.rl.target_leverage,
        transaction_cost=config.backtest.taker_fee,
        slippage=config.backtest.slippage,
    )

    # Create runner
    runner = BenchmarkRunner(
        data=data,
        config=benchmark_config,
        output_dir=output_dir,
    )

    # Run strategies
    results = runner.run(
        strategies=strategies,
        start_date=start_date,
        end_date=end_date,
        initial_nav=initial_nav,
    )

    # Generate outputs
    comparison = runner.compare_results(results)
    runner.generate_report(results)
    runner.save_results(results)

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON RESULTS")
    print("=" * 80)
    print(comparison.to_string())
    print("=" * 80)

    return results


def list_benchmarks():
    """List all available benchmarks."""
    print("\n" + "=" * 60)
    print("AVAILABLE BENCHMARKS")
    print("=" * 60)

    print("\nIndividual Strategies:")
    print("-" * 40)
    for name, cls in BENCHMARK_REGISTRY.items():
        print(f"  {name}: {cls.__name__}")

    print("\nPredefined Suites:")
    print("-" * 40)
    for name, suite in BENCHMARK_SUITES.items():
        print(f"  {name}: {suite.description}")
        print(f"    Strategies: {', '.join(suite.strategies)}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Run benchmark comparison for STAIR-RL'
    )
    parser.add_argument(
        '--suite', type=str, default=None,
        choices=list(BENCHMARK_SUITES.keys()),
        help='Predefined benchmark suite to run'
    )
    parser.add_argument(
        '--strategies', type=str, default=None,
        help='Comma-separated list of strategies to run'
    )
    parser.add_argument(
        '--start', type=str, default=None,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end', type=str, default=None,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--initial-nav', type=float, default=100000.0,
        help='Initial portfolio value'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List available benchmarks and suites'
    )

    args = parser.parse_args()

    # List benchmarks if requested
    if args.list:
        list_benchmarks()
        return

    # Check that we have something to run
    if not args.suite and not args.strategies:
        logger.error("Please specify --suite or --strategies (or use --list to see options)")
        sys.exit(1)

    # Load config
    if args.config:
        config = Config.from_yaml(Path(args.config))
    else:
        config = Config()

    # Set dates
    start_date = args.start or config.backtest.test_start
    end_date = args.end or datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = BASE_DIR / 'benchmark_results' / f'run_{timestamp}'

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")

    # Run benchmarks
    if args.suite:
        run_benchmark_suite(
            suite_name=args.suite,
            config=config,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            initial_nav=args.initial_nav,
        )
    else:
        strategies = [s.strip() for s in args.strategies.split(',')]
        run_specific_strategies(
            strategies=strategies,
            config=config,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            initial_nav=args.initial_nav,
        )


if __name__ == '__main__':
    main()
