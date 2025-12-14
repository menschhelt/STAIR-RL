"""
Benchmark Runner - Orchestrates running all benchmark strategies.

Provides unified interface for:
1. Running individual or all benchmarks
2. Comparing results across strategies
3. Generating comparison tables and visualizations
4. Statistical significance testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
from datetime import datetime
import logging
import json
from dataclasses import dataclass, field

from .base_benchmark import BaseBenchmark, BenchmarkConfig, BacktestResult
from .equal_weight import EqualWeightBenchmark, EqualRiskContributionBenchmark
from .cap_weight import CapWeightBenchmark, VolumeWeightBenchmark, SqrtCapWeightBenchmark
from .markowitz import MarkowitzBenchmark, MinVarianceBenchmark, MaxSharpeRatioBenchmark
from .price_only_rl import PriceOnlyRLBenchmark, MomentumRLBenchmark
from .factor_only_rl import FactorOnlyRLBenchmark
from .sentiment_score_rl import SentimentScoreRLBenchmark, SentimentMomentumRLBenchmark
from .fingpt_mvo import FinGPTMVOBenchmark, FinGPTMomentumBenchmark

logger = logging.getLogger(__name__)


# Registry of all available benchmarks
BENCHMARK_REGISTRY: Dict[str, Type[BaseBenchmark]] = {
    # Traditional strategies
    'equal_weight': EqualWeightBenchmark,
    'equal_risk': EqualRiskContributionBenchmark,
    'cap_weight': CapWeightBenchmark,
    'volume_weight': VolumeWeightBenchmark,
    'sqrt_cap_weight': SqrtCapWeightBenchmark,

    # Optimization-based
    'markowitz': MarkowitzBenchmark,
    'min_variance': MinVarianceBenchmark,
    'max_sharpe': MaxSharpeRatioBenchmark,

    # RL-based (no LLM)
    'price_only_rl': PriceOnlyRLBenchmark,
    'momentum_rl': MomentumRLBenchmark,
    'factor_only_rl': FactorOnlyRLBenchmark,  # = STAIR-RL
    'sentiment_score_rl': SentimentScoreRLBenchmark,
    'sentiment_momentum_rl': SentimentMomentumRLBenchmark,

    # LLM-based (vLLM)
    'fingpt_mvo': FinGPTMVOBenchmark,
    'fingpt_momentum': FinGPTMomentumBenchmark,
}


@dataclass
class BenchmarkSuite:
    """Configuration for a benchmark suite."""
    name: str
    strategies: List[str]
    description: str = ""


# Predefined benchmark suites
BENCHMARK_SUITES = {
    'paper': BenchmarkSuite(
        name='Paper Comparison',
        strategies=[
            'equal_weight',
            'cap_weight',
            'markowitz',
            'price_only_rl',
            'factor_only_rl',
            'sentiment_score_rl',
            'fingpt_mvo',
        ],
        description='Main comparison from STAIR-RL paper'
    ),
    'traditional': BenchmarkSuite(
        name='Traditional Strategies',
        strategies=['equal_weight', 'cap_weight', 'markowitz', 'min_variance'],
        description='Non-ML baseline strategies'
    ),
    'rl_only': BenchmarkSuite(
        name='RL Strategies',
        strategies=['price_only_rl', 'factor_only_rl', 'sentiment_score_rl'],
        description='RL-based strategies without LLM'
    ),
    'full': BenchmarkSuite(
        name='Full Comparison',
        strategies=list(BENCHMARK_REGISTRY.keys()),
        description='All available strategies'
    ),
}


class BenchmarkRunner:
    """
    Orchestrates running multiple benchmark strategies.

    Usage:
    ```python
    runner = BenchmarkRunner(
        data=historical_data,
        config=BenchmarkConfig(n_assets=20),
    )

    # Run specific strategies
    results = runner.run(['equal_weight', 'factor_only_rl'])

    # Run predefined suite
    results = runner.run_suite('paper')

    # Compare results
    comparison = runner.compare_results(results)
    print(comparison)
    ```
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[BenchmarkConfig] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize benchmark runner.

        Args:
            data: Historical data for backtesting
            config: Benchmark configuration
            output_dir: Directory for saving results
        """
        self.data = data
        self.config = config or BenchmarkConfig()
        self.output_dir = Path(output_dir) if output_dir else Path('benchmark_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, BacktestResult] = {}

    def run(
        self,
        strategies: List[str],
        start_date: str,
        end_date: str,
        initial_nav: float = 100_000.0,
        **kwargs,
    ) -> Dict[str, BacktestResult]:
        """
        Run specified benchmark strategies.

        Args:
            strategies: List of strategy names from BENCHMARK_REGISTRY
            start_date: Backtest start date
            end_date: Backtest end date
            initial_nav: Initial portfolio value
            **kwargs: Additional arguments passed to benchmark constructors

        Returns:
            Dict of strategy_name -> BacktestResult
        """
        results = {}

        for strategy_name in strategies:
            if strategy_name not in BENCHMARK_REGISTRY:
                logger.warning(f"Unknown strategy: {strategy_name}, skipping")
                continue

            logger.info(f"Running benchmark: {strategy_name}")

            try:
                # Instantiate strategy
                benchmark_class = BENCHMARK_REGISTRY[strategy_name]
                benchmark = benchmark_class(config=self.config, **kwargs)

                # Run backtest
                result = benchmark.run_backtest(
                    data=self.data.copy(),
                    start_date=start_date,
                    end_date=end_date,
                    initial_nav=initial_nav,
                )

                results[strategy_name] = result
                self.results[strategy_name] = result

                logger.info(
                    f"  {strategy_name}: "
                    f"Sharpe={result.sharpe_ratio:.3f}, "
                    f"Return={result.annual_return*100:.1f}%, "
                    f"MaxDD={result.max_drawdown*100:.1f}%"
                )

            except Exception as e:
                logger.error(f"Failed to run {strategy_name}: {e}")
                continue

        return results

    def run_suite(
        self,
        suite_name: str,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> Dict[str, BacktestResult]:
        """
        Run a predefined benchmark suite.

        Args:
            suite_name: Name of suite ('paper', 'traditional', 'rl_only', 'full')
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Dict of strategy_name -> BacktestResult
        """
        if suite_name not in BENCHMARK_SUITES:
            raise ValueError(f"Unknown suite: {suite_name}. Available: {list(BENCHMARK_SUITES.keys())}")

        suite = BENCHMARK_SUITES[suite_name]
        logger.info(f"Running suite: {suite.name} ({suite.description})")

        return self.run(
            strategies=suite.strategies,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )

    def compare_results(
        self,
        results: Optional[Dict[str, BacktestResult]] = None,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create comparison table of results.

        Args:
            results: Results to compare (uses self.results if None)
            metrics: Specific metrics to include

        Returns:
            DataFrame with comparison
        """
        results = results or self.results

        if not results:
            logger.warning("No results to compare")
            return pd.DataFrame()

        # Default metrics
        if metrics is None:
            metrics = [
                'total_return',
                'annual_return',
                'sharpe_ratio',
                'sortino_ratio',
                'max_drawdown',
                'calmar_ratio',
                'cvar_95',
                'volatility',
                'total_turnover',
                'avg_turnover',
                'total_cost',
            ]

        # Build comparison table
        rows = []
        for strategy_name, result in results.items():
            row = {'strategy': strategy_name}
            for metric in metrics:
                value = getattr(result, metric, None)
                if value is not None:
                    row[metric] = value
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.set_index('strategy')

        # Sort by Sharpe ratio (descending)
        if 'sharpe_ratio' in df.columns:
            df = df.sort_values('sharpe_ratio', ascending=False)

        return df

    def compute_relative_performance(
        self,
        results: Optional[Dict[str, BacktestResult]] = None,
        baseline: str = 'equal_weight',
    ) -> pd.DataFrame:
        """
        Compute performance relative to a baseline strategy.

        Args:
            results: Results to analyze
            baseline: Baseline strategy name

        Returns:
            DataFrame with relative metrics
        """
        results = results or self.results

        if baseline not in results:
            logger.warning(f"Baseline {baseline} not in results")
            return pd.DataFrame()

        baseline_result = results[baseline]

        rows = []
        for strategy_name, result in results.items():
            if strategy_name == baseline:
                continue

            row = {
                'strategy': strategy_name,
                'excess_return': result.annual_return - baseline_result.annual_return,
                'sharpe_improvement': result.sharpe_ratio - baseline_result.sharpe_ratio,
                'dd_improvement': baseline_result.max_drawdown - result.max_drawdown,
                'turnover_diff': result.avg_turnover - baseline_result.avg_turnover,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.set_index('strategy')
            df = df.sort_values('sharpe_improvement', ascending=False)

        return df

    def statistical_tests(
        self,
        results: Optional[Dict[str, BacktestResult]] = None,
        baseline: str = 'equal_weight',
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform statistical significance tests.

        Tests:
        - Paired t-test on returns
        - Bootstrap confidence intervals

        Args:
            results: Results to test
            baseline: Baseline strategy for comparison

        Returns:
            Dict of test results
        """
        results = results or self.results

        if baseline not in results:
            return {}

        baseline_returns = results[baseline].returns_series

        test_results = {}
        for strategy_name, result in results.items():
            if strategy_name == baseline:
                continue

            strategy_returns = result.returns_series

            # Align returns
            aligned = pd.concat([baseline_returns, strategy_returns], axis=1, join='inner')
            if len(aligned) < 30:
                continue

            baseline_rets = aligned.iloc[:, 0].values
            strategy_rets = aligned.iloc[:, 1].values

            # Paired t-test
            try:
                from scipy import stats
                diff = strategy_rets - baseline_rets
                t_stat, p_value = stats.ttest_1samp(diff, 0)

                # Bootstrap mean difference
                n_bootstrap = 1000
                bootstrap_diffs = []
                for _ in range(n_bootstrap):
                    idx = np.random.choice(len(diff), size=len(diff), replace=True)
                    bootstrap_diffs.append(np.mean(diff[idx]))

                ci_lower = np.percentile(bootstrap_diffs, 2.5)
                ci_upper = np.percentile(bootstrap_diffs, 97.5)

                test_results[strategy_name] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'mean_difference': np.mean(diff),
                    'ci_95_lower': ci_lower,
                    'ci_95_upper': ci_upper,
                    'significant_95': p_value < 0.05,
                }

            except ImportError:
                logger.warning("scipy not installed, skipping statistical tests")
                break

        return test_results

    def generate_report(
        self,
        results: Optional[Dict[str, BacktestResult]] = None,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate markdown report of benchmark results.

        Args:
            results: Results to report
            output_path: Path to save report

        Returns:
            Report as markdown string
        """
        results = results or self.results
        output_path = output_path or self.output_dir / 'benchmark_report.md'

        # Build report
        lines = [
            "# Benchmark Comparison Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## Summary\n",
        ]

        # Comparison table
        comparison = self.compare_results(results)
        if not comparison.empty:
            # Format percentages
            pct_cols = ['total_return', 'annual_return', 'max_drawdown', 'cvar_95', 'volatility']
            for col in pct_cols:
                if col in comparison.columns:
                    comparison[col] = comparison[col].apply(lambda x: f"{x*100:.2f}%")

            # Format ratios
            ratio_cols = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
            for col in ratio_cols:
                if col in comparison.columns:
                    comparison[col] = comparison[col].apply(lambda x: f"{x:.3f}")

            lines.append(comparison.to_markdown())
            lines.append("\n")

        # Best performers
        lines.append("## Best Performers\n")
        if 'sharpe_ratio' in self.compare_results(results).columns:
            comp = self.compare_results(results)
            best_sharpe = comp['sharpe_ratio'].idxmax()
            lines.append(f"- **Best Sharpe Ratio**: {best_sharpe}\n")

        if 'annual_return' in self.compare_results(results).columns:
            comp = self.compare_results(results)
            best_return = comp['annual_return'].idxmax()
            lines.append(f"- **Best Annual Return**: {best_return}\n")

        # Relative performance
        lines.append("\n## Relative Performance (vs Equal-Weight)\n")
        rel_perf = self.compute_relative_performance(results)
        if not rel_perf.empty:
            lines.append(rel_perf.to_markdown())
            lines.append("\n")

        # Statistical tests
        lines.append("\n## Statistical Significance\n")
        stats_results = self.statistical_tests(results)
        for strategy, stats in stats_results.items():
            sig = "Yes" if stats['significant_95'] else "No"
            lines.append(
                f"- **{strategy}**: p={stats['p_value']:.4f}, "
                f"significant at 95%: {sig}\n"
            )

        report = "\n".join(lines)

        # Save
        with open(output_path, 'w') as f:
            f.write(report)

        logger.info(f"Report saved to {output_path}")

        return report

    def save_results(
        self,
        results: Optional[Dict[str, BacktestResult]] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Save results to files.

        Saves:
        - results.json: Metrics summary
        - nav_series.parquet: NAV time series
        - returns_series.parquet: Returns time series
        """
        results = results or self.results
        output_dir = output_dir or self.output_dir

        # Save metrics summary
        metrics = {}
        for strategy_name, result in results.items():
            metrics[strategy_name] = result.to_dict()

        with open(output_dir / 'results.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        # Save time series
        nav_dict = {}
        returns_dict = {}

        for strategy_name, result in results.items():
            if result.nav_series is not None:
                nav_dict[strategy_name] = result.nav_series
            if result.returns_series is not None:
                returns_dict[strategy_name] = result.returns_series

        if nav_dict:
            nav_df = pd.DataFrame(nav_dict)
            nav_df.to_parquet(output_dir / 'nav_series.parquet')

        if returns_dict:
            returns_df = pd.DataFrame(returns_dict)
            returns_df.to_parquet(output_dir / 'returns_series.parquet')

        logger.info(f"Results saved to {output_dir}")

    def load_results(self, input_dir: Optional[Path] = None) -> Dict[str, BacktestResult]:
        """Load previously saved results."""
        input_dir = input_dir or self.output_dir

        # Load metrics
        with open(input_dir / 'results.json', 'r') as f:
            metrics = json.load(f)

        # Load time series
        nav_df = pd.read_parquet(input_dir / 'nav_series.parquet')
        returns_df = pd.read_parquet(input_dir / 'returns_series.parquet')

        # Reconstruct BacktestResult objects
        results = {}
        for strategy_name, m in metrics.items():
            result = BacktestResult(
                strategy_name=strategy_name,
                total_return=m.get('total_return', 0),
                annual_return=m.get('annual_return', 0),
                sharpe_ratio=m.get('sharpe_ratio', 0),
                sortino_ratio=m.get('sortino_ratio', 0),
                max_drawdown=m.get('max_drawdown', 0),
                calmar_ratio=m.get('calmar_ratio', 0),
                cvar_95=m.get('cvar_95', 0),
                volatility=m.get('volatility', 0),
                total_turnover=m.get('total_turnover', 0),
            )

            if strategy_name in nav_df.columns:
                result.nav_series = nav_df[strategy_name]
            if strategy_name in returns_df.columns:
                result.returns_series = returns_df[strategy_name]

            results[strategy_name] = result

        self.results = results
        return results


def run_benchmark_comparison(
    data_path: Path,
    output_dir: Path,
    start_date: str,
    end_date: str,
    suite: str = 'paper',
):
    """
    Convenience function to run benchmark comparison.

    Args:
        data_path: Path to historical data parquet
        output_dir: Output directory for results
        start_date: Backtest start date
        end_date: Backtest end date
        suite: Benchmark suite to run
    """
    # Load data
    data = pd.read_parquet(data_path)

    # Run benchmarks
    runner = BenchmarkRunner(data=data, output_dir=output_dir)
    results = runner.run_suite(suite, start_date, end_date)

    # Generate outputs
    runner.generate_report(results)
    runner.save_results(results)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON RESULTS")
    print("=" * 60)
    print(runner.compare_results(results).to_string())
    print("=" * 60)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run benchmark comparison')
    parser.add_argument('--data', type=str, required=True, help='Path to data parquet')
    parser.add_argument('--output', type=str, default='benchmark_results', help='Output directory')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--suite', type=str, default='paper', help='Benchmark suite')

    args = parser.parse_args()

    run_benchmark_comparison(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        start_date=args.start,
        end_date=args.end,
        suite=args.suite,
    )
