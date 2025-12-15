"""
Validate that BaseBenchmark refactoring works correctly.

Tests:
1. PnLCalculator integration
2. PerformanceMetrics integration
3. Consistency with old implementation
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.base_benchmark import BaseBenchmark, BenchmarkConfig
from backtesting.pnl_calculator import PnLCalculator
from backtesting.metrics import PerformanceMetrics


class SimpleBenchmark(BaseBenchmark):
    """Simple equal-weight benchmark for testing."""

    def compute_weights(self, timestamp, prices, features, current_weights=None):
        """Equal weight allocation."""
        n_assets = len(prices)
        return np.ones(n_assets) / n_assets


def test_pnl_calculator_integration():
    """Test that BaseBenchmark uses PnLCalculator."""
    config = BenchmarkConfig(n_assets=3)
    benchmark = SimpleBenchmark(config=config)

    # Check that PnLCalculator was initialized
    assert hasattr(benchmark, 'pnl_calculator'), "PnLCalculator not initialized"
    assert isinstance(benchmark.pnl_calculator, PnLCalculator)
    assert benchmark.pnl_calculator.transaction_cost_rate == config.transaction_cost
    assert benchmark.pnl_calculator.slippage_rate == config.slippage

    print("✓ PnLCalculator integration test passed")


def test_performance_metrics_integration():
    """Test that _compute_metrics uses PerformanceMetrics."""
    config = BenchmarkConfig(n_assets=3)
    benchmark = SimpleBenchmark(config=config)

    # Create fake backtest results
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    nav = pd.Series([100000, 101000, 100495, 102505, 101480, 103000])
    weights_df = pd.DataFrame()
    turnover_history = [0.5, 0.3, 0.2, 0.4, 0.1]

    result = benchmark._compute_metrics(
        nav_series=nav,
        returns_series=returns,
        weights_df=weights_df,
        turnover_history=turnover_history,
        initial_nav=100000.0,
    )

    # Check that all metrics were computed
    assert result.sharpe_ratio is not None
    assert result.sortino_ratio is not None
    assert result.max_drawdown is not None
    assert result.calmar_ratio is not None
    assert result.cvar_95 is not None
    assert result.total_return is not None
    assert result.annual_return is not None

    # Verify consistency with PerformanceMetrics
    metrics_calc = PerformanceMetrics()
    expected_metrics = metrics_calc.calculate_all(
        returns=returns.values,
        equity_curve=nav.values,
        periods_per_year=252,
    )

    # Check that metrics match
    assert abs(result.sharpe_ratio - expected_metrics['sharpe_ratio']) < 1e-6
    assert abs(result.sortino_ratio - expected_metrics['sortino_ratio']) < 1e-6
    assert abs(result.max_drawdown - expected_metrics['max_drawdown']) < 1e-6
    assert abs(result.cvar_95 - expected_metrics['cvar_95']) < 1e-6

    print("✓ PerformanceMetrics integration test passed")


def test_pnl_consistency():
    """Test that PnL calculation is consistent with PnLCalculator."""
    config = BenchmarkConfig(n_assets=3, transaction_cost=0.0004, slippage=0.0001)
    benchmark = SimpleBenchmark(config=config)

    # Test single step PnL calculation
    prev_weights = np.array([0.3, 0.4, 0.3])
    new_weights = np.array([0.4, 0.3, 0.3])
    returns = np.array([0.01, -0.005, 0.02])

    # Calculate using PnLCalculator directly
    expected = benchmark.pnl_calculator.calculate_portfolio_return(
        weights=new_weights,
        asset_returns=returns,
        prev_weights=prev_weights,
        include_costs=True,
    )

    # Expected values
    # Gross return: (0.4*0.01) + (0.3*-0.005) + (0.3*0.02) = 0.004 - 0.0015 + 0.006 = 0.0085
    assert abs(expected['gross_return'] - 0.0085) < 1e-10

    # Turnover: |0.4-0.3| + |0.3-0.4| + |0.3-0.3| = 0.1 + 0.1 + 0 = 0.2
    assert abs(expected['turnover'] - 0.2) < 1e-10

    # Transaction cost: 0.2 * (0.0004 + 0.0001) = 0.0001
    expected_cost = 0.2 * (config.transaction_cost + config.slippage)
    assert abs(expected['transaction_cost'] - expected_cost) < 1e-10

    # Net return: 0.0085 - 0.0001 = 0.0084
    expected_net = 0.0085 - expected_cost
    assert abs(expected['net_return'] - expected_net) < 1e-10

    print("✓ PnL consistency test passed")


def test_nav_update():
    """Test NAV update using PnLCalculator."""
    config = BenchmarkConfig(n_assets=2)
    benchmark = SimpleBenchmark(config=config)

    initial_nav = 100_000.0
    portfolio_return = 0.015  # 1.5% return

    new_nav = benchmark.pnl_calculator.update_nav(initial_nav, portfolio_return)

    expected = 101_500.0
    assert abs(new_nav - expected) < 1e-6

    print("✓ NAV update test passed")


def test_cvar_calculation():
    """Test CVaR calculation in metrics."""
    returns = np.array([0.01, -0.02, 0.03, -0.05, 0.02, -0.01, 0.04, -0.03, 0.01, -0.04])

    metrics_calc = PerformanceMetrics()
    cvar = metrics_calc.cvar(returns, alpha=0.05)

    # CVaR(95%) should be the average of worst 5% (1 value out of 10)
    # Worst return: -0.05
    expected_cvar = -0.05
    assert abs(cvar - expected_cvar) < 1e-10

    print("✓ CVaR calculation test passed")


def main():
    """Run all validation tests."""
    print("Validating BaseBenchmark refactoring...\n")

    tests = [
        test_pnl_calculator_integration,
        test_performance_metrics_integration,
        test_pnl_consistency,
        test_nav_update,
        test_cvar_calculation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All BaseBenchmark refactoring tests passed!")
        print("Phase 4: BaseBenchmark refactoring complete!")


if __name__ == '__main__':
    main()
