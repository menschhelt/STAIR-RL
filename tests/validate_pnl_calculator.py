"""
Simple validation script for PnLCalculator (no pytest required).
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.pnl_calculator import PnLCalculator, PnLResult


def test_simple_portfolio_return():
    """Test basic portfolio return calculation."""
    calc = PnLCalculator()
    weights = np.array([0.5, 0.5])
    returns = np.array([0.01, -0.005])

    result = calc.calculate_portfolio_return(
        weights=weights,
        asset_returns=returns,
        include_costs=False,
    )

    expected = 0.0025  # (0.5 * 0.01) + (0.5 * -0.005)
    assert abs(result['gross_return'] - expected) < 1e-10, \
        f"Expected {expected}, got {result['gross_return']}"
    assert result['net_return'] == result['gross_return']
    print("✓ Simple portfolio return test passed")


def test_portfolio_return_with_costs():
    """Test portfolio return with transaction costs."""
    calc = PnLCalculator(transaction_cost_rate=0.0004, slippage_rate=0.0001)
    prev_weights = np.array([0.5, 0.5])
    new_weights = np.array([0.7, 0.3])
    returns = np.array([0.01, -0.005])

    result = calc.calculate_portfolio_return(
        weights=new_weights,
        asset_returns=returns,
        prev_weights=prev_weights,
        include_costs=True,
    )

    # Turnover: |0.7-0.5| + |0.3-0.5| = 0.4
    assert abs(result['turnover'] - 0.4) < 1e-10
    # Transaction cost (includes slippage): 0.4 * (0.0004 + 0.0001) = 0.0002
    assert abs(result['transaction_cost'] - 0.0002) < 1e-10
    print("✓ Portfolio return with costs test passed")


def test_nav_update():
    """Test NAV update."""
    calc = PnLCalculator()
    prev_nav = 100_000.0
    portfolio_return = 0.01

    new_nav = calc.update_nav(prev_nav, portfolio_return)

    expected = 101_000.0
    assert abs(new_nav - expected) < 1e-6, \
        f"Expected {expected}, got {new_nav}"
    print("✓ NAV update test passed")


def test_transaction_cost_calculation():
    """Test detailed transaction cost breakdown."""
    calc = PnLCalculator(
        transaction_cost_rate=0.0004,
        slippage_rate=0.0001,
    )
    prev_weights = np.array([0.5, 0.5])
    new_weights = np.array([0.7, 0.3])
    nav = 100_000.0

    result = calc.calculate_transaction_cost(
        prev_weights=prev_weights,
        new_weights=new_weights,
        nav=nav,
    )

    # Turnover: 0.4
    assert abs(result['turnover'] - 0.4) < 1e-10
    # Notional: 40,000
    assert abs(result['notional_traded'] - 40_000.0) < 1e-6
    # Transaction cost: 16
    assert abs(result['transaction_cost'] - 16.0) < 1e-6
    # Slippage: 4
    assert abs(result['slippage_cost'] - 4.0) < 1e-6
    # Total: 20
    assert abs(result['total_cost'] - 20.0) < 1e-6
    print("✓ Transaction cost calculation test passed")


def test_leveraged_portfolio():
    """Test portfolio with leverage."""
    calc = PnLCalculator()
    weights = np.array([0.8, 0.8, 0.4])  # 2x leverage
    returns = np.array([0.01, -0.005, 0.02])

    result = calc.calculate_portfolio_return(
        weights=weights,
        asset_returns=returns,
        include_costs=False,
    )

    expected = 0.012  # (0.8*0.01) + (0.8*-0.005) + (0.4*0.02)
    assert abs(result['gross_return'] - expected) < 1e-10
    print("✓ Leveraged portfolio test passed")


def test_short_positions():
    """Test portfolio with short positions."""
    calc = PnLCalculator()
    weights = np.array([0.5, -0.3, 0.8])
    returns = np.array([0.01, 0.02, -0.005])

    result = calc.calculate_portfolio_return(
        weights=weights,
        asset_returns=returns,
        include_costs=False,
    )

    expected = -0.005  # (0.5*0.01) + (-0.3*0.02) + (0.8*-0.005)
    assert abs(result['gross_return'] - expected) < 1e-10
    print("✓ Short positions test passed")


def test_funding_rates():
    """Test funding rate payments."""
    calc = PnLCalculator()
    weights = np.array([0.5, -0.3, 0.2])
    returns = np.array([0.01, -0.005, 0.02])
    funding_rates = np.array([0.0001, 0.0001, 0.0001])

    result = calc.calculate_portfolio_return(
        weights=weights,
        asset_returns=returns,
        funding_rates=funding_rates,
        include_costs=True,
    )

    # Funding cost: (0.5*0.0001) + (-0.3*0.0001) + (0.2*0.0001) = 0.00004
    assert abs(result['funding_cost'] - 0.00004) < 1e-10
    print("✓ Funding rates test passed")


def test_weight_validation():
    """Test weight validation."""
    calc = PnLCalculator()

    # Valid weights
    assert calc.validate_weights(np.array([0.5, 0.3, 0.2])) is True

    # Invalid: NaN
    assert calc.validate_weights(np.array([0.5, np.nan, 0.5])) is False

    # Invalid: Inf
    assert calc.validate_weights(np.array([0.5, np.inf, 0.5])) is False

    # Valid with leverage
    assert calc.validate_weights(np.array([0.8, 0.8, 0.4]), max_leverage=3.0) is True

    # Invalid: too much leverage
    assert calc.validate_weights(np.array([2.0, 2.0, 0.5]), max_leverage=3.0) is False

    print("✓ Weight validation test passed")


def test_pnl_result():
    """Test PnLResult container."""
    result = PnLResult(
        gross_return=0.01,
        funding_cost=0.0,
        transaction_cost=0.0,
        net_return=0.01,
        turnover=0.0,
        prev_nav=100_000.0,
        new_nav=101_000.0,
    )

    assert abs(result.pnl - 1_000.0) < 1e-6
    assert 'pnl' in result.to_dict()
    assert 'PnLResult' in repr(result)
    print("✓ PnLResult container test passed")


def test_complete_workflow():
    """Test complete PnL calculation workflow."""
    calc = PnLCalculator(transaction_cost_rate=0.0004, slippage_rate=0.0001)

    # Initial state
    nav = 100_000.0
    prev_weights = np.array([0.5, 0.5])
    new_weights = np.array([0.7, 0.3])
    returns = np.array([0.02, -0.01])

    # Calculate return
    result = calc.calculate_portfolio_return(
        weights=new_weights,
        asset_returns=returns,
        prev_weights=prev_weights,
        include_costs=True,
    )

    # Update NAV
    new_nav = calc.update_nav(nav, result['net_return'])

    # Verify
    # Gross return: (0.7*0.02) + (0.3*-0.01) = 0.011
    assert abs(result['gross_return'] - 0.011) < 1e-10
    # Turnover: 0.4, Cost (transaction + slippage): 0.0002
    # Net return: 0.011 - 0.0002 = 0.0108
    assert abs(result['net_return'] - 0.0108) < 1e-10
    # New NAV: 101,080
    assert abs(new_nav - 101_080.0) < 1e-6

    print("✓ Complete workflow test passed")


def main():
    """Run all tests."""
    print("Running PnLCalculator validation tests...\n")

    tests = [
        test_simple_portfolio_return,
        test_portfolio_return_with_costs,
        test_nav_update,
        test_transaction_cost_calculation,
        test_leveraged_portfolio,
        test_short_positions,
        test_funding_rates,
        test_weight_validation,
        test_pnl_result,
        test_complete_workflow,
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
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")


if __name__ == '__main__':
    main()
