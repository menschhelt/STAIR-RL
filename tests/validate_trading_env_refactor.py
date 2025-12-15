"""
Validate that TradingEnv refactoring to use PnLCalculator works correctly.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.trading_env import TradingEnv, EnvConfig


def create_test_data(n_steps=100, n_assets=5):
    """Create synthetic test data."""
    np.random.seed(42)

    data = {
        'states': np.random.randn(n_steps, n_assets, 36).astype(np.float32),
        'returns': np.random.randn(n_steps, n_assets) * 0.01,  # 1% std returns
        'prices': np.cumsum(np.random.randn(n_steps, n_assets) * 0.01, axis=0) + 100.0,
        'timestamps': np.arange(n_steps),
        'funding_rates': np.random.randn(n_steps, n_assets) * 0.0001,  # Small funding rates
    }

    return data


def test_env_initialization():
    """Test that environment initializes with PnLCalculator."""
    config = EnvConfig(n_assets=5)
    data = create_test_data(n_steps=100, n_assets=5)

    env = TradingEnv(config=config, data=data)

    # Check that PnLCalculator was initialized
    assert hasattr(env, 'pnl_calculator'), "PnLCalculator not initialized"
    assert env.pnl_calculator.transaction_cost_rate == config.transaction_cost_rate
    assert env.pnl_calculator.slippage_rate == config.slippage_rate

    print("✓ Environment initialization test passed")


def test_single_step():
    """Test that a single step works with PnLCalculator."""
    config = EnvConfig(n_assets=5, initial_nav=100_000.0)
    data = create_test_data(n_steps=100, n_assets=5)

    env = TradingEnv(config=config, data=data)
    obs, info = env.reset()

    # Take an action (equal weights)
    action = np.ones(5) * 0.2  # Equal weights

    obs, reward, terminated, truncated, info = env.step(action)

    # Verify that PnL was calculated
    assert 'port_return' in info
    assert 'transaction_cost' in info
    assert 'slippage_cost' in info
    assert 'total_cost' in info
    assert 'turnover' in info

    # Verify NAV was updated
    assert env._portfolio.nav != config.initial_nav

    print("✓ Single step test passed")


def test_multiple_steps():
    """Test that multiple steps work correctly."""
    config = EnvConfig(n_assets=5, initial_nav=100_000.0)
    data = create_test_data(n_steps=100, n_assets=5)

    env = TradingEnv(config=config, data=data)
    obs, info = env.reset()

    initial_nav = env._portfolio.nav
    total_return = 0.0

    # Take 10 random steps
    for i in range(10):
        action = np.random.randn(5) * 0.1  # Random actions
        action = np.clip(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)

        # Accumulate returns
        total_return += info['port_return']

        # Verify consistency
        assert not np.isnan(reward), f"NaN reward at step {i}"
        assert not np.isnan(env._portfolio.nav), f"NaN NAV at step {i}"

        if terminated or truncated:
            break

    # Verify final NAV is consistent with accumulated returns
    expected_nav = initial_nav * np.prod([1 + r for r in env._episode_returns])
    actual_nav = env._portfolio.nav

    relative_error = abs(expected_nav - actual_nav) / initial_nav
    assert relative_error < 1e-6, \
        f"NAV inconsistency: expected {expected_nav:.2f}, got {actual_nav:.2f}"

    print("✓ Multiple steps test passed")


def test_pnl_calculation_consistency():
    """Test that PnL calculation is consistent with manual calculation."""
    config = EnvConfig(
        n_assets=3,
        initial_nav=100_000.0,
        transaction_cost_rate=0.0004,
        slippage_rate=0.0001,
    )

    # Create deterministic data
    np.random.seed(123)
    data = create_test_data(n_steps=10, n_assets=3)

    env = TradingEnv(config=config, data=data)
    obs, info = env.reset()

    # Take first step with known weights
    action1 = np.array([0.5, 0.3, 0.2])
    obs1, reward1, _, _, info1 = env.step(action1)

    # Take second step with different weights
    action2 = np.array([0.4, 0.4, 0.2])
    obs2, reward2, _, _, info2 = env.step(action2)

    # Manual calculation for second step
    prev_weights = env.pnl_calculator.validate_weights(action1, tolerance=0.5)  # Should be True
    turnover_manual = np.abs(action2 - action1).sum()  # Should match info2['turnover']

    assert abs(info2['turnover'] - turnover_manual) < 1e-10, \
        f"Turnover mismatch: {info2['turnover']} vs {turnover_manual}"

    print("✓ PnL calculation consistency test passed")


def test_no_compute_portfolio_return_method():
    """Verify that the old _compute_portfolio_return method was removed."""
    from environments.trading_env import TradingEnv
    import inspect

    # Check that _compute_portfolio_return is NOT in the class methods
    methods = [name for name, _ in inspect.getmembers(TradingEnv, predicate=inspect.ismethod)]

    # It should not exist as a method anymore
    has_old_method = '_compute_portfolio_return' in [
        name for name in dir(TradingEnv)
        if callable(getattr(TradingEnv, name, None)) and not name.startswith('__')
    ]

    assert not has_old_method, "Old _compute_portfolio_return method still exists!"

    print("✓ Old method removal test passed")


def main():
    """Run all validation tests."""
    print("Validating TradingEnv refactoring to use PnLCalculator...\n")

    tests = [
        test_env_initialization,
        test_single_step,
        test_multiple_steps,
        test_pnl_calculation_consistency,
        test_no_compute_portfolio_return_method,
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
        print("\n✓ All TradingEnv refactoring tests passed!")
        print("Phase 2: TradingEnv refactoring complete!")


if __name__ == '__main__':
    main()
