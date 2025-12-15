"""
Unit tests for PnLCalculator.

Tests the core PnL calculation logic that is shared between
BacktestEngine, TradingEnv, and benchmark strategies.
"""

import numpy as np
import pytest
from backtesting.pnl_calculator import PnLCalculator, PnLResult


class TestPnLCalculator:
    """Test cases for PnLCalculator."""

    def setup_method(self):
        """Create calculator instance for tests."""
        self.calc = PnLCalculator(
            transaction_cost_rate=0.0004,  # 4 bps
            slippage_rate=0.0001,          # 1 bp
        )

    def test_simple_portfolio_return(self):
        """Test basic portfolio return calculation without costs."""
        weights = np.array([0.5, 0.5])
        returns = np.array([0.01, -0.005])

        result = self.calc.calculate_portfolio_return(
            weights=weights,
            asset_returns=returns,
            include_costs=False,
        )

        # Expected: (0.5 * 0.01) + (0.5 * -0.005) = 0.005 - 0.0025 = 0.0025
        assert abs(result['gross_return'] - 0.0025) < 1e-10
        assert result['funding_cost'] == 0.0
        assert result['transaction_cost'] == 0.0
        assert abs(result['net_return'] - 0.0025) < 1e-10
        assert result['turnover'] == 0.0

    def test_portfolio_return_with_transaction_cost(self):
        """Test portfolio return with transaction costs."""
        prev_weights = np.array([0.5, 0.5])
        new_weights = np.array([0.7, 0.3])
        returns = np.array([0.01, -0.005])

        result = self.calc.calculate_portfolio_return(
            weights=new_weights,
            asset_returns=returns,
            prev_weights=prev_weights,
            include_costs=True,
        )

        # Gross return: (0.7 * 0.01) + (0.3 * -0.005) = 0.007 - 0.0015 = 0.0055
        assert abs(result['gross_return'] - 0.0055) < 1e-10

        # Turnover: |0.7-0.5| + |0.3-0.5| = 0.2 + 0.2 = 0.4
        assert abs(result['turnover'] - 0.4) < 1e-10

        # Transaction cost: 0.4 * 0.0004 = 0.00016
        assert abs(result['transaction_cost'] - 0.00016) < 1e-10

        # Net return: 0.0055 - 0.00016 = 0.00534
        assert abs(result['net_return'] - 0.00534) < 1e-10

    def test_portfolio_return_with_funding_rates(self):
        """Test portfolio return with funding rate payments."""
        weights = np.array([0.5, -0.3, 0.2])  # Long, Short, Long
        returns = np.array([0.01, -0.005, 0.02])
        funding_rates = np.array([0.0001, 0.0001, 0.0001])  # 0.01% funding

        result = self.calc.calculate_portfolio_return(
            weights=weights,
            asset_returns=returns,
            funding_rates=funding_rates,
            include_costs=True,
        )

        # Funding cost: (0.5 * 0.0001) + (-0.3 * 0.0001) + (0.2 * 0.0001)
        #             = 0.00005 - 0.00003 + 0.00002 = 0.00004
        # Longs pay 0.00005 + 0.00002 = 0.00007
        # Shorts receive 0.00003
        # Net cost: 0.00004
        assert abs(result['funding_cost'] - 0.00004) < 1e-10

    def test_leveraged_portfolio(self):
        """Test portfolio with leverage (weights sum > 1)."""
        weights = np.array([0.8, 0.8, 0.4])  # 2x leverage
        returns = np.array([0.01, -0.005, 0.02])

        result = self.calc.calculate_portfolio_return(
            weights=weights,
            asset_returns=returns,
            include_costs=False,
        )

        # Expected: (0.8 * 0.01) + (0.8 * -0.005) + (0.4 * 0.02)
        #         = 0.008 - 0.004 + 0.008 = 0.012
        assert abs(result['gross_return'] - 0.012) < 1e-10

    def test_short_positions(self):
        """Test portfolio with short positions (negative weights)."""
        weights = np.array([0.5, -0.3, 0.8])  # Long, Short, Long
        returns = np.array([0.01, 0.02, -0.005])

        result = self.calc.calculate_portfolio_return(
            weights=weights,
            asset_returns=returns,
            include_costs=False,
        )

        # Expected: (0.5 * 0.01) + (-0.3 * 0.02) + (0.8 * -0.005)
        #         = 0.005 - 0.006 - 0.004 = -0.005
        assert abs(result['gross_return'] - (-0.005)) < 1e-10

    def test_nav_update(self):
        """Test NAV update calculation."""
        prev_nav = 100_000.0
        portfolio_return = 0.01  # 1% return

        new_nav = self.calc.update_nav(prev_nav, portfolio_return)

        assert abs(new_nav - 101_000.0) < 1e-6

    def test_nav_update_negative_return(self):
        """Test NAV update with negative return."""
        prev_nav = 100_000.0
        portfolio_return = -0.05  # -5% return

        new_nav = self.calc.update_nav(prev_nav, portfolio_return)

        assert abs(new_nav - 95_000.0) < 1e-6

    def test_transaction_cost_calculation(self):
        """Test detailed transaction cost breakdown."""
        prev_weights = np.array([0.5, 0.5])
        new_weights = np.array([0.7, 0.3])
        nav = 100_000.0

        result = self.calc.calculate_transaction_cost(
            prev_weights=prev_weights,
            new_weights=new_weights,
            nav=nav,
        )

        # Turnover: |0.7-0.5| + |0.3-0.5| = 0.4
        assert abs(result['turnover'] - 0.4) < 1e-10

        # Notional traded: 0.4 * 100,000 = 40,000
        assert abs(result['notional_traded'] - 40_000.0) < 1e-6

        # Transaction cost: 40,000 * 0.0004 = 16.0
        assert abs(result['transaction_cost'] - 16.0) < 1e-6

        # Slippage cost: 40,000 * 0.0001 = 4.0
        assert abs(result['slippage_cost'] - 4.0) < 1e-6

        # Total cost: 16 + 4 = 20
        assert abs(result['total_cost'] - 20.0) < 1e-6

        # Cost ratio: 20 / 100,000 = 0.0002
        assert abs(result['cost_ratio'] - 0.0002) < 1e-10

    def test_turnover_calculation(self):
        """Test turnover calculation."""
        prev_weights = np.array([0.3, 0.4, 0.3])
        new_weights = np.array([0.5, 0.2, 0.3])

        turnover = self.calc.calculate_turnover(prev_weights, new_weights)

        # Expected: |0.5-0.3| + |0.2-0.4| + |0.3-0.3| = 0.2 + 0.2 + 0.0 = 0.4
        assert abs(turnover - 0.4) < 1e-10

    def test_zero_turnover(self):
        """Test case with no rebalancing."""
        weights = np.array([0.5, 0.5])
        returns = np.array([0.01, -0.005])

        result = self.calc.calculate_portfolio_return(
            weights=weights,
            asset_returns=returns,
            prev_weights=weights,  # Same as current
            include_costs=True,
        )

        assert result['turnover'] == 0.0
        assert result['transaction_cost'] == 0.0

    def test_validate_weights_normal(self):
        """Test weight validation for normal case."""
        weights = np.array([0.5, 0.3, 0.2])
        assert self.calc.validate_weights(weights) is True

    def test_validate_weights_leveraged(self):
        """Test weight validation with leverage."""
        weights = np.array([0.8, 0.8, 0.4])  # 2x leverage
        assert self.calc.validate_weights(weights, max_leverage=3.0) is True

    def test_validate_weights_too_much_leverage(self):
        """Test weight validation fails with excessive leverage."""
        weights = np.array([2.0, 2.0, 0.5])  # 4.5x leverage
        assert self.calc.validate_weights(weights, max_leverage=3.0) is False

    def test_validate_weights_with_nan(self):
        """Test weight validation fails with NaN."""
        weights = np.array([0.5, np.nan, 0.5])
        assert self.calc.validate_weights(weights) is False

    def test_validate_weights_with_inf(self):
        """Test weight validation fails with Inf."""
        weights = np.array([0.5, np.inf, 0.5])
        assert self.calc.validate_weights(weights) is False

    def test_validate_weights_sum_too_small(self):
        """Test weight validation fails when sum is too small."""
        weights = np.array([0.1, 0.2, 0.1])  # Sum = 0.4 < 1.0
        assert self.calc.validate_weights(weights, tolerance=0.3) is True
        assert self.calc.validate_weights(weights, tolerance=0.1) is False

    def test_complete_workflow(self):
        """Test complete PnL calculation workflow."""
        # Initial state
        nav = 100_000.0
        prev_weights = np.array([0.5, 0.5])

        # New weights and returns
        new_weights = np.array([0.7, 0.3])
        returns = np.array([0.02, -0.01])

        # Calculate return
        result = self.calc.calculate_portfolio_return(
            weights=new_weights,
            asset_returns=returns,
            prev_weights=prev_weights,
            include_costs=True,
        )

        # Update NAV
        new_nav = self.calc.update_nav(nav, result['net_return'])

        # Verify
        # Gross return: (0.7 * 0.02) + (0.3 * -0.01) = 0.014 - 0.003 = 0.011
        assert abs(result['gross_return'] - 0.011) < 1e-10

        # Turnover: |0.7-0.5| + |0.3-0.5| = 0.4
        # Transaction cost: 0.4 * 0.0004 = 0.00016
        assert abs(result['transaction_cost'] - 0.00016) < 1e-10

        # Net return: 0.011 - 0.00016 = 0.01084
        assert abs(result['net_return'] - 0.01084) < 1e-10

        # New NAV: 100,000 * (1 + 0.01084) = 101,084
        assert abs(new_nav - 101_084.0) < 1e-6


class TestPnLResult:
    """Test cases for PnLResult container."""

    def test_pnl_result_creation(self):
        """Test creating PnLResult."""
        result = PnLResult(
            gross_return=0.01,
            funding_cost=0.0001,
            transaction_cost=0.0002,
            net_return=0.0097,
            turnover=0.5,
            prev_nav=100_000.0,
            new_nav=100_970.0,
        )

        assert result.gross_return == 0.01
        assert result.funding_cost == 0.0001
        assert result.transaction_cost == 0.0002
        assert result.net_return == 0.0097
        assert result.turnover == 0.5
        assert result.prev_nav == 100_000.0
        assert result.new_nav == 100_970.0

    def test_pnl_property(self):
        """Test PnL property calculation."""
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

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = PnLResult(
            gross_return=0.01,
            funding_cost=0.0,
            transaction_cost=0.0,
            net_return=0.01,
            turnover=0.0,
            prev_nav=100_000.0,
            new_nav=101_000.0,
        )

        d = result.to_dict()

        assert d['gross_return'] == 0.01
        assert d['net_return'] == 0.01
        assert d['prev_nav'] == 100_000.0
        assert d['new_nav'] == 101_000.0
        assert abs(d['pnl'] - 1_000.0) < 1e-6

    def test_repr(self):
        """Test string representation."""
        result = PnLResult(
            gross_return=0.01,
            funding_cost=0.0,
            transaction_cost=0.0,
            net_return=0.01,
            turnover=0.5,
            prev_nav=100_000.0,
            new_nav=101_000.0,
        )

        repr_str = repr(result)
        assert 'PnLResult' in repr_str
        assert '0.0100' in repr_str  # net_return
        assert '0.5000' in repr_str  # turnover


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Create calculator instance for tests."""
        self.calc = PnLCalculator()

    def test_empty_arrays(self):
        """Test with empty arrays."""
        weights = np.array([])
        returns = np.array([])

        result = self.calc.calculate_portfolio_return(weights, returns)

        assert result['gross_return'] == 0.0
        assert result['net_return'] == 0.0

    def test_single_asset(self):
        """Test with single asset."""
        weights = np.array([1.0])
        returns = np.array([0.05])

        result = self.calc.calculate_portfolio_return(weights, returns)

        assert abs(result['gross_return'] - 0.05) < 1e-10

    def test_very_small_numbers(self):
        """Test with very small numbers."""
        weights = np.array([1e-8, 1.0 - 1e-8])
        returns = np.array([0.01, 0.02])

        result = self.calc.calculate_portfolio_return(weights, returns)

        # Should be dominated by second asset
        assert abs(result['gross_return'] - 0.02) < 1e-6

    def test_extreme_returns(self):
        """Test with extreme returns."""
        weights = np.array([0.5, 0.5])
        returns = np.array([5.0, -0.99])  # 500% and -99%

        result = self.calc.calculate_portfolio_return(weights, returns)

        # (0.5 * 5.0) + (0.5 * -0.99) = 2.5 - 0.495 = 2.005
        assert abs(result['gross_return'] - 2.005) < 1e-10

    def test_shape_mismatch_error(self):
        """Test that shape mismatch raises assertion error."""
        weights = np.array([0.5, 0.5])
        returns = np.array([0.01, 0.02, 0.03])  # Wrong shape!

        with pytest.raises(AssertionError):
            self.calc.calculate_portfolio_return(weights, returns)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
