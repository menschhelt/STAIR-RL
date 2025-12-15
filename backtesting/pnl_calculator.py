"""
PnL Calculator - Unified PnL calculation for RL training and backtesting.

This module provides a single source of truth for all PnL calculations
across the codebase, eliminating duplication between BacktestEngine,
TradingEnv, and benchmark strategies.
"""

from typing import Dict, Optional
import numpy as np


class PnLCalculator:
    """
    Unified PnL calculation for both RL training and backtesting.

    Responsibilities:
    - Portfolio return calculation (weight-based)
    - Transaction cost modeling
    - Funding rate payments (for perpetual futures)
    - NAV updates
    - Turnover calculation

    Design Philosophy:
    - Stateless: All calculations are pure functions
    - Weight-based: Uses portfolio weights rather than positions
    - Consistent: Same logic for RL and backtesting
    """

    def __init__(
        self,
        transaction_cost_rate: float = 0.0004,  # 4 bps (taker fee)
        slippage_rate: float = 0.0001,          # 1 bp
        min_trade_size: float = 100.0,          # Minimum trade size in USDT
    ):
        """
        Initialize PnL calculator.

        Args:
            transaction_cost_rate: Transaction cost as ratio of notional (default: 4 bps)
            slippage_rate: Slippage as ratio of notional (default: 1 bp)
            min_trade_size: Minimum trade size in USDT (default: $100)
        """
        self.transaction_cost_rate = transaction_cost_rate
        self.slippage_rate = slippage_rate
        self.min_trade_size = min_trade_size

    def calculate_portfolio_return(
        self,
        weights: np.ndarray,              # (N,) current weights
        asset_returns: np.ndarray,        # (N,) asset returns
        prev_weights: Optional[np.ndarray] = None,  # (N,) previous weights
        funding_rates: Optional[np.ndarray] = None, # (N,) funding rates
        include_costs: bool = True,       # Whether to deduct transaction costs
    ) -> Dict[str, float]:
        """
        Calculate portfolio return for one step.

        This is the core PnL calculation used by both RL training and backtesting.

        Args:
            weights: Current portfolio weights (N,), should sum to ~1.0 (allowing leverage)
            asset_returns: Asset returns for this period (N,)
            prev_weights: Previous portfolio weights (N,), used for transaction cost calc
            funding_rates: Funding rates for perpetual futures (N,), optional
            include_costs: Whether to include transaction and funding costs

        Returns:
            Dictionary with:
                - gross_return: Portfolio return before costs
                - funding_cost: Funding rate payments (negative = cost, positive = income)
                - transaction_cost: Transaction cost as ratio of NAV
                - net_return: Final portfolio return after all costs
                - turnover: Portfolio turnover (sum of absolute weight changes)

        Example:
            >>> calc = PnLCalculator()
            >>> weights = np.array([0.5, 0.5])
            >>> returns = np.array([0.01, -0.005])
            >>> result = calc.calculate_portfolio_return(weights, returns)
            >>> result['gross_return']
            0.0025  # (0.5 * 0.01) + (0.5 * -0.005)
        """
        # Validate inputs
        assert len(weights) == len(asset_returns), \
            f"Shape mismatch: weights {weights.shape} vs returns {asset_returns.shape}"

        # Calculate gross return (before costs)
        gross_return = np.dot(weights, asset_returns)

        # Calculate funding cost
        if funding_rates is not None and include_costs:
            # Positive funding rate = longs pay shorts
            # So positive weights (longs) pay funding, negative weights (shorts) receive
            funding_cost = np.dot(weights, funding_rates)
        else:
            funding_cost = 0.0

        # Calculate transaction cost (includes slippage)
        if prev_weights is not None and include_costs:
            turnover = np.abs(weights - prev_weights).sum()
            # Total cost = transaction fee + slippage
            transaction_cost = turnover * (self.transaction_cost_rate + self.slippage_rate)
        else:
            turnover = 0.0
            transaction_cost = 0.0

        # Calculate net return
        net_return = gross_return - funding_cost - transaction_cost

        return {
            'gross_return': float(gross_return),
            'funding_cost': float(funding_cost),
            'transaction_cost': float(transaction_cost),
            'net_return': float(net_return),
            'turnover': float(turnover),
        }

    def update_nav(
        self,
        prev_nav: float,
        portfolio_return: float,
    ) -> float:
        """
        Update NAV based on portfolio return.

        Args:
            prev_nav: Previous NAV value
            portfolio_return: Portfolio return as decimal (e.g., 0.01 for 1%)

        Returns:
            Updated NAV

        Example:
            >>> calc = PnLCalculator()
            >>> calc.update_nav(100_000.0, 0.01)
            101000.0
        """
        return prev_nav * (1.0 + portfolio_return)

    def calculate_transaction_cost(
        self,
        prev_weights: np.ndarray,
        new_weights: np.ndarray,
        nav: float,
    ) -> Dict[str, float]:
        """
        Calculate transaction costs for rebalancing.

        This is useful for detailed trade analysis and reporting.

        Args:
            prev_weights: Previous portfolio weights (N,)
            new_weights: New target portfolio weights (N,)
            nav: Current portfolio NAV in USDT

        Returns:
            Dictionary with:
                - turnover: Portfolio turnover ratio (|w_new - w_old|.sum())
                - notional_traded: Notional value traded in USDT
                - transaction_cost: Transaction cost in USDT
                - slippage_cost: Slippage cost in USDT (if applicable)
                - total_cost: Total cost in USDT
                - cost_ratio: Total cost as ratio of NAV

        Example:
            >>> calc = PnLCalculator()
            >>> prev = np.array([0.5, 0.5])
            >>> new = np.array([0.7, 0.3])
            >>> result = calc.calculate_transaction_cost(prev, new, 100_000.0)
            >>> result['turnover']
            0.4  # |0.7-0.5| + |0.3-0.5| = 0.2 + 0.2
        """
        # Validate inputs
        assert len(prev_weights) == len(new_weights), \
            f"Shape mismatch: prev {prev_weights.shape} vs new {new_weights.shape}"

        # Calculate turnover
        weight_changes = np.abs(new_weights - prev_weights)
        turnover = weight_changes.sum()

        # Calculate notional traded
        notional_traded = turnover * nav

        # Calculate costs
        transaction_cost = notional_traded * self.transaction_cost_rate
        slippage_cost = notional_traded * self.slippage_rate
        total_cost = transaction_cost + slippage_cost

        # Cost as ratio of NAV
        cost_ratio = total_cost / nav if nav > 0 else 0.0

        return {
            'turnover': float(turnover),
            'notional_traded': float(notional_traded),
            'transaction_cost': float(transaction_cost),
            'slippage_cost': float(slippage_cost),
            'total_cost': float(total_cost),
            'cost_ratio': float(cost_ratio),
        }

    def calculate_turnover(
        self,
        prev_weights: np.ndarray,
        new_weights: np.ndarray,
    ) -> float:
        """
        Calculate portfolio turnover.

        Turnover is defined as the sum of absolute weight changes:
        turnover = Σ|w_new_i - w_old_i|

        Args:
            prev_weights: Previous portfolio weights (N,)
            new_weights: New portfolio weights (N,)

        Returns:
            Turnover ratio

        Example:
            >>> calc = PnLCalculator()
            >>> calc.calculate_turnover(
            ...     np.array([0.5, 0.5]),
            ...     np.array([0.7, 0.3])
            ... )
            0.4
        """
        return np.abs(new_weights - prev_weights).sum()

    def validate_weights(
        self,
        weights: np.ndarray,
        max_leverage: float = 3.0,
        tolerance: float = 0.01,
    ) -> bool:
        """
        Validate portfolio weights.

        Args:
            weights: Portfolio weights (N,)
            max_leverage: Maximum allowed leverage (default: 3x)
            tolerance: Tolerance for weight sum check (default: 1%)

        Returns:
            True if weights are valid, False otherwise

        Checks:
        - Weights sum is within (1 - tolerance, max_leverage + tolerance)
        - No NaN or Inf values
        - No individual weight > max_leverage
        """
        # Check for invalid values
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            return False

        # Check weight sum (allowing for leverage and small numerical errors)
        weight_sum = np.abs(weights).sum()
        if weight_sum < (1.0 - tolerance) or weight_sum > (max_leverage + tolerance):
            return False

        # Check individual weights
        if np.any(np.abs(weights) > max_leverage):
            return False

        return True


class PnLResult:
    """
    Container for PnL calculation results.

    This provides a structured way to pass PnL information between components.
    """

    def __init__(
        self,
        gross_return: float,
        funding_cost: float,
        transaction_cost: float,
        net_return: float,
        turnover: float,
        prev_nav: float,
        new_nav: float,
    ):
        self.gross_return = gross_return
        self.funding_cost = funding_cost
        self.transaction_cost = transaction_cost
        self.net_return = net_return
        self.turnover = turnover
        self.prev_nav = prev_nav
        self.new_nav = new_nav

    @property
    def pnl(self) -> float:
        """PnL in USDT."""
        return self.new_nav - self.prev_nav

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'gross_return': self.gross_return,
            'funding_cost': self.funding_cost,
            'transaction_cost': self.transaction_cost,
            'net_return': self.net_return,
            'turnover': self.turnover,
            'prev_nav': self.prev_nav,
            'new_nav': self.new_nav,
            'pnl': self.pnl,
        }

    def __repr__(self) -> str:
        return (
            f"PnLResult(net_return={self.net_return:.4f}, "
            f"turnover={self.turnover:.4f}, "
            f"nav={self.prev_nav:.0f}→{self.new_nav:.0f})"
        )
