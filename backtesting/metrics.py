"""
Backtesting Metrics - Performance and risk metrics calculation.

Calculates:
- Returns (total, annual, monthly)
- Risk metrics (Sharpe, Sortino, Max Drawdown)
- Trading metrics (win rate, profit factor)
- Portfolio metrics (turnover, exposure)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging


class PerformanceMetrics:
    """
    Calculates performance and risk metrics for backtesting.

    All metrics assume:
    - Returns are in decimal form (0.01 = 1%)
    - Risk-free rate default: 0 (crypto has no risk-free rate)
    - Annualization factor: 365 days (crypto trades 24/7)
    """

    # Annualization factors
    MINUTES_PER_YEAR = 525600  # 365 * 24 * 60
    HOURS_PER_YEAR = 8760      # 365 * 24
    DAYS_PER_YEAR = 365

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        annualize: bool = True,
    ):
        """
        Initialize metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 0 for crypto)
            annualize: Whether to annualize metrics
        """
        self.risk_free_rate = risk_free_rate
        self.annualize = annualize

        self.logger = logging.getLogger(self.__class__.__name__)

    # ========== Return Metrics ==========

    def total_return(self, returns: np.ndarray) -> float:
        """
        Calculate total cumulative return.

        Args:
            returns: Array of period returns

        Returns:
            Total return (e.g., 0.5 = 50%)
        """
        return np.prod(1 + returns) - 1

    def annualized_return(
        self,
        returns: np.ndarray,
        periods_per_year: float = None,
    ) -> float:
        """
        Calculate annualized return.

        Args:
            returns: Array of period returns
            periods_per_year: Number of periods per year

        Returns:
            Annualized return
        """
        if periods_per_year is None:
            periods_per_year = self.DAYS_PER_YEAR

        total = self.total_return(returns)
        n_periods = len(returns)

        if n_periods == 0:
            return 0.0

        years = n_periods / periods_per_year
        if years <= 0:
            return 0.0

        return (1 + total) ** (1 / years) - 1

    def cagr(
        self,
        equity_curve: np.ndarray,
        periods_per_year: float = None,
    ) -> float:
        """
        Calculate Compound Annual Growth Rate.

        Args:
            equity_curve: Array of portfolio values
            periods_per_year: Number of periods per year

        Returns:
            CAGR
        """
        if periods_per_year is None:
            periods_per_year = self.DAYS_PER_YEAR

        if len(equity_curve) < 2 or equity_curve[0] <= 0:
            return 0.0

        n_periods = len(equity_curve) - 1
        years = n_periods / periods_per_year

        if years <= 0:
            return 0.0

        return (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1

    # ========== Risk Metrics ==========

    def volatility(
        self,
        returns: np.ndarray,
        periods_per_year: float = None,
    ) -> float:
        """
        Calculate annualized volatility.

        Args:
            returns: Array of period returns
            periods_per_year: Number of periods per year

        Returns:
            Annualized volatility
        """
        if periods_per_year is None:
            periods_per_year = self.DAYS_PER_YEAR

        if len(returns) < 2:
            return 0.0

        std = np.std(returns, ddof=1)

        if self.annualize:
            return std * np.sqrt(periods_per_year)
        return std

    def downside_volatility(
        self,
        returns: np.ndarray,
        target: float = 0.0,
        periods_per_year: float = None,
    ) -> float:
        """
        Calculate downside volatility (semi-deviation).

        Args:
            returns: Array of period returns
            target: Target return (default 0)
            periods_per_year: Number of periods per year

        Returns:
            Downside volatility
        """
        if periods_per_year is None:
            periods_per_year = self.DAYS_PER_YEAR

        downside = returns[returns < target] - target

        if len(downside) < 2:
            return 0.0

        std = np.sqrt(np.mean(downside ** 2))

        if self.annualize:
            return std * np.sqrt(periods_per_year)
        return std

    def sharpe_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: float = None,
    ) -> float:
        """
        Calculate Sharpe Ratio.

        Formula: (mean_return - rf) / std_dev

        Args:
            returns: Array of period returns
            periods_per_year: Number of periods per year

        Returns:
            Sharpe ratio
        """
        if periods_per_year is None:
            periods_per_year = self.DAYS_PER_YEAR

        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        # Convert risk-free rate to period rate
        rf_period = self.risk_free_rate / periods_per_year

        sharpe = (mean_return - rf_period) / std_return

        if self.annualize:
            return sharpe * np.sqrt(periods_per_year)
        return sharpe

    def sortino_ratio(
        self,
        returns: np.ndarray,
        target: float = 0.0,
        periods_per_year: float = None,
    ) -> float:
        """
        Calculate Sortino Ratio.

        Formula: (mean_return - target) / downside_std

        Args:
            returns: Array of period returns
            target: Target return
            periods_per_year: Number of periods per year

        Returns:
            Sortino ratio
        """
        if periods_per_year is None:
            periods_per_year = self.DAYS_PER_YEAR

        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        downside_std = self.downside_volatility(returns, target, 1.0)  # Non-annualized

        if downside_std == 0:
            return 0.0

        sortino = (mean_return - target) / downside_std

        if self.annualize:
            return sortino * np.sqrt(periods_per_year)
        return sortino

    def calmar_ratio(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray = None,
        periods_per_year: float = None,
    ) -> float:
        """
        Calculate Calmar Ratio.

        Formula: CAGR / Max Drawdown

        Args:
            returns: Array of period returns
            equity_curve: Array of portfolio values (optional)
            periods_per_year: Number of periods per year

        Returns:
            Calmar ratio
        """
        if periods_per_year is None:
            periods_per_year = self.DAYS_PER_YEAR

        # Build equity curve if not provided
        if equity_curve is None:
            equity_curve = np.cumprod(1 + returns)

        cagr = self.cagr(equity_curve, periods_per_year)
        max_dd = self.max_drawdown(equity_curve)

        if max_dd == 0:
            return 0.0

        return cagr / abs(max_dd)

    def cvar(
        self,
        returns: np.ndarray,
        alpha: float = 0.05,
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

        CVaR is the expected return in the worst alpha% of cases.
        For example, CVaR(0.05) is the average return in the worst 5% of periods.

        Args:
            returns: Array of period returns
            alpha: Confidence level (default: 0.05 for 95% CVaR)

        Returns:
            CVaR (negative value indicates loss)
        """
        if len(returns) == 0:
            return 0.0

        # Sort returns from worst to best
        sorted_returns = np.sort(returns)

        # Get cutoff index for worst alpha% of returns
        cutoff = max(1, int(len(sorted_returns) * alpha))

        # CVaR is the average of the worst alpha% returns
        cvar_value = np.mean(sorted_returns[:cutoff])

        return float(cvar_value)

    # ========== Drawdown Metrics ==========

    def max_drawdown(self, equity_curve: np.ndarray) -> float:
        """
        Calculate Maximum Drawdown.

        Args:
            equity_curve: Array of portfolio values

        Returns:
            Maximum drawdown (negative value)
        """
        if len(equity_curve) < 2:
            return 0.0

        # Running maximum
        running_max = np.maximum.accumulate(equity_curve)

        # Drawdown series
        drawdowns = (equity_curve - running_max) / running_max

        return np.min(drawdowns)

    def max_drawdown_duration(
        self,
        equity_curve: np.ndarray,
    ) -> int:
        """
        Calculate Maximum Drawdown Duration (periods).

        Args:
            equity_curve: Array of portfolio values

        Returns:
            Maximum duration in periods
        """
        if len(equity_curve) < 2:
            return 0

        # Running maximum
        running_max = np.maximum.accumulate(equity_curve)

        # Find periods where we're in drawdown
        in_drawdown = equity_curve < running_max

        # Count consecutive drawdown periods
        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def drawdown_series(self, equity_curve: np.ndarray) -> np.ndarray:
        """
        Calculate drawdown series.

        Args:
            equity_curve: Array of portfolio values

        Returns:
            Array of drawdown values
        """
        running_max = np.maximum.accumulate(equity_curve)
        return (equity_curve - running_max) / running_max

    # ========== Trading Metrics ==========

    def win_rate(self, returns: np.ndarray) -> float:
        """
        Calculate win rate.

        Args:
            returns: Array of period returns

        Returns:
            Win rate (0-1)
        """
        if len(returns) == 0:
            return 0.0

        wins = np.sum(returns > 0)
        return wins / len(returns)

    def profit_factor(self, returns: np.ndarray) -> float:
        """
        Calculate profit factor.

        Formula: sum(profits) / sum(losses)

        Args:
            returns: Array of period returns

        Returns:
            Profit factor
        """
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())

        if losses == 0:
            return float('inf') if profits > 0 else 0.0

        return profits / losses

    def avg_win_loss_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate average win / average loss ratio.

        Args:
            returns: Array of period returns

        Returns:
            Win/loss ratio
        """
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.0

        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss == 0:
            return float('inf')

        return avg_win / avg_loss

    def expectancy(self, returns: np.ndarray) -> float:
        """
        Calculate expectancy (expected return per trade).

        Formula: win_rate * avg_win - loss_rate * avg_loss

        Args:
            returns: Array of period returns

        Returns:
            Expectancy
        """
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(returns) == 0:
            return 0.0

        win_rate = len(wins) / len(returns)
        loss_rate = len(losses) / len(returns)

        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0

        return win_rate * avg_win - loss_rate * avg_loss

    # ========== Portfolio Metrics ==========

    def turnover(
        self,
        positions: np.ndarray,
    ) -> float:
        """
        Calculate average portfolio turnover.

        Args:
            positions: Array of position sizes (T, assets) or (T,)

        Returns:
            Average turnover
        """
        if len(positions) < 2:
            return 0.0

        # Calculate position changes
        changes = np.abs(np.diff(positions, axis=0))

        if changes.ndim > 1:
            changes = changes.sum(axis=1)

        return np.mean(changes)

    def exposure(self, positions: np.ndarray) -> Dict:
        """
        Calculate exposure statistics.

        Args:
            positions: Array of position sizes

        Returns:
            Dict with exposure stats
        """
        if len(positions) == 0:
            return {'avg': 0, 'max': 0, 'min': 0}

        abs_positions = np.abs(positions)

        if abs_positions.ndim > 1:
            abs_positions = abs_positions.sum(axis=1)

        return {
            'avg': float(np.mean(abs_positions)),
            'max': float(np.max(abs_positions)),
            'min': float(np.min(abs_positions)),
        }

    # ========== Comprehensive Report ==========

    def calculate_all(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray = None,
        positions: np.ndarray = None,
        periods_per_year: float = None,
    ) -> Dict:
        """
        Calculate all metrics.

        Args:
            returns: Array of period returns
            equity_curve: Array of portfolio values (optional)
            positions: Array of positions (optional)
            periods_per_year: Number of periods per year

        Returns:
            Dict with all metrics
        """
        if periods_per_year is None:
            periods_per_year = self.DAYS_PER_YEAR

        # Build equity curve if not provided
        if equity_curve is None:
            equity_curve = np.cumprod(1 + returns) if len(returns) > 0 else np.array([1.0])

        metrics = {
            # Returns
            'total_return': self.total_return(returns),
            'annualized_return': self.annualized_return(returns, periods_per_year),
            'cagr': self.cagr(equity_curve, periods_per_year),

            # Risk
            'volatility': self.volatility(returns, periods_per_year),
            'downside_volatility': self.downside_volatility(returns, 0, periods_per_year),
            'sharpe_ratio': self.sharpe_ratio(returns, periods_per_year),
            'sortino_ratio': self.sortino_ratio(returns, 0, periods_per_year),
            'calmar_ratio': self.calmar_ratio(returns, equity_curve, periods_per_year),
            'cvar_95': self.cvar(returns, alpha=0.05),

            # Drawdown
            'max_drawdown': self.max_drawdown(equity_curve),
            'max_drawdown_duration': self.max_drawdown_duration(equity_curve),

            # Trading
            'win_rate': self.win_rate(returns),
            'profit_factor': self.profit_factor(returns),
            'avg_win_loss_ratio': self.avg_win_loss_ratio(returns),
            'expectancy': self.expectancy(returns),

            # Stats
            'n_periods': len(returns),
            'n_positive': int(np.sum(returns > 0)),
            'n_negative': int(np.sum(returns < 0)),
            'best_return': float(np.max(returns)) if len(returns) > 0 else 0,
            'worst_return': float(np.min(returns)) if len(returns) > 0 else 0,
        }

        # Add position metrics if available
        if positions is not None:
            metrics['turnover'] = self.turnover(positions)
            metrics['exposure'] = self.exposure(positions)

        return metrics

    def print_report(self, metrics: Dict):
        """Print metrics report to console."""
        print("\n" + "=" * 50)
        print("PERFORMANCE REPORT")
        print("=" * 50)

        print("\nReturns:")
        print(f"  Total Return:      {metrics['total_return']:.2%}")
        print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"  CAGR:              {metrics['cagr']:.2%}")

        print("\nRisk:")
        print(f"  Volatility:        {metrics['volatility']:.2%}")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:     {metrics['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio:      {metrics['calmar_ratio']:.2f}")
        print(f"  CVaR (95%):        {metrics['cvar_95']:.2%}")

        print("\nDrawdown:")
        print(f"  Max Drawdown:      {metrics['max_drawdown']:.2%}")
        print(f"  Max DD Duration:   {metrics['max_drawdown_duration']} periods")

        print("\nTrading:")
        print(f"  Win Rate:          {metrics['win_rate']:.2%}")
        print(f"  Profit Factor:     {metrics['profit_factor']:.2f}")
        print(f"  Expectancy:        {metrics['expectancy']:.4f}")

        print("\nStatistics:")
        print(f"  Total Periods:     {metrics['n_periods']}")
        print(f"  Winning Periods:   {metrics['n_positive']}")
        print(f"  Losing Periods:    {metrics['n_negative']}")
        print(f"  Best Period:       {metrics['best_return']:.2%}")
        print(f"  Worst Period:      {metrics['worst_return']:.2%}")

        print("\n" + "=" * 50)


# ========== Standalone Testing ==========

if __name__ == '__main__':
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.randn(252) * 0.02 + 0.001  # Daily returns, ~40% annual

    metrics = PerformanceMetrics()
    report = metrics.calculate_all(returns, periods_per_year=365)
    metrics.print_report(report)
