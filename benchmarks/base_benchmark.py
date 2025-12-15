"""
Base Benchmark - Abstract base class for all portfolio benchmarks.

All benchmark strategies inherit from this class and implement
the compute_weights() method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import logging

from backtesting.metrics import PerformanceMetrics
from backtesting.pnl_calculator import PnLCalculator

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark strategies."""
    n_assets: int = 20
    rebalance_freq: str = 'daily'  # 'daily', '5min', 'hourly'
    target_leverage: float = 2.0
    transaction_cost: float = 0.0004  # 0.04% (Binance taker)
    slippage: float = 0.0005  # 0.05%
    min_position_value: float = 100.0  # Minimum $100 per position
    max_single_weight: float = 0.5  # Max 50% in single asset
    lookback_days: int = 252  # 1 year for estimation


@dataclass
class BacktestResult:
    """Results from running a backtest."""
    strategy_name: str

    # Time series
    nav_series: pd.Series = None
    returns_series: pd.Series = None
    weights_history: pd.DataFrame = None

    # Performance metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    cvar_95: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    downside_volatility: float = 0.0

    # Trading metrics
    total_turnover: float = 0.0
    avg_turnover: float = 0.0
    total_cost: float = 0.0

    # Additional info
    num_rebalances: int = 0
    start_date: datetime = None
    end_date: datetime = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for comparison."""
        return {
            'strategy': self.strategy_name,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'cvar_95': self.cvar_95,
            'volatility': self.volatility,
            'total_turnover': self.total_turnover,
            'total_cost': self.total_cost,
        }


class BaseBenchmark(ABC):
    """
    Abstract base class for portfolio benchmark strategies.

    All benchmark strategies must implement:
    - compute_weights(): Calculate target portfolio weights

    The base class provides:
    - run_backtest(): Execute backtest with given data
    - _compute_metrics(): Calculate performance metrics
    - _apply_constraints(): Apply position constraints
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize benchmark strategy.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.name = self.__class__.__name__

        # Initialize PnL calculator (unified with TradingEnv and BacktestEngine)
        self.pnl_calculator = PnLCalculator(
            transaction_cost_rate=self.config.transaction_cost,
            slippage_rate=self.config.slippage,
        )

    @abstractmethod
    def compute_weights(
        self,
        timestamp: pd.Timestamp,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute target portfolio weights.

        Args:
            timestamp: Current timestamp
            prices: Current asset prices (n_assets,)
            features: Dictionary of features
                - 'returns': Historical returns (lookback, n_assets)
                - 'market_caps': Market capitalizations (n_assets,)
                - 'factors': Factor values (n_assets, n_factors)
                - 'sentiment': Sentiment scores (n_assets,)
            current_weights: Current portfolio weights (n_assets,)

        Returns:
            weights: Target portfolio weights (n_assets,)
                     Range: [-1, 1] for long-short support
        """
        pass

    def run_backtest(
        self,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        initial_nav: float = 100_000.0,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with columns:
                - timestamp: Datetime index
                - symbol: Asset symbol
                - close: Close price
                - returns: Period returns
                - market_cap: Market capitalization (optional)
                - factors: Factor values (optional)
                - sentiment: Sentiment scores (optional)
            start_date: Backtest start date
            end_date: Backtest end date
            initial_nav: Initial portfolio value

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Running backtest for {self.name}: {start_date} to {end_date}")

        # Filter data by date range
        data = data[(data.index >= start_date) & (data.index <= end_date)]

        # Get unique timestamps for rebalancing
        timestamps = data.index.unique()

        # Initialize tracking variables
        nav = initial_nav
        nav_history = []
        returns_history = []
        weights_history = []
        turnover_history = []

        current_weights = np.zeros(self.config.n_assets)

        for i, ts in enumerate(timestamps):
            # Get current data slice
            ts_data = data.loc[ts]

            # Prepare features
            prices, features = self._prepare_features(data, ts, i)

            if prices is None:
                continue

            # Compute target weights
            target_weights = self.compute_weights(
                timestamp=ts,
                prices=prices,
                features=features,
                current_weights=current_weights,
            )

            # Apply constraints
            target_weights = self._apply_constraints(target_weights)

            # Get returns for this period
            if 'returns' in features:
                period_returns = features['returns'][-1] if len(features['returns']) > 0 else np.zeros(self.config.n_assets)
            else:
                period_returns = np.zeros(self.config.n_assets)

            # Calculate portfolio return using PnLCalculator (unified with TradingEnv)
            pnl_result = self.pnl_calculator.calculate_portfolio_return(
                weights=target_weights,
                asset_returns=period_returns,
                prev_weights=current_weights,
                funding_rates=None,  # No funding for spot trading
                include_costs=True,
            )

            port_return = pnl_result['net_return']
            turnover = pnl_result['turnover']

            # Update NAV
            nav = self.pnl_calculator.update_nav(nav, port_return)

            # Record history
            nav_history.append({'timestamp': ts, 'nav': nav})
            returns_history.append({'timestamp': ts, 'return': port_return})
            weights_history.append({'timestamp': ts, **{f'w_{j}': w for j, w in enumerate(target_weights)}})
            turnover_history.append(turnover)

            # Update current weights
            current_weights = target_weights

        # Convert to DataFrames
        nav_df = pd.DataFrame(nav_history).set_index('timestamp')['nav']
        returns_df = pd.DataFrame(returns_history).set_index('timestamp')['return']
        weights_df = pd.DataFrame(weights_history).set_index('timestamp')

        # Compute metrics
        result = self._compute_metrics(
            nav_series=nav_df,
            returns_series=returns_df,
            weights_df=weights_df,
            turnover_history=turnover_history,
            initial_nav=initial_nav,
        )

        result.strategy_name = self.name
        result.start_date = pd.Timestamp(start_date)
        result.end_date = pd.Timestamp(end_date)

        return result

    def _prepare_features(
        self,
        data: pd.DataFrame,
        timestamp: pd.Timestamp,
        idx: int,
    ) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Prepare features for weight computation.

        Returns:
            prices: Current prices (n_assets,)
            features: Dictionary of feature arrays
        """
        try:
            # Get current slice
            ts_data = data.loc[timestamp]

            # Handle case where ts_data is a single row or multiple rows
            if isinstance(ts_data, pd.Series):
                prices = np.array([ts_data.get('close', 0.0)])
            else:
                prices = ts_data['close'].values[:self.config.n_assets]

            # Pad if needed
            if len(prices) < self.config.n_assets:
                prices = np.pad(prices, (0, self.config.n_assets - len(prices)), constant_values=0)

            # Get lookback data for returns
            lookback_start = max(0, idx - self.config.lookback_days)
            timestamps = data.index.unique()
            lookback_timestamps = timestamps[lookback_start:idx+1]

            features = {}

            # Historical returns
            if len(lookback_timestamps) > 1:
                lookback_data = data.loc[lookback_timestamps]
                if 'returns' in lookback_data.columns:
                    returns = lookback_data.pivot_table(
                        index=lookback_data.index,
                        columns='symbol',
                        values='returns',
                        aggfunc='first'
                    ).values
                    features['returns'] = returns
                else:
                    features['returns'] = np.zeros((len(lookback_timestamps), self.config.n_assets))
            else:
                features['returns'] = np.zeros((1, self.config.n_assets))

            # Market caps
            if 'market_cap' in data.columns:
                if isinstance(ts_data, pd.Series):
                    features['market_caps'] = np.array([ts_data.get('market_cap', 1.0)])
                else:
                    features['market_caps'] = ts_data['market_cap'].values[:self.config.n_assets]

            # Factors
            factor_cols = [c for c in data.columns if c.startswith('factor_')]
            if factor_cols:
                if isinstance(ts_data, pd.Series):
                    features['factors'] = ts_data[factor_cols].values.reshape(1, -1)
                else:
                    features['factors'] = ts_data[factor_cols].values[:self.config.n_assets]

            # Sentiment
            if 'sentiment' in data.columns:
                if isinstance(ts_data, pd.Series):
                    features['sentiment'] = np.array([ts_data.get('sentiment', 0.0)])
                else:
                    features['sentiment'] = ts_data['sentiment'].values[:self.config.n_assets]

            return prices, features

        except Exception as e:
            logger.warning(f"Error preparing features at {timestamp}: {e}")
            return None, {}

    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """
        Apply position constraints to weights.

        Constraints:
        1. Clip individual weights to max_single_weight
        2. Scale to target leverage if needed

        Args:
            weights: Raw target weights

        Returns:
            Constrained weights
        """
        # Clip individual weights
        weights = np.clip(weights, -self.config.max_single_weight, self.config.max_single_weight)

        # Scale to target leverage if gross exposure exceeds limit
        gross_exposure = np.abs(weights).sum()
        if gross_exposure > self.config.target_leverage:
            scale = self.config.target_leverage / gross_exposure
            weights = weights * scale

        return weights

    def _compute_metrics(
        self,
        nav_series: pd.Series,
        returns_series: pd.Series,
        weights_df: pd.DataFrame,
        turnover_history: List[float],
        initial_nav: float,
    ) -> BacktestResult:
        """
        Compute performance metrics from backtest results.

        Uses PerformanceMetrics for consistent metric calculation across
        backtesting and RL training.

        Args:
            nav_series: NAV time series
            returns_series: Returns time series
            weights_df: Weight history
            turnover_history: Turnover per period
            initial_nav: Initial NAV

        Returns:
            BacktestResult with computed metrics
        """
        result = BacktestResult(strategy_name=self.name)

        result.nav_series = nav_series
        result.returns_series = returns_series
        result.weights_history = weights_df

        if len(returns_series) == 0:
            return result

        # Determine periods per year based on rebalance frequency
        if self.config.rebalance_freq == '5min':
            periods_per_year = 252 * 24 * 12  # 5-min bars per year
        elif self.config.rebalance_freq == 'hourly':
            periods_per_year = 252 * 24
        else:  # daily
            periods_per_year = 252

        # Use PerformanceMetrics for unified calculation
        metrics_calc = PerformanceMetrics(risk_free_rate=0.0, annualize=True)

        returns_np = returns_series.values
        nav_np = nav_series.values

        metrics = metrics_calc.calculate_all(
            returns=returns_np,
            equity_curve=nav_np,
            positions=None,  # Not needed for basic metrics
            periods_per_year=periods_per_year,
        )

        # Map to BacktestResult fields
        result.total_return = metrics['total_return']
        result.annual_return = metrics['annualized_return']
        result.volatility = metrics['volatility']
        result.sharpe_ratio = metrics['sharpe_ratio']
        result.sortino_ratio = metrics['sortino_ratio']
        result.downside_volatility = metrics['downside_volatility']
        result.max_drawdown = metrics['max_drawdown']
        result.calmar_ratio = metrics['calmar_ratio']
        result.cvar_95 = metrics['cvar_95']

        # Trading metrics (not in PerformanceMetrics)
        result.total_turnover = sum(turnover_history)
        result.avg_turnover = result.total_turnover / len(turnover_history) if turnover_history else 0
        result.total_cost = result.total_turnover * (self.config.transaction_cost + self.config.slippage)
        result.num_rebalances = len(turnover_history)

        return result
