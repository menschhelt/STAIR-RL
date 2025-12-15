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
        universe_timeline: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run vectorized backtest with dynamic universe support.

        The backtest now properly handles daily universe changes:
        - Each day uses that day's top N symbols by volume
        - Portfolio weights are computed on the current universe
        - Universe changes create turnover (position transfers)

        Args:
            data: DataFrame with columns:
                - timestamp: Datetime index
                - symbol: Asset symbol
                - close: Close price
                - volume: Trading volume (for cap-weighting)
            start_date: Backtest start date
            end_date: Backtest end date
            initial_nav: Initial portfolio value
            universe_timeline: Optional DataFrame with daily universe
                columns: ['date', 'slot', 'symbol', 'quote_volume']

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Running backtest for {self.name}: {start_date} to {end_date}")

        # Filter data by date range
        data = data[(data.index >= start_date) & (data.index <= end_date)]

        if len(data) == 0:
            logger.warning("No data in date range")
            return BacktestResult(strategy_name=self.name)

        # === PIVOT ALL DATA (all symbols) ===
        logger.info("Pivoting data to matrices...")

        # Get ALL unique symbols in data
        all_symbols = data['symbol'].unique().tolist()

        prices_matrix = data.pivot_table(
            index=data.index, columns='symbol', values='close', aggfunc='first'
        ).ffill().fillna(0)

        returns_matrix = prices_matrix.pct_change().fillna(0)

        # Volume for cap-weighting
        if 'volume' in data.columns:
            volume_matrix = data.pivot_table(
                index=data.index, columns='symbol', values='volume', aggfunc='first'
            ).ffill().fillna(0)
        else:
            volume_matrix = pd.DataFrame(1.0, index=prices_matrix.index, columns=prices_matrix.columns)

        all_timestamps = prices_matrix.index
        n_assets = self.config.n_assets

        # === BUILD DAILY UNIVERSE MAPPING ===
        # For each unique date, determine the top N symbols
        unique_dates = pd.Series(all_timestamps).dt.date.unique()

        # Build universe per date: date -> list of top N symbols
        daily_universe = {}

        if universe_timeline is not None and len(universe_timeline) > 0:
            # Use provided universe timeline
            for d in unique_dates:
                day_uni = universe_timeline[universe_timeline['date'] == d]
                day_uni = day_uni[day_uni['slot'] <= n_assets].sort_values('slot')
                symbols_today = day_uni['symbol'].tolist()
                # Filter to symbols that exist in our data
                symbols_today = [s for s in symbols_today if s in prices_matrix.columns]
                daily_universe[d] = symbols_today[:n_assets]
        else:
            # Compute universe from volume data (last bar of previous day)
            for d in unique_dates:
                # Get volume at end of this day (or use average)
                day_mask = pd.Series(all_timestamps).dt.date == d
                day_indices = np.where(day_mask.values)[0]
                if len(day_indices) > 0:
                    last_idx = day_indices[-1]
                    day_volumes = volume_matrix.iloc[last_idx]
                    top_symbols = day_volumes.nlargest(n_assets).index.tolist()
                    daily_universe[d] = top_symbols
                else:
                    daily_universe[d] = all_symbols[:n_assets]

        # === DETERMINE REBALANCING POINTS ===
        if self.config.rebalance_freq == 'daily':
            # Last bar of each day
            rebalance_mask = pd.Series(False, index=all_timestamps)
            for d in unique_dates:
                day_ts = all_timestamps[pd.Series(all_timestamps).dt.date.values == d]
                if len(day_ts) > 0:
                    rebalance_mask[day_ts[-1]] = True
        elif self.config.rebalance_freq == 'hourly':
            rebalance_mask = pd.Series([ts.minute == 0 for ts in all_timestamps], index=all_timestamps)
        else:
            rebalance_mask = pd.Series(True, index=all_timestamps)

        rebalance_indices = np.where(rebalance_mask.values)[0]
        logger.info(f"Total bars: {len(all_timestamps)}, Rebalance points: {len(rebalance_indices)}")

        # === COMPUTE PORTFOLIO RETURNS WITH DYNAMIC UNIVERSE ===
        nav = initial_nav
        nav_list = [nav]  # Start with initial NAV
        returns_list = [0.0]  # First bar has 0 return
        turnover_list = [0.0]

        # Track current holdings: symbol -> weight
        current_holdings = {}
        prev_reb_idx = 0

        for i, reb_idx in enumerate(rebalance_indices):
            ts = all_timestamps[reb_idx]
            current_date = ts.date()
            universe_today = daily_universe.get(current_date, all_symbols[:n_assets])

            # === COMPUTE RETURNS FROM PREVIOUS BAR TO CURRENT BAR ===
            # Process bars from (prev_reb_idx+1) to reb_idx (exclusive of reb_idx for now)
            start_bar = prev_reb_idx + 1 if i > 0 else 1  # Skip first bar (already added)

            for bar_idx in range(start_bar, reb_idx):
                if current_holdings:
                    bar_return = 0.0
                    for sym, weight in current_holdings.items():
                        if sym in returns_matrix.columns:
                            bar_return += weight * returns_matrix[sym].iloc[bar_idx]
                    nav = nav * (1 + bar_return)
                    nav_list.append(nav)
                    returns_list.append(bar_return)
                    turnover_list.append(0.0)
                else:
                    # No holdings yet - no return
                    nav_list.append(nav)
                    returns_list.append(0.0)
                    turnover_list.append(0.0)

            # === COMPUTE RETURN AT REBALANCE BAR (before rebalancing) ===
            if current_holdings:
                bar_return = 0.0
                for sym, weight in current_holdings.items():
                    if sym in returns_matrix.columns:
                        bar_return += weight * returns_matrix[sym].iloc[reb_idx]
                nav = nav * (1 + bar_return)
            else:
                bar_return = 0.0

            # === PREPARE FEATURES FOR WEIGHT COMPUTATION ===
            lookback_start = max(0, reb_idx - self.config.lookback_days)

            universe_returns = np.zeros((reb_idx - lookback_start + 1, n_assets))
            universe_volumes = np.zeros(n_assets)
            universe_prices = np.zeros(n_assets)

            for slot, sym in enumerate(universe_today[:n_assets]):
                if sym in returns_matrix.columns:
                    universe_returns[:, slot] = returns_matrix[sym].iloc[lookback_start:reb_idx+1].values
                    universe_volumes[slot] = volume_matrix[sym].iloc[reb_idx]
                    universe_prices[slot] = prices_matrix[sym].iloc[reb_idx]

            features = {
                'returns': universe_returns,
                'market_caps': universe_volumes,
                'volumes': universe_volumes,
            }

            # === COMPUTE NEW WEIGHTS ===
            current_weights_arr = np.zeros(n_assets)
            for slot, sym in enumerate(universe_today[:n_assets]):
                current_weights_arr[slot] = current_holdings.get(sym, 0.0)

            target_weights = self.compute_weights(
                timestamp=ts,
                prices=universe_prices,
                features=features,
                current_weights=current_weights_arr,
            )
            target_weights = self._apply_constraints(target_weights)

            # === CALCULATE TURNOVER ===
            new_holdings = {}
            for slot, sym in enumerate(universe_today[:n_assets]):
                new_holdings[sym] = target_weights[slot]

            turnover = 0.0
            all_syms = set(current_holdings.keys()) | set(new_holdings.keys())
            for sym in all_syms:
                old_w = current_holdings.get(sym, 0.0)
                new_w = new_holdings.get(sym, 0.0)
                turnover += abs(new_w - old_w)

            # Apply transaction cost to this bar's return
            cost_rate = turnover * (self.config.transaction_cost + self.config.slippage)
            net_return = bar_return - cost_rate
            nav_after_cost = nav * (1 - cost_rate) / (1 + bar_return) if (1 + bar_return) != 0 else nav
            # Simpler: just deduct cost from NAV
            nav = nav - nav * cost_rate

            # Record this rebalance bar
            nav_list.append(nav)
            returns_list.append(net_return)
            turnover_list.append(turnover)

            # Update holdings
            current_holdings = new_holdings
            prev_reb_idx = reb_idx

        # === HANDLE REMAINING BARS AFTER LAST REBALANCE ===
        for bar_idx in range(prev_reb_idx + 1, len(all_timestamps)):
            if current_holdings:
                bar_return = 0.0
                for sym, weight in current_holdings.items():
                    if sym in returns_matrix.columns:
                        bar_return += weight * returns_matrix[sym].iloc[bar_idx]
                nav = nav * (1 + bar_return)
                nav_list.append(nav)
                returns_list.append(bar_return)
                turnover_list.append(0.0)
            else:
                nav_list.append(nav)
                returns_list.append(0.0)
                turnover_list.append(0.0)

        # === TRIM TO MATCH TIMESTAMPS ===
        nav_list = nav_list[:len(all_timestamps)]
        returns_list = returns_list[:len(all_timestamps)]
        turnover_list = turnover_list[:len(all_timestamps)]

        # Pad if needed
        while len(nav_list) < len(all_timestamps):
            nav_list.append(nav_list[-1] if nav_list else initial_nav)
            returns_list.append(0.0)
            turnover_list.append(0.0)

        # Convert to pandas
        nav_df = pd.Series(nav_list[:len(all_timestamps)], index=all_timestamps)
        returns_df = pd.Series(returns_list[:len(all_timestamps)], index=all_timestamps)

        # Compute metrics
        result = self._compute_metrics(
            nav_series=nav_df,
            returns_series=returns_df,
            weights_df=pd.DataFrame(),  # Not tracking per-slot weights
            turnover_history=turnover_list,
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

        # Determine periods per year based on DATA frequency (always 5-min bars)
        # Crypto trades 24/7, so use 365 days
        # Returns are computed at 5-min granularity regardless of rebalance frequency
        periods_per_year = 365 * 24 * 12  # 5-min bars per year (crypto 24/7)

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
