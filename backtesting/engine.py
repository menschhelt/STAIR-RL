"""
Backtest Engine - Simulates trading on historical data.

Features:
- Event-driven backtesting at configurable intervals
- Factor model integration (Loading Matrix, Crypto Factors)
- State generation for RL agent
- Portfolio tracking and PnL calculation
- Transaction cost modeling
- Funding rate simulation
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
import logging
from dataclasses import dataclass, field

from config.settings import DATA_DIR, BacktestConfig
from backtesting.data_loader import ParquetDataLoader, BacktestDataProvider
from backtesting.metrics import PerformanceMetrics
from backtesting.pnl_calculator import PnLCalculator
from backtesting.validator import DataValidator
from features.state_builder import StateBuilder
from factors.loading_matrix import LoadingMatrixCalculator
from factors.crypto_factors import CryptoFactorEngine


@dataclass
class Position:
    """Represents a single position."""
    symbol: str
    size: float  # Positive = long, Negative = short
    entry_price: float
    entry_time: datetime
    slot: int  # Universe slot (1-20)


@dataclass
class Portfolio:
    """Portfolio state."""
    cash: float = 1_000_000.0  # Initial capital
    positions: Dict[str, Position] = field(default_factory=dict)
    equity_history: List[float] = field(default_factory=list)
    returns_history: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        position_value = 0.0
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                position_value += pos.size * current_prices[symbol]
        return self.cash + position_value

    def get_position_weights(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Get position weights."""
        total = self.get_total_value(current_prices)
        if total == 0:
            return {}

        weights = {}
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                weights[symbol] = (pos.size * current_prices[symbol]) / total

        return weights


@dataclass
class Trade:
    """Represents an executed trade."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    price: float
    cost: float  # Transaction cost
    slot: int


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Supports:
    - Long/short positions with leverage
    - Transaction costs (taker/maker fees)
    - Funding rate payments (perpetual futures)
    - Universe rotation (Top 20 by volume)
    - Factor model state generation
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        data_loader: Optional[ParquetDataLoader] = None,
    ):
        """
        Initialize Backtest Engine.

        Args:
            config: Backtest configuration
            data_loader: Data loader instance
        """
        self.config = config or BacktestConfig()
        self.data_loader = data_loader or ParquetDataLoader()

        # Components
        self.state_builder = StateBuilder(n_assets=self.config.universe_size)
        self.metrics_calculator = PerformanceMetrics()

        # Note: BacktestEngine uses position-based tracking for accuracy.
        # PnLCalculator (weight-based) is used by TradingEnv for RL training speed.
        # Both approaches are valid for their respective use cases.
        self.pnl_calculator = PnLCalculator(
            transaction_cost_rate=self.config.taker_fee,
            slippage_rate=self.config.slippage,
        )

        self.validator = DataValidator()
        self.loading_matrix_calc = LoadingMatrixCalculator()
        self.factor_engine = CryptoFactorEngine()

        # State
        self.portfolio = Portfolio(cash=self.config.initial_capital)
        self.trades: List[Trade] = []
        self.current_timestamp: Optional[datetime] = None

        # Data caches
        self._ohlcv_cache: Dict[str, pd.DataFrame] = {}
        self._macro_data: Optional[pd.DataFrame] = None
        self._loading_matrix: Dict[str, Dict[str, float]] = {}
        self._factor_returns: Dict[str, float] = {}

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def run(
        self,
        start_date: datetime,
        end_date: datetime,
        strategy: Callable[[np.ndarray, Dict], Dict[str, float]],
        interval: str = '5m',
        warmup_days: int = 30,
    ) -> Dict:
        """
        Run backtest.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            strategy: Strategy function (state -> target_weights)
            interval: Data interval
            warmup_days: Days to skip for warmup

        Returns:
            Backtest results dictionary
        """
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")

        # Initialize
        self.portfolio = Portfolio(cash=self.config.initial_capital)
        self.trades = []

        # Warmup period
        warmup_end = start_date - timedelta(days=warmup_days)
        actual_start = start_date

        # Load data
        self._load_data(warmup_end, end_date, interval)

        # Get all timestamps
        timestamps = self._get_timestamps(actual_start, end_date, interval)
        self.logger.info(f"Processing {len(timestamps)} timestamps")

        # Main loop
        for i, ts in enumerate(timestamps):
            self.current_timestamp = ts

            # Get current universe
            universe = self.data_loader.get_universe_for_date(ts.date())

            # Get current prices
            current_prices = self._get_current_prices(ts, universe)

            # Update portfolio equity
            equity = self.portfolio.get_total_value(current_prices)
            self.portfolio.equity_history.append(equity)
            self.portfolio.timestamps.append(ts)

            # Calculate return
            if len(self.portfolio.equity_history) > 1:
                ret = (equity / self.portfolio.equity_history[-2]) - 1
            else:
                ret = 0.0
            self.portfolio.returns_history.append(ret)

            # Process funding rate payments (every 8 hours)
            if ts.hour in [0, 8, 16] and ts.minute == 0:
                self._process_funding_payments(ts, universe, current_prices)

            # Generate state for strategy
            state = self._build_state(ts, universe, current_prices)

            # Get portfolio info for strategy
            portfolio_info = {
                'cash': self.portfolio.cash,
                'equity': equity,
                'positions': self.portfolio.get_position_weights(current_prices),
                'timestamp': ts,
                'universe': universe,
            }

            # Get target weights from strategy
            target_weights = strategy(state, portfolio_info)

            # Execute trades to achieve target weights
            self._rebalance(ts, target_weights, universe, current_prices)

            # Log progress
            if (i + 1) % 1000 == 0:
                self.logger.info(
                    f"Processed {i + 1}/{len(timestamps)} | "
                    f"Equity: ${equity:,.0f} | "
                    f"Positions: {len(self.portfolio.positions)}"
                )

        # Calculate final metrics
        results = self._calculate_results()

        self.logger.info(
            f"Backtest complete | "
            f"Total Return: {results['total_return']:.2%} | "
            f"Sharpe: {results['sharpe_ratio']:.2f} | "
            f"Max DD: {results['max_drawdown']:.2%}"
        )

        return results

    def _load_data(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ):
        """Load all required data."""
        self.logger.info("Loading data...")

        # Get universe symbols
        symbols = self.data_loader.get_universe_symbols(
            start_date.date(), end_date.date()
        )

        # Load OHLCV
        self._ohlcv_cache = self.data_loader.load_ohlcv_multi(
            symbols, start_date, end_date, interval
        )
        self.logger.info(f"Loaded OHLCV for {len(self._ohlcv_cache)} symbols")

        # Load macro data (if available)
        try:
            macro_dir = self.data_loader.data_dir / 'macro'
            if macro_dir.exists():
                macro_files = list(macro_dir.glob('*.parquet'))
                if macro_files:
                    dfs = [pd.read_parquet(f) for f in macro_files]
                    self._macro_data = pd.concat(dfs, ignore_index=True)
                    self.logger.info(f"Loaded {len(self._macro_data)} macro records")
        except Exception as e:
            self.logger.warning(f"Could not load macro data: {e}")
            self._macro_data = pd.DataFrame()

    def _get_timestamps(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> List[datetime]:
        """Get all unique timestamps in date range."""
        all_ts = set()

        for symbol, df in self._ohlcv_cache.items():
            if 'timestamp' in df.columns:
                filtered = df[
                    (df['timestamp'] >= start_date) &
                    (df['timestamp'] <= end_date)
                ]
                all_ts.update(filtered['timestamp'].tolist())

        return sorted(all_ts)

    def _get_current_prices(
        self,
        timestamp: datetime,
        universe: Dict[int, Optional[str]],
    ) -> Dict[str, float]:
        """Get current prices for universe symbols."""
        prices = {}

        for slot, symbol in universe.items():
            if symbol and symbol in self._ohlcv_cache:
                df = self._ohlcv_cache[symbol]
                if 'timestamp' in df.columns:
                    row = df[df['timestamp'] == timestamp]
                    if not row.empty:
                        # Use mark price if available, else close
                        if 'mark_price' in row.columns and not pd.isna(row.iloc[0]['mark_price']):
                            prices[symbol] = row.iloc[0]['mark_price']
                        else:
                            prices[symbol] = row.iloc[0]['close']

        return prices

    def _build_state(
        self,
        timestamp: datetime,
        universe: Dict[int, Optional[str]],
        current_prices: Dict[str, float],
    ) -> np.ndarray:
        """Build state matrix for strategy."""
        # Update loading matrix (every 5 hours)
        if not self._loading_matrix or self._should_update_loading_matrix(timestamp):
            self._update_loading_matrix(timestamp, universe)

        # Update factor returns
        self._update_factor_returns(timestamp, universe, current_prices)

        # Get portfolio weights
        portfolio_weights = self.portfolio.get_position_weights(current_prices)

        # Build state
        state = self.state_builder.build_state(
            timestamp=pd.Timestamp(timestamp),
            universe=universe,
            ohlcv_data=self._ohlcv_cache,
            loading_matrix=self._loading_matrix,
            factor_returns=self._factor_returns,
            macro_data=self._macro_data if self._macro_data is not None else pd.DataFrame(),
            sentiment_data=None,  # Can add sentiment later
            portfolio_weights=portfolio_weights,
        )

        return state

    def _should_update_loading_matrix(self, timestamp: datetime) -> bool:
        """Check if loading matrix should be updated."""
        # Update at 00:00, 05:00, 10:00, 15:00, 20:00
        return timestamp.hour % 5 == 0 and timestamp.minute == 0

    def _update_loading_matrix(
        self,
        timestamp: datetime,
        universe: Dict[int, Optional[str]],
    ):
        """Update factor loading matrix."""
        # Get returns for each symbol
        asset_returns = {}
        for slot, symbol in universe.items():
            if symbol and symbol in self._ohlcv_cache:
                df = self._ohlcv_cache[symbol]
                if 'close' in df.columns and len(df) >= 30:
                    returns = df.set_index('timestamp')['close'].pct_change().dropna()
                    if len(returns) >= 30:
                        asset_returns[symbol] = returns

        if not asset_returns:
            return

        # Build factor returns DataFrame
        factor_df = pd.DataFrame(self._factor_returns, index=[pd.Timestamp(timestamp)])

        if not factor_df.empty and len(factor_df.columns) > 0:
            self._loading_matrix = self.loading_matrix_calc.calculate_loading_matrix(
                asset_returns,
                factor_df,
                pd.Timestamp(timestamp)
            )

    def _update_factor_returns(
        self,
        timestamp: datetime,
        universe: Dict[int, Optional[str]],
        current_prices: Dict[str, float],
    ):
        """Update factor returns."""
        symbols = [s for s in universe.values() if s]

        # Build price data dict
        price_data = {
            s: self._ohlcv_cache[s].set_index('timestamp')
            for s in symbols if s in self._ohlcv_cache
        }

        if len(price_data) < 3:
            return

        try:
            self._factor_returns = self.factor_engine.calculate_all_factors(
                universe_symbols=symbols,
                price_data=price_data,
                timestamp=pd.Timestamp(timestamp),
                include_optional=False,
            )
        except Exception as e:
            self.logger.debug(f"Could not calculate factors: {e}")

    def _process_funding_payments(
        self,
        timestamp: datetime,
        universe: Dict[int, Optional[str]],
        current_prices: Dict[str, float],
    ):
        """Process 8-hourly funding rate payments."""
        for symbol, pos in list(self.portfolio.positions.items()):
            if symbol not in self._ohlcv_cache:
                continue

            df = self._ohlcv_cache[symbol]
            if 'funding_rate' not in df.columns:
                continue

            # Get funding rate at this timestamp
            row = df[df['timestamp'] == timestamp]
            if row.empty or pd.isna(row.iloc[0]['funding_rate']):
                continue

            funding_rate = row.iloc[0]['funding_rate']
            price = current_prices.get(symbol, pos.entry_price)
            position_value = abs(pos.size) * price

            # Funding payment: long pays short when funding > 0
            if pos.size > 0:  # Long position
                payment = -position_value * funding_rate
            else:  # Short position
                payment = position_value * funding_rate

            self.portfolio.cash += payment

    def _rebalance(
        self,
        timestamp: datetime,
        target_weights: Dict[str, float],
        universe: Dict[int, Optional[str]],
        current_prices: Dict[str, float],
    ):
        """Rebalance portfolio to target weights."""
        if not target_weights:
            return

        equity = self.portfolio.get_total_value(current_prices)
        current_weights = self.portfolio.get_position_weights(current_prices)

        # Calculate trades needed
        for symbol, target_weight in target_weights.items():
            if symbol not in current_prices:
                continue

            current_weight = current_weights.get(symbol, 0.0)
            weight_diff = target_weight - current_weight

            # Skip small changes
            if abs(weight_diff) < 0.001:
                continue

            # Calculate trade size
            target_value = equity * target_weight
            price = current_prices[symbol]
            target_size = target_value / price

            current_pos = self.portfolio.positions.get(symbol)
            current_size = current_pos.size if current_pos else 0.0

            size_diff = target_size - current_size

            if abs(size_diff) * price < 100:  # Min trade size $100
                continue

            # Execute trade
            self._execute_trade(timestamp, symbol, size_diff, price, universe)

    def _execute_trade(
        self,
        timestamp: datetime,
        symbol: str,
        size: float,
        price: float,
        universe: Dict[int, Optional[str]],
    ):
        """Execute a trade."""
        # Calculate transaction cost
        notional = abs(size) * price
        cost = notional * self.config.taker_fee

        # Determine side
        side = 'buy' if size > 0 else 'sell'

        # Find slot for symbol
        slot = 0
        for s, sym in universe.items():
            if sym == symbol:
                slot = s
                break

        # Record trade
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            size=abs(size),
            price=price,
            cost=cost,
            slot=slot,
        )
        self.trades.append(trade)

        # Update cash
        if side == 'buy':
            self.portfolio.cash -= (notional + cost)
        else:
            self.portfolio.cash += (notional - cost)

        # Update position
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            new_size = pos.size + size

            if abs(new_size) < 1e-8:  # Close position
                del self.portfolio.positions[symbol]
            else:
                pos.size = new_size
        else:
            # Open new position
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                size=size,
                entry_price=price,
                entry_time=timestamp,
                slot=slot,
            )

    def _calculate_results(self) -> Dict:
        """Calculate backtest results."""
        returns = np.array(self.portfolio.returns_history)
        equity_curve = np.array(self.portfolio.equity_history)

        # Use metrics calculator
        metrics = self.metrics_calculator.calculate_all(
            returns=returns,
            equity_curve=equity_curve,
            periods_per_year=365 * 24 * 12,  # 5-minute intervals
        )

        # Add trade statistics
        metrics['n_trades'] = len(self.trades)
        metrics['total_costs'] = sum(t.cost for t in self.trades)
        metrics['avg_trade_size'] = (
            np.mean([t.size * t.price for t in self.trades])
            if self.trades else 0
        )

        # Add equity and timestamps for plotting
        metrics['equity_curve'] = equity_curve
        metrics['timestamps'] = self.portfolio.timestamps
        metrics['trades'] = self.trades

        return metrics

    def validate_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict:
        """Validate data before backtest."""
        self.logger.info("Validating data...")

        # Load universe
        universe_history = self.data_loader.load_universe_history()

        # Get all symbols
        symbols = self.data_loader.get_universe_symbols(
            start_date.date(), end_date.date()
        )

        # Load OHLCV
        ohlcv_dict = self.data_loader.load_ohlcv_multi(
            symbols, start_date, end_date, '5m'
        )

        # Run validation
        report = self.validator.validate_all(
            ohlcv_dict=ohlcv_dict,
            universe_history=universe_history,
            start_date=start_date,
            end_date=end_date,
        )

        return report


# ========== Example Strategy ==========

def buy_and_hold_strategy(state: np.ndarray, portfolio_info: Dict) -> Dict[str, float]:
    """Simple buy-and-hold strategy: equal weight top 10 by market cap."""
    universe = portfolio_info.get('universe', {})

    # Equal weight for slots 1-10 (largest by volume)
    weights = {}
    for slot in range(1, 11):
        symbol = universe.get(slot)
        if symbol:
            weights[symbol] = 0.1  # 10% each

    return weights


def momentum_strategy(state: np.ndarray, portfolio_info: Dict) -> Dict[str, float]:
    """Momentum strategy: long high momentum, short low momentum."""
    universe = portfolio_info.get('universe', {})

    # Get momentum from state (index 6 in local features)
    momentum_idx = 6
    momentums = state[:, momentum_idx]

    # Get top 5 and bottom 5
    valid_slots = [i for i in range(len(momentums)) if momentums[i] != 0]

    if len(valid_slots) < 10:
        return {}

    sorted_slots = sorted(valid_slots, key=lambda i: momentums[i], reverse=True)
    long_slots = sorted_slots[:5]
    short_slots = sorted_slots[-5:]

    weights = {}
    for slot in long_slots:
        symbol = universe.get(slot + 1)
        if symbol:
            weights[symbol] = 0.1  # 10% long

    for slot in short_slots:
        symbol = universe.get(slot + 1)
        if symbol:
            weights[symbol] = -0.1  # 10% short

    return weights


# ========== Standalone Testing ==========

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Backtest Engine')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--strategy', type=str, default='buy_hold', help='Strategy (buy_hold, momentum)')
    parser.add_argument('--validate', action='store_true', help='Validate data only')

    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    engine = BacktestEngine()

    if args.validate:
        report = engine.validate_data(start, end)
        engine.validator.print_report(report)
    else:
        # Select strategy
        if args.strategy == 'momentum':
            strategy = momentum_strategy
        else:
            strategy = buy_and_hold_strategy

        results = engine.run(start, end, strategy)

        # Print results
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annualized Return: {results['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Total Trades: {results['n_trades']}")
        print(f"Total Costs: ${results['total_costs']:,.2f}")
        print("=" * 60)
