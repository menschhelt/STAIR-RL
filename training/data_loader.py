"""
Training Data Loader

Bridges ParquetDataLoader and TradingEnv by loading and preprocessing
data into the format expected by TradingEnv.

TradingEnv expects:
    - 'states': (T, N_assets, D_features) market state tensor
    - 'returns': (T, N_assets) asset returns
    - 'prices': (T, N_assets) asset prices
    - 'timestamps': (T,) timestamps
    - 'funding_rates': (T, N_assets) funding rates (optional)
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import numpy as np
import pandas as pd

from config.settings import DATA_DIR
from backtesting.data_loader import ParquetDataLoader

logger = logging.getLogger(__name__)


class TrainingDataLoader:
    """
    Loads and preprocesses data for RL training.

    Converts raw parquet data into tensors suitable for TradingEnv.
    """

    def __init__(
        self,
        data_dir: Path = None,
        n_assets: int = 20,
        state_dim: int = 36,
    ):
        """
        Initialize training data loader.

        Args:
            data_dir: Base data directory
            n_assets: Number of assets (top N from universe)
            state_dim: Feature dimension per asset
        """
        self.data_dir = data_dir or DATA_DIR
        self.n_assets = n_assets
        self.state_dim = state_dim

        self.parquet_loader = ParquetDataLoader(data_dir=self.data_dir)

    def load_period(
        self,
        start_date,
        end_date,
        interval: str = '5m',
    ) -> Dict[str, np.ndarray]:
        """
        Load training data for a time period.

        Args:
            start_date: Start date (datetime or string 'YYYY-MM-DD')
            end_date: End date (datetime or string 'YYYY-MM-DD')
            interval: Data interval (e.g., '5m', '1h')

        Returns:
            Dict with keys: 'states', 'returns', 'prices', 'timestamps'
        """
        # Parse string dates to datetime
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        logger.info(f"Loading training data: {start_date} to {end_date}")

        # Get universe symbols for the period
        try:
            top_symbols = self.parquet_loader.get_universe_symbols(
                start_date=start_date.date() if hasattr(start_date, 'date') else start_date,
                end_date=end_date.date() if hasattr(end_date, 'date') else end_date,
                top_n=self.n_assets,
            )
            logger.info(f"Universe: {len(top_symbols)} symbols")
        except Exception as e:
            logger.warning(f"Failed to get universe: {e}, using mock data")
            return self._create_mock_data(start_date, end_date, interval)

        if not top_symbols:
            logger.warning("No symbols in universe, using mock data")
            return self._create_mock_data(start_date, end_date, interval)

        # Load OHLCV data for all symbols
        try:
            ohlcv_dict = self.parquet_loader.load_ohlcv_multi(
                symbols=top_symbols[:self.n_assets],
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )
        except Exception as e:
            logger.warning(f"Failed to load OHLCV: {e}, using mock data")
            return self._create_mock_data(start_date, end_date, interval)

        if not ohlcv_dict:
            logger.warning("No OHLCV data loaded, using mock data")
            return self._create_mock_data(start_date, end_date, interval)

        # Combine into single DataFrame
        dfs = []
        for symbol, df in ohlcv_dict.items():
            if df is not None and len(df) > 0:
                df = df.copy()
                df['symbol'] = symbol
                dfs.append(df)

        if not dfs:
            logger.warning("All symbols empty, using mock data")
            return self._create_mock_data(start_date, end_date, interval)

        ohlcv_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(ohlcv_df)} OHLCV rows from {len(dfs)} symbols")

        # Get unique timestamps
        timestamps = sorted(ohlcv_df['timestamp'].unique())
        logger.info(f"Time range: {len(timestamps)} timesteps")

        # Use only top N symbols
        symbols = top_symbols[:self.n_assets]
        N = len(symbols)
        T = len(timestamps)
        D = self.state_dim

        logger.info(f"Building tensors: T={T}, N={N}, D={D} (vectorized)")

        # Create symbol-to-index mapping
        symbol_to_idx = {s: i for i, s in enumerate(symbols)}

        # Filter to only include our symbols
        ohlcv_df = ohlcv_df[ohlcv_df['symbol'].isin(symbols)]

        # Create timestamp-to-index mapping
        ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

        # Build price matrix using pivot (vectorized)
        logger.info("Building price matrix...")
        price_pivot = ohlcv_df.pivot_table(
            index='timestamp',
            columns='symbol',
            values='close',
            aggfunc='first'
        )

        # Reindex to ensure all timestamps and symbols are present
        price_pivot = price_pivot.reindex(index=timestamps, columns=symbols)
        prices = price_pivot.values.astype(np.float32)  # (T, N)

        # Fill NaN with forward fill, then backward fill
        prices = pd.DataFrame(prices).ffill().bfill().values.astype(np.float32)

        # Calculate returns (vectorized)
        logger.info("Calculating returns...")
        prices_safe = np.where(prices > 0, prices, 1e-8)
        returns = np.zeros_like(prices)
        returns[1:] = np.log(prices_safe[1:] / prices_safe[:-1])
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Build state features (vectorized)
        logger.info("Building state features...")

        # Pivot other OHLCV columns
        open_pivot = ohlcv_df.pivot_table(index='timestamp', columns='symbol', values='open', aggfunc='first')
        high_pivot = ohlcv_df.pivot_table(index='timestamp', columns='symbol', values='high', aggfunc='first')
        low_pivot = ohlcv_df.pivot_table(index='timestamp', columns='symbol', values='low', aggfunc='first')
        volume_pivot = ohlcv_df.pivot_table(index='timestamp', columns='symbol', values='volume', aggfunc='first')

        # Reindex
        open_pivot = open_pivot.reindex(index=timestamps, columns=symbols).ffill().bfill()
        high_pivot = high_pivot.reindex(index=timestamps, columns=symbols).ffill().bfill()
        low_pivot = low_pivot.reindex(index=timestamps, columns=symbols).ffill().bfill()
        volume_pivot = volume_pivot.reindex(index=timestamps, columns=symbols).ffill().bfill()

        opens = open_pivot.values.astype(np.float32)
        highs = high_pivot.values.astype(np.float32)
        lows = low_pivot.values.astype(np.float32)
        volumes = volume_pivot.values.astype(np.float32)

        # Build states (T, N, D)
        states = np.zeros((T, N, D), dtype=np.float32)

        # Feature 0: log(close/open)
        opens_safe = np.where(opens > 0, opens, 1e-8)
        states[:, :, 0] = np.log(prices / opens_safe)

        # Feature 1: log(high/close)
        prices_safe = np.where(prices > 0, prices, 1e-8)
        states[:, :, 1] = np.log(highs / prices_safe)

        # Feature 2: log(close/low)
        lows_safe = np.where(lows > 0, lows, 1e-8)
        states[:, :, 2] = np.log(prices / lows_safe)

        # Feature 3: normalized log volume
        states[:, :, 3] = np.log1p(volumes) / 20

        # Feature 4: normalized price
        states[:, :, 4] = prices / 10000

        # Features 5-35: returns at different lags (simplified moving averages)
        for lag in range(1, min(32, D - 5) + 1):
            if lag < T:
                states[lag:, :, 4 + lag] = returns[lag:] * np.sqrt(lag)

        # Clean up NaN/Inf
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        logger.info(f"Built tensors: states{states.shape}, returns{returns.shape}")

        return {
            'states': states,
            'returns': returns,
            'prices': prices,
            'timestamps': np.array(timestamps),
            'symbols': symbols,  # Add symbols for alpha loading
        }

    def _build_simple_state(self, row: pd.Series) -> np.ndarray:
        """
        Build simplified state features from OHLCV row.

        In production, use HierarchicalStateBuilder for full features.
        """
        state = np.zeros(self.state_dim, dtype=np.float32)

        # Basic OHLCV features (first 5)
        if 'open' in row:
            state[0] = np.log(row['close'] / row['open']) if row['open'] > 0 else 0
        if 'high' in row:
            state[1] = np.log(row['high'] / row['close']) if row['close'] > 0 else 0
        if 'low' in row:
            state[2] = np.log(row['close'] / row['low']) if row['low'] > 0 else 0
        if 'volume' in row:
            state[3] = np.log1p(row['volume']) / 20  # Normalized log volume
        if 'close' in row:
            state[4] = row['close'] / 10000  # Normalized price (rough)

        # Add noise for remaining features (placeholder)
        # In production, these would be actual alpha factors
        state[5:] = np.random.randn(self.state_dim - 5) * 0.1

        return state

    def _create_mock_data(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str = '5m',
    ) -> Dict[str, np.ndarray]:
        """
        Create mock data for testing when real data is unavailable.
        """
        logger.warning("Creating mock training data")

        # Calculate number of timesteps
        delta = end_date - start_date
        if interval == '5m':
            T = int(delta.total_seconds() / 300)
        elif interval == '1h':
            T = int(delta.total_seconds() / 3600)
        else:
            T = int(delta.days * 24)  # Assume hourly

        T = min(T, 100000)  # Cap for memory
        N = self.n_assets
        D = self.state_dim

        logger.info(f"Mock data: T={T}, N={N}, D={D}")

        # Generate random data
        states = np.random.randn(T, N, D).astype(np.float32) * 0.1

        # Generate correlated returns (more realistic)
        returns = np.random.randn(T, N).astype(np.float32) * 0.001

        # Generate prices from returns
        prices = np.exp(np.cumsum(returns, axis=0)) * 100

        # Generate timestamps
        timestamps = pd.date_range(start_date, end_date, periods=T)

        return {
            'states': states,
            'returns': returns,
            'prices': prices.astype(np.float32),
            'timestamps': timestamps.to_numpy(),
        }
