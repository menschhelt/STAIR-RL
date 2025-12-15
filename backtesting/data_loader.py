"""
Data Loader - Load Parquet data for backtesting.

Handles:
- Loading OHLCV from monthly Parquet files
- Applying universe filters
- Merging text features
- Time-aligned data retrieval
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

from config.settings import DATA_DIR, BacktestConfig


class ParquetDataLoader:
    """
    Loads and manages Parquet data for backtesting.

    Supports:
    - Binance Futures (monthly partitions)
    - Nostr text (weekly partitions)
    - GDELT news (weekly partitions)
    - Universe history
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        config: Optional[BacktestConfig] = None,
    ):
        """
        Initialize Data Loader.

        Args:
            data_dir: Base data directory
            config: Backtest configuration
        """
        self.data_dir = data_dir or DATA_DIR
        self.config = config or BacktestConfig()

        # Data directories
        self.binance_dir = self.data_dir / 'binance'
        self.nostr_dir = self.data_dir / 'nostr'
        self.gdelt_dir = self.data_dir / 'gdelt'
        self.universe_dir = self.data_dir / 'universe'
        self.features_dir = self.data_dir / 'features'

        # Cache
        self._ohlcv_cache: Dict[str, pd.DataFrame] = {}
        self._universe_cache: Optional[pd.DataFrame] = None

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

    # ========== OHLCV Loading ==========

    def load_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '5m',
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            start_date: Start datetime
            end_date: End datetime
            interval: Data interval

        Returns:
            DataFrame with OHLCV + funding rate + mark price
        """
        # FIX: Convert naive datetime to UTC-aware for comparison with DataFrame timestamps
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        # Get partition keys (YYYYMM)
        partition_keys = self._get_monthly_partitions(start_date, end_date)

        dfs = []
        for key in partition_keys:
            file_path = self.binance_dir / f"binance_futures_{interval}_{key}.parquet"

            if not file_path.exists():
                continue

            # Load and filter by symbol
            df = pq.read_table(
                file_path,
                filters=[('symbol', '==', symbol)]
            ).to_pandas()

            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)

        # Filter by time range
        result = result[
            (result['timestamp'] >= start_date) &
            (result['timestamp'] <= end_date)
        ]

        # Sort by timestamp
        result = result.sort_values('timestamp').reset_index(drop=True)

        return result

    def load_ohlcv_multi(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = '5m',
    ) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for multiple symbols.

        Args:
            symbols: List of trading pairs
            start_date: Start datetime
            end_date: End datetime
            interval: Data interval

        Returns:
            Dict mapping symbol to DataFrame
        """
        # FIX: Convert naive datetime to UTC-aware for comparison with DataFrame timestamps
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        result = {}

        # Get partition keys
        partition_keys = self._get_monthly_partitions(start_date, end_date)

        for key in partition_keys:
            file_path = self.binance_dir / f"binance_futures_{interval}_{key}.parquet"

            if not file_path.exists():
                continue

            # Load full partition
            df = pq.read_table(file_path).to_pandas()

            # Filter by symbols
            df = df[df['symbol'].isin(symbols)]

            # Filter by time
            df = df[
                (df['timestamp'] >= start_date) &
                (df['timestamp'] <= end_date)
            ]

            # Group by symbol
            for symbol, group in df.groupby('symbol'):
                if symbol not in result:
                    result[symbol] = []
                result[symbol].append(group)

        # Concatenate and sort
        for symbol in result:
            if isinstance(result[symbol], list):
                result[symbol] = pd.concat(result[symbol], ignore_index=True)
                result[symbol] = result[symbol].sort_values('timestamp').reset_index(drop=True)

        return result

    def _get_monthly_partitions(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[str]:
        """Get list of monthly partition keys (YYYYMM)."""
        partitions = []
        current = start_date.replace(day=1)

        while current <= end_date:
            partitions.append(current.strftime('%Y%m'))
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return partitions

    # ========== Universe Loading ==========

    def load_universe_history(self) -> pd.DataFrame:
        """Load universe history."""
        if self._universe_cache is not None:
            return self._universe_cache

        history_file = self.universe_dir / 'universe_history.parquet'

        if not history_file.exists():
            self.logger.warning("Universe history not found")
            return pd.DataFrame()

        self._universe_cache = pq.read_table(history_file).to_pandas()
        return self._universe_cache

    def get_universe_for_date(self, target_date: date) -> Dict[int, Optional[str]]:
        """Get universe (slot -> symbol) for a specific date."""
        history = self.load_universe_history()

        if history.empty:
            return {}

        date_df = history[history['date'] == target_date]

        universe = {}
        for _, row in date_df.iterrows():
            universe[row['slot']] = row['symbol']

        return universe

    def get_universe_symbols(
        self,
        start_date: date,
        end_date: date,
        top_n: int = 20,
    ) -> List[str]:
        """
        Get top N symbols in universe during period (filtered by slot).

        Args:
            start_date: Start date
            end_date: End date
            top_n: Number of top symbols to include (default: 20)

        Returns:
            List of unique symbols that were in top N during the period
        """
        history = self.load_universe_history()

        if history.empty:
            return []

        # Filter by date range and slot (already ranked by quote_volume)
        filtered = history[
            (history['date'] >= start_date) &
            (history['date'] <= end_date) &
            (history['slot'] <= top_n)
        ]

        symbols = filtered['symbol'].dropna().unique().tolist()
        return symbols

    def get_universe_timeline(
        self,
        start_date: date,
        end_date: date,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Get full universe timeline with slot->symbol mapping for each date.

        This enables dynamic universe rebalancing where:
        - Agent trades "slots" (Top 1, Top 2, ...) not fixed symbols
        - When universe rebalances, slots get new symbols
        - Portfolio weights carry over to new symbols in same slot

        Args:
            start_date: Start date
            end_date: End date
            top_n: Number of top slots to include (default: 20)

        Returns:
            DataFrame with columns: ['date', 'slot', 'symbol', 'quote_volume']
            - One row per (date, slot) combination
            - Sorted by (date, slot)

        Example:
            >>> df = loader.get_universe_timeline('2021-01-01', '2021-01-07', top_n=3)
            >>> df.head(6)
               date  slot    symbol  quote_volume
            0  2021-01-01     1  BTCUSDT   1e10
            1  2021-01-01     2  ETHUSDT   5e9
            2  2021-01-01     3  BNBUSDT   2e9
            3  2021-01-02     1  BTCUSDT   1.1e10
            4  2021-01-02     2  ETHUSDT   5.2e9
            5  2021-01-02     3  BNBUSDT   2.1e9
        """
        history = self.load_universe_history()

        if history.empty:
            return pd.DataFrame()

        # Filter by date range and top N slots
        filtered = history[
            (history['date'] >= start_date) &
            (history['date'] <= end_date) &
            (history['slot'] <= top_n)
        ].copy()

        # Sort by date and slot for consistent ordering
        filtered = filtered.sort_values(['date', 'slot']).reset_index(drop=True)

        return filtered

    def get_rebalance_dates(
        self,
        start_date: date,
        end_date: date,
        top_n: int = 20,
    ) -> List[Tuple[date, Dict[int, str], Dict[int, str]]]:
        """
        Get dates where universe composition changed.

        Returns list of (date, old_universe, new_universe) tuples for each rebalance event.

        Args:
            start_date: Start date
            end_date: End date
            top_n: Number of slots

        Returns:
            List of tuples: (rebalance_date, old_mapping, new_mapping)
            where mapping is {slot: symbol}
        """
        timeline = self.get_universe_timeline(start_date, end_date, top_n)

        if timeline.empty:
            return []

        rebalances = []
        prev_mapping = None

        for dt, group in timeline.groupby('date'):
            curr_mapping = {row['slot']: row['symbol'] for _, row in group.iterrows()}

            if prev_mapping is not None:
                # Check if any slot changed symbol
                changed = False
                for slot in range(1, top_n + 1):
                    old_sym = prev_mapping.get(slot)
                    new_sym = curr_mapping.get(slot)
                    if old_sym != new_sym:
                        changed = True
                        break

                if changed:
                    rebalances.append((dt, prev_mapping, curr_mapping))

            prev_mapping = curr_mapping

        self.logger.info(f"Found {len(rebalances)} rebalance events between {start_date} and {end_date}")
        return rebalances

    # ========== Text Data Loading ==========

    def load_nostr_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Load Nostr text data from weekly Parquet files."""
        partition_keys = self._get_weekly_partitions(start_date, end_date)

        dfs = []
        for key in partition_keys:
            file_path = self.nostr_dir / f"nostr_{key}.parquet"

            if not file_path.exists():
                continue

            df = pq.read_table(file_path).to_pandas()
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)

        # Filter by time range
        if 'created_at' in result.columns:
            result = result[
                (result['created_at'] >= start_date) &
                (result['created_at'] <= end_date)
            ]

        return result

    def load_gdelt_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Load GDELT news data from weekly Parquet files."""
        partition_keys = self._get_weekly_partitions(start_date, end_date)

        dfs = []
        for key in partition_keys:
            file_path = self.gdelt_dir / f"gdelt_{key}.parquet"

            if not file_path.exists():
                continue

            df = pq.read_table(file_path).to_pandas()
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)

        # Filter by time range
        if 'published_at' in result.columns:
            result = result[
                (result['published_at'] >= start_date) &
                (result['published_at'] <= end_date)
            ]

        return result

    def _get_weekly_partitions(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[str]:
        """Get list of weekly partition keys (YYYYWW)."""
        partitions = set()
        current = start_date

        while current <= end_date:
            iso_year, iso_week, _ = current.isocalendar()
            partitions.add(f"{iso_year}{iso_week:02d}")
            current += timedelta(days=1)

        return sorted(partitions)

    # ========== Feature Loading ==========

    def load_cached_features(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load pre-calculated alpha features for a symbol."""
        cache_dir = self.features_dir / 'alpha_cache'
        cache_file = cache_dir / f"{symbol}_features.parquet"

        if not cache_file.exists():
            return pd.DataFrame()

        df = pq.read_table(cache_file).to_pandas()

        if start_date and 'timestamp' in df.columns:
            df = df[df['timestamp'] >= start_date]

        if end_date and 'timestamp' in df.columns:
            df = df[df['timestamp'] <= end_date]

        return df

    def load_text_features(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Load aggregated text sentiment features."""
        cache_file = self.features_dir / 'text_features.parquet'

        if not cache_file.exists():
            return pd.DataFrame()

        df = pq.read_table(cache_file).to_pandas()

        # Filter by time range
        if 'timestamp' in df.columns:
            df = df[
                (df['timestamp'] >= start_date) &
                (df['timestamp'] <= end_date)
            ]

        return df

    # ========== Tensor Loading ==========

    def load_tensor(
        self,
        name: str,
    ) -> Tuple[np.ndarray, List[str], List[datetime]]:
        """
        Load pre-built tensor from disk.

        Args:
            name: Tensor name (e.g., 'train', 'val', 'test')

        Returns:
            Tuple of (tensor, feature_names, timestamps)
        """
        tensor_dir = self.features_dir / 'tensors'
        tensor_file = tensor_dir / f"{name}.npz"

        if not tensor_file.exists():
            raise FileNotFoundError(f"Tensor not found: {tensor_file}")

        data = np.load(tensor_file, allow_pickle=True)

        tensor = data['tensor']
        feature_names = data['feature_names'].tolist()
        timestamps = [datetime.fromisoformat(ts) for ts in data['timestamps']]

        return tensor, feature_names, timestamps


class BacktestDataProvider:
    """
    Provides time-aligned data for backtesting.

    Iterates through timestamps and provides:
    - Current universe
    - OHLCV for universe symbols
    - Features
    - Text sentiment
    """

    def __init__(
        self,
        data_loader: ParquetDataLoader,
        start_date: datetime,
        end_date: datetime,
        interval: str = '5m',
    ):
        """
        Initialize data provider.

        Args:
            data_loader: ParquetDataLoader instance
            start_date: Backtest start
            end_date: Backtest end
            interval: Data interval
        """
        self.data_loader = data_loader
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

        # Pre-load universe symbols
        self.universe_symbols = data_loader.get_universe_symbols(
            start_date.date(),
            end_date.date()
        )

        # Pre-load OHLCV data
        self.ohlcv_data = data_loader.load_ohlcv_multi(
            self.universe_symbols,
            start_date,
            end_date,
            interval
        )

        # Build timestamp index
        self.timestamps = self._build_timestamp_index()

        self.logger = logging.getLogger(self.__class__.__name__)

    def _build_timestamp_index(self) -> List[datetime]:
        """Build sorted list of unique timestamps."""
        all_ts = set()

        for symbol, df in self.ohlcv_data.items():
            if 'timestamp' in df.columns:
                all_ts.update(df['timestamp'].tolist())

        return sorted(all_ts)

    def get_data_at_timestamp(
        self,
        timestamp: datetime,
    ) -> Dict[str, pd.Series]:
        """
        Get OHLCV data for all universe symbols at timestamp.

        Args:
            timestamp: Target timestamp

        Returns:
            Dict mapping symbol to Series with OHLCV data
        """
        result = {}

        for symbol, df in self.ohlcv_data.items():
            if 'timestamp' not in df.columns:
                continue

            row = df[df['timestamp'] == timestamp]
            if not row.empty:
                result[symbol] = row.iloc[0]

        return result

    def get_universe_at_timestamp(
        self,
        timestamp: datetime,
    ) -> Dict[int, Optional[str]]:
        """Get universe at timestamp."""
        target_date = timestamp.date() if isinstance(timestamp, datetime) else timestamp
        return self.data_loader.get_universe_for_date(target_date)

    def iterate_timestamps(self):
        """Generator that yields (timestamp, data) tuples."""
        for ts in self.timestamps:
            data = self.get_data_at_timestamp(ts)
            universe = self.get_universe_at_timestamp(ts)
            yield ts, data, universe


# ========== Standalone Testing ==========

if __name__ == '__main__':
    loader = ParquetDataLoader()

    # Test loading universe
    history = loader.load_universe_history()
    print(f"Universe history: {len(history)} records")

    # Test getting symbols
    from datetime import date
    symbols = loader.get_universe_symbols(
        date(2024, 1, 1),
        date(2024, 1, 31)
    )
    print(f"Symbols in Jan 2024: {len(symbols)}")
