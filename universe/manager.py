"""
Universe Manager - Dynamic Crypto Universe Selection.

Manages the daily rebalancing of the trading universe based on 24h quote volume.
Implements slot-based approach where RL agent sees "Rank 1 coin" not "BTC".

Key features:
- Daily rebalancing at 00:00 UTC
- Top 20 by 24h Quote Volume
- Minimum $10M threshold (empty slot if not met)
- Slot-based: Asset agnostic approach
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import logging

from config.settings import DATA_DIR, UniverseConfig


class UniverseManager:
    """
    Manages dynamic crypto universe selection.

    The universe is rebalanced daily at 00:00 UTC based on 24h quote volume.
    This enables:
    - Slot-based approach: Agent sees "Rank 1" not "BTC"
    - Historical consistency: Know which coins were top 20 on any date
    - Dynamic adaptation: New listings/delistings handled automatically

    Output:
    - Daily universe snapshots stored in universe_history.parquet
    - Symbol <-> Slot mapping for any historical date
    """

    def __init__(
        self,
        binance_data_dir: Optional[Path] = None,
        universe_dir: Optional[Path] = None,
        config: Optional[UniverseConfig] = None,
    ):
        """
        Initialize Universe Manager.

        Args:
            binance_data_dir: Directory with Binance Parquet files
            universe_dir: Directory to store universe history
            config: Universe configuration
        """
        self.binance_data_dir = binance_data_dir or DATA_DIR / 'binance'
        self.universe_dir = universe_dir or DATA_DIR / 'universe'
        self.config = config or UniverseConfig()

        # Ensure directories exist
        self.universe_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.top_n = self.config.top_n
        self.min_volume_usd = self.config.min_volume_usd
        self.rebalance_hour_utc = self.config.rebalance_hour_utc

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

        # Cache for universe history
        self._universe_cache: Optional[pd.DataFrame] = None

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

    @property
    def history_file(self) -> Path:
        """Path to universe history Parquet file."""
        return self.universe_dir / 'universe_history.parquet'

    @property
    def schema(self) -> pa.Schema:
        """PyArrow schema for universe history."""
        return pa.schema([
            ('date', pa.date32()),
            ('slot', pa.int32()),
            ('symbol', pa.string()),
            ('quote_volume_24h', pa.float64()),
        ])

    # ========== Volume Calculation ==========

    def calculate_24h_volume(
        self,
        target_date: date,
        interval: str = '5m',
    ) -> pd.DataFrame:
        """
        Calculate 24h quote volume for all symbols as of target_date 00:00 UTC.

        Args:
            target_date: Date to calculate volume for
            interval: Data interval used in Binance files

        Returns:
            DataFrame with columns: symbol, quote_volume_24h
        """
        # Calculate time range (previous 24 hours)
        end_dt = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(hours=24)

        self.logger.debug(f"Calculating 24h volume from {start_dt} to {end_dt}")

        # Find relevant Parquet files
        # Monthly partition format: binance_futures_5m_YYYYMM.parquet
        partition_keys = set()

        current = start_dt
        while current <= end_dt:
            key = current.strftime('%Y%m')
            partition_keys.add(key)
            # Move to next month
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                current = datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)

        # Read and combine data
        dfs = []
        for key in partition_keys:
            file_path = self.binance_data_dir / f"binance_futures_{interval}_{key}.parquet"
            if file_path.exists():
                # Read volume and close to calculate quote_volume (freqtrade doesn't provide quote_volume)
                df = pq.read_table(file_path, columns=['timestamp', 'symbol', 'volume', 'close']).to_pandas()
                # Calculate quote_volume = volume * close (USDT volume)
                df['quote_volume'] = df['volume'] * df['close']
                dfs.append(df[['timestamp', 'symbol', 'quote_volume']])

        if not dfs:
            self.logger.warning(f"No data found for {target_date}")
            return pd.DataFrame(columns=['symbol', 'quote_volume_24h'])

        df = pd.concat(dfs, ignore_index=True)

        # Filter to exact 24h window
        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)]

        if df.empty:
            return pd.DataFrame(columns=['symbol', 'quote_volume_24h'])

        # Aggregate by symbol
        volume_df = df.groupby('symbol')['quote_volume'].sum().reset_index()
        volume_df.columns = ['symbol', 'quote_volume_24h']

        # Sort by volume descending
        volume_df = volume_df.sort_values('quote_volume_24h', ascending=False)

        self.logger.debug(f"Calculated volume for {len(volume_df)} symbols")
        return volume_df

    # ========== Universe Selection ==========

    def select_universe(
        self,
        target_date: date,
        volume_df: Optional[pd.DataFrame] = None,
    ) -> Dict[int, Optional[str]]:
        """
        Select top N symbols by volume.

        Args:
            target_date: Date for universe selection
            volume_df: Pre-calculated volume DataFrame (optional)

        Returns:
            Dict mapping slot (1-20) to symbol or None
            Example: {1: 'BTCUSDT', 2: 'ETHUSDT', ..., 20: None}
        """
        if volume_df is None:
            volume_df = self.calculate_24h_volume(target_date)

        # Initialize all slots as empty
        universe: Dict[int, Optional[str]] = {i: None for i in range(1, self.top_n + 1)}

        if volume_df.empty:
            self.logger.warning(f"No volume data for {target_date}, returning empty universe")
            return universe

        # Filter by minimum volume
        qualified = volume_df[volume_df['quote_volume_24h'] >= self.min_volume_usd]

        # Assign to slots
        for rank, (_, row) in enumerate(qualified.head(self.top_n).iterrows(), start=1):
            universe[rank] = row['symbol']

        # Log stats
        filled_slots = sum(1 for s in universe.values() if s is not None)
        self.logger.info(
            f"Universe for {target_date}: {filled_slots}/{self.top_n} slots filled, "
            f"top coin: {universe.get(1, 'None')}"
        )

        return universe

    # ========== Universe History Management ==========

    def build_universe_history(
        self,
        start_date: date,
        end_date: date,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Build universe history for a date range.

        Args:
            start_date: Start date
            end_date: End date
            overwrite: Whether to overwrite existing history

        Returns:
            DataFrame with universe history
        """
        self.logger.info(f"Building universe history from {start_date} to {end_date}")

        records = []
        current = start_date

        while current <= end_date:
            # Calculate volume for this date
            volume_df = self.calculate_24h_volume(current)

            # Select universe
            universe = self.select_universe(current, volume_df)

            # Create records for each slot
            for slot, symbol in universe.items():
                volume = 0.0
                if symbol and not volume_df.empty:
                    match = volume_df[volume_df['symbol'] == symbol]
                    if not match.empty:
                        volume = match['quote_volume_24h'].iloc[0]

                records.append({
                    'date': current,
                    'slot': slot,
                    'symbol': symbol,
                    'quote_volume_24h': volume,
                })

            current += timedelta(days=1)

        history_df = pd.DataFrame(records)

        # Save to Parquet
        if overwrite or not self.history_file.exists():
            table = pa.Table.from_pandas(history_df, schema=self.schema, preserve_index=False)
            pq.write_table(table, self.history_file, compression='snappy')
            self.logger.info(f"Saved universe history to {self.history_file}")
        else:
            # Append new dates
            existing = self.load_universe_history()
            if existing is not None:
                history_df = pd.concat([existing, history_df], ignore_index=True)
                history_df = history_df.drop_duplicates(subset=['date', 'slot'], keep='last')
                history_df = history_df.sort_values(['date', 'slot'])

            table = pa.Table.from_pandas(history_df, schema=self.schema, preserve_index=False)
            pq.write_table(table, self.history_file, compression='snappy')

        # Clear cache
        self._universe_cache = None

        return history_df

    def load_universe_history(self) -> Optional[pd.DataFrame]:
        """
        Load universe history from Parquet file.

        Returns:
            DataFrame with universe history or None
        """
        if self._universe_cache is not None:
            return self._universe_cache

        if not self.history_file.exists():
            return None

        self._universe_cache = pq.read_table(self.history_file).to_pandas()
        return self._universe_cache

    def get_universe_for_date(self, target_date: date) -> Dict[int, Optional[str]]:
        """
        Get universe for a specific date from history.

        Args:
            target_date: Date to query

        Returns:
            Dict mapping slot to symbol
        """
        history = self.load_universe_history()

        if history is None or history.empty:
            self.logger.warning(f"No universe history found, calculating on-the-fly")
            return self.select_universe(target_date)

        # Filter to target date
        date_df = history[history['date'] == target_date]

        if date_df.empty:
            self.logger.warning(f"No universe data for {target_date}")
            return self.select_universe(target_date)

        universe = {}
        for _, row in date_df.iterrows():
            universe[row['slot']] = row['symbol']

        return universe

    def get_slot_for_symbol(
        self,
        symbol: str,
        target_date: date,
    ) -> Optional[int]:
        """
        Get the slot number for a symbol on a given date.

        Args:
            symbol: Symbol to look up
            target_date: Date to query

        Returns:
            Slot number (1-20) or None if not in universe
        """
        universe = self.get_universe_for_date(target_date)

        for slot, sym in universe.items():
            if sym == symbol:
                return slot

        return None

    def get_symbol_for_slot(
        self,
        slot: int,
        target_date: date,
    ) -> Optional[str]:
        """
        Get the symbol for a slot on a given date.

        Args:
            slot: Slot number (1-20)
            target_date: Date to query

        Returns:
            Symbol or None if slot is empty
        """
        universe = self.get_universe_for_date(target_date)
        return universe.get(slot)

    # ========== Analysis Methods ==========

    def get_universe_stats(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict:
        """
        Get statistics about universe composition over time.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            Dict with statistics
        """
        history = self.load_universe_history()

        if history is None or history.empty:
            return {'error': 'No history available'}

        if start_date:
            history = history[history['date'] >= start_date]
        if end_date:
            history = history[history['date'] <= end_date]

        # Calculate stats
        total_days = history['date'].nunique()
        unique_symbols = history['symbol'].dropna().unique()

        # Symbol frequency (how often in top 20)
        symbol_counts = history.groupby('symbol').size().sort_values(ascending=False)

        # Average fill rate
        fill_rates = history.groupby('date').apply(
            lambda x: x['symbol'].notna().sum() / self.top_n * 100
        )

        return {
            'total_days': total_days,
            'unique_symbols': len(unique_symbols),
            'avg_fill_rate_pct': round(fill_rates.mean(), 2),
            'min_fill_rate_pct': round(fill_rates.min(), 2),
            'most_frequent_symbols': symbol_counts.head(10).to_dict(),
        }

    def get_turnover_rate(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.Series:
        """
        Calculate daily turnover rate (% of universe changed).

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            Series with daily turnover rates
        """
        history = self.load_universe_history()

        if history is None or history.empty:
            return pd.Series()

        if start_date:
            history = history[history['date'] >= start_date]
        if end_date:
            history = history[history['date'] <= end_date]

        # Pivot to get symbols per date
        pivot = history.pivot(index='date', columns='slot', values='symbol')
        pivot = pivot.sort_index()

        # Calculate turnover
        turnovers = []
        dates = pivot.index.tolist()

        for i in range(1, len(dates)):
            prev_set = set(pivot.iloc[i-1].dropna())
            curr_set = set(pivot.iloc[i].dropna())

            if prev_set or curr_set:
                changed = len(prev_set.symmetric_difference(curr_set))
                total = len(prev_set.union(curr_set))
                turnover = changed / total * 100 if total > 0 else 0
            else:
                turnover = 0

            turnovers.append(turnover)

        return pd.Series(turnovers, index=dates[1:], name='turnover_pct')


# ========== Slot Mapper (Helper Class) ==========

class SlotMapper:
    """
    Maps between symbols and slots for a specific date.
    Utility class for feature engineering.
    """

    def __init__(self, universe_manager: UniverseManager):
        self.manager = universe_manager

    def to_slot_tensor(
        self,
        data_dict: Dict[str, pd.DataFrame],
        target_date: date,
        fill_value: float = 0.0,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert symbol-keyed data to slot-ordered tensor.

        Args:
            data_dict: Dict mapping symbol -> DataFrame with features
            target_date: Date for universe lookup
            fill_value: Value to use for empty slots

        Returns:
            Tuple of (tensor[slots, features], feature_names)
        """
        universe = self.manager.get_universe_for_date(target_date)

        # Get feature names from first available symbol
        feature_names = []
        for df in data_dict.values():
            if not df.empty:
                feature_names = df.columns.tolist()
                break

        if not feature_names:
            return np.array([]), []

        # Build tensor (slots x features)
        tensor = np.full((self.manager.top_n, len(feature_names)), fill_value)

        for slot in range(1, self.manager.top_n + 1):
            symbol = universe.get(slot)
            if symbol and symbol in data_dict:
                df = data_dict[symbol]
                if not df.empty:
                    tensor[slot - 1] = df.iloc[-1].values  # Use latest row

        return tensor, feature_names


# ========== Standalone Testing ==========

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Universe Manager')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--build', action='store_true', help='Build universe history')
    parser.add_argument('--stats', action='store_true', help='Show statistics')

    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d').date()
    end = datetime.strptime(args.end, '%Y-%m-%d').date()

    manager = UniverseManager()

    if args.build:
        history = manager.build_universe_history(start, end)
        print(f"Built universe history: {len(history)} records")

    if args.stats:
        stats = manager.get_universe_stats(start, end)
        print("Universe Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
