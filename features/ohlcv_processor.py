#!/usr/bin/env python3
"""
OHLCV Sequence Builder for Transformer Price Encoder

Loads 5-minute candle sequences from Binance parquet files and prepares them
for the PriceTransformerEncoder.

File format: /home/work/data/stair-local/binance/binance_futures_5m_YYYYMM.parquet
"""
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class OHLCVSequenceBuilder:
    """
    Loads and processes 5-minute OHLCV sequences for price encoding.

    Purpose: Extract fixed-length sequences (default 288 = 1 day) of 5-minute candles
    for each symbol, normalized and ready for Transformer input.

    Args:
        data_dir: Root directory containing binance_futures_5m_YYYYMM.parquet files
        lookback: Number of 5-minute candles to load (default 288 = 24h * 12/h)
    """

    def __init__(
        self,
        data_dir: str = "/home/work/data/stair-local/binance",
        lookback: int = 288,
    ):
        self.data_dir = Path(data_dir)
        self.lookback = lookback

        # Cache for loaded monthly files
        self._cache = {}

        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def _get_parquet_path(self, year: int, month: int) -> Path:
        """Get path to monthly parquet file."""
        filename = f"binance_futures_5m_{year:04d}{month:02d}.parquet"
        return self.data_dir / filename

    def _load_monthly_data(self, year: int, month: int) -> Optional[pd.DataFrame]:
        """
        Load monthly parquet file with caching.

        Returns:
            DataFrame with columns: [date, open, high, low, close, volume, symbol, timestamp, quote_volume]
            or None if file doesn't exist
        """
        # Check cache
        cache_key = (year, month)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Load from disk
        path = self._get_parquet_path(year, month)
        if not path.exists():
            logger.warning(f"Monthly file not found: {path}")
            return None

        try:
            df = pd.read_parquet(path, engine='pyarrow')
            self._cache[cache_key] = df
            logger.debug(f"Loaded {len(df)} rows from {path.name}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None

    def _load_date_range(
        self,
        symbol: str,
        end_date: datetime,
        lookback_minutes: int,
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a symbol across date range.

        Args:
            symbol: Symbol name (e.g., 'BTCUSDT')
            end_date: End timestamp (inclusive)
            lookback_minutes: Minutes to look back from end_date

        Returns:
            DataFrame filtered by symbol and date range, sorted by timestamp
        """
        start_date = end_date - timedelta(minutes=lookback_minutes)

        # Determine which monthly files to load
        months_to_load = []
        current = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_month = end_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        while current <= end_month:
            months_to_load.append((current.year, current.month))
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        # Load and concatenate monthly data
        dfs = []
        for year, month in months_to_load:
            df = self._load_monthly_data(year, month)
            if df is not None:
                # Filter by symbol and date range
                mask = (
                    (df['symbol'] == symbol) &
                    (df['timestamp'] >= start_date) &
                    (df['timestamp'] <= end_date)
                )
                filtered = df[mask].copy()
                if len(filtered) > 0:
                    dfs.append(filtered)

        if not dfs:
            logger.warning(f"No data found for {symbol} in range {start_date} to {end_date}")
            return pd.DataFrame()

        # Concatenate and sort
        result = pd.concat(dfs, ignore_index=True)
        result = result.sort_values('timestamp').reset_index(drop=True)

        return result

    def load_5min_sequence(
        self,
        symbol: str,
        target_date: date,
        lookback: Optional[int] = None,
    ) -> np.ndarray:
        """
        Load the last N 5-minute candles before target date.

        Args:
            symbol: Symbol name (e.g., 'BTCUSDT', 'ETHUSDT')
            target_date: Target date (will load candles up to 23:59 of this date)
            lookback: Number of 5-minute candles (default: self.lookback = 288)

        Returns:
            Array of shape (lookback, 5) with normalized [open, high, low, close, volume]

            Normalization (Log Returns for Stationarity):
            - OHLC: Log returns log(p_t / p_{t-1}) * 100 for better gradient flow
            - Volume: Log change log(v_t / v_{t-1})
            - First row is padded with zeros (no previous candle)

            Returns zeros if insufficient data available.
        """
        if lookback is None:
            lookback = self.lookback

        # Convert target_date to end timestamp (23:55:00 UTC to include last 5-min candle of day)
        end_datetime = datetime.combine(target_date, datetime.min.time()).replace(
            hour=23, minute=55, second=0, microsecond=0, tzinfo=pd.Timestamp.now(tz='UTC').tz
        )

        # Load data with extra buffer (lookback + 10% margin)
        lookback_minutes = lookback * 5
        buffer_minutes = lookback_minutes + int(lookback_minutes * 0.1)

        df = self._load_date_range(symbol, end_datetime, buffer_minutes)

        # Check if we have enough data
        if len(df) < lookback:
            logger.warning(
                f"Insufficient data for {symbol} on {target_date}: "
                f"got {len(df)} candles, need {lookback}"
            )
            # Return zeros for missing data
            return np.zeros((lookback, 5), dtype=np.float32)

        # Take last `lookback` candles
        df = df.tail(lookback).copy()

        # Extract OHLCV
        ohlc = df[['open', 'high', 'low', 'close']].values  # (lookback, 4)
        volume = df['volume'].values  # (lookback,)

        # ======================
        # Log Returns for Stationarity (시계열 정상성 확보)
        # ======================
        # 기존 방식: (ohlc / first_close) - 1.0 → 비정상 (non-stationary)
        # 새 방식: log(p_t / p_{t-1}) → 정상 (stationary)

        # Prevent division by zero
        ohlc_safe = np.maximum(ohlc, 1e-8)
        volume_safe = np.maximum(volume, 1e-8)

        # Log returns: log(price_t / price_{t-1})
        # np.diff reduces length by 1, so we pad with zeros at the beginning
        log_ohlc = np.log(ohlc_safe)
        ohlc_log_returns = np.diff(log_ohlc, axis=0)  # (lookback-1, 4)
        ohlc_log_returns = np.vstack([np.zeros((1, 4)), ohlc_log_returns])  # (lookback, 4)

        # Volume log change: log(volume_t / volume_{t-1})
        log_volume = np.log(volume_safe)
        volume_log_change = np.diff(log_volume)  # (lookback-1,)
        volume_log_change = np.concatenate([[0.0], volume_log_change])  # (lookback,)

        # Scale returns for better gradient flow (optional but recommended)
        # Typical crypto 5-min log returns are ~0.001, scale by 100 to ~0.1
        scale_factor = 100.0
        ohlc_normalized = ohlc_log_returns * scale_factor
        volume_normalized = volume_log_change  # Volume change already in reasonable range

        # Combine: [open_logret, high_logret, low_logret, close_logret, volume_logchange]
        sequence = np.concatenate([
            ohlc_normalized,
            volume_normalized[:, np.newaxis]
        ], axis=1).astype(np.float32)

        return sequence  # (lookback, 5)

    def load_batch_sequences(
        self,
        symbols: List[str],
        target_date: date,
        lookback: Optional[int] = None,
    ) -> np.ndarray:
        """
        Load sequences for multiple symbols at once.

        Args:
            symbols: List of symbol names
            target_date: Target date
            lookback: Number of 5-minute candles (default: self.lookback)

        Returns:
            Array of shape (N, lookback, 5) where N = len(symbols)
        """
        if lookback is None:
            lookback = self.lookback

        sequences = []
        for symbol in symbols:
            seq = self.load_5min_sequence(symbol, target_date, lookback)
            sequences.append(seq)

        return np.stack(sequences, axis=0)  # (N, lookback, 5)

    def clear_cache(self):
        """Clear monthly data cache to free memory."""
        self._cache.clear()
        logger.debug("Cleared OHLCV data cache")


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Create builder
    builder = OHLCVSequenceBuilder(lookback=288)

    # Load single symbol
    print("\n=== Testing Single Symbol ===")
    seq = builder.load_5min_sequence('BTCUSDT', date(2024, 6, 15))
    print(f"Sequence shape: {seq.shape}")  # Should be (288, 5)
    print(f"First 3 candles:\n{seq[:3]}")
    print(f"Last 3 candles:\n{seq[-3:]}")

    # Load batch
    print("\n=== Testing Batch Load ===")
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    batch = builder.load_batch_sequences(symbols, date(2024, 6, 15))
    print(f"Batch shape: {batch.shape}")  # Should be (3, 288, 5)

    print("\n✅ OHLCVSequenceBuilder test complete")
