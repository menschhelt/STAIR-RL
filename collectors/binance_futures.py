"""
Binance Futures Local Collector - OHLCV + Funding Rate + Mark Price + Open Interest.

Adapts the existing BinanceOHLCVCollector for local Parquet storage.
Collects ALL futures symbols for dynamic universe selection.

File naming: binance_futures_YYYYMM.parquet (monthly partitioning)
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import pyarrow as pa
import numpy as np

from collectors.base_local_collector import BaseLocalCollector, SimpleRateLimiter
from config.settings import DATA_DIR, BinanceConfig


class BinanceFuturesLocalCollector(BaseLocalCollector):
    """
    Collects Binance Futures data to local Parquet files.

    Features:
    - OHLCV candle data (5m default)
    - Funding rate (every 8 hours)
    - Mark price and Index price
    - Open interest
    - All USDT-M perpetual symbols

    Data Schema:
    - timestamp: datetime64[ns, UTC]
    - symbol: string
    - open, high, low, close: float64
    - volume, quote_volume: float64
    - trades: int64
    - taker_buy_volume, taker_buy_quote_volume: float64
    - funding_rate: float64 (nullable)
    - mark_price: float64 (nullable)
    - index_price: float64 (nullable)
    - open_interest: float64 (nullable)
    """

    # API endpoints
    FUTURES_BASE_URL = "https://fapi.binance.com"

    # Rate limits (conservative)
    WEIGHT_LIMIT = 2400  # per minute
    MAX_LIMIT = 1500  # max candles per request

    # Interval mappings
    INTERVALS = {
        '1m': timedelta(minutes=1),
        '3m': timedelta(minutes=3),
        '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15),
        '30m': timedelta(minutes=30),
        '1h': timedelta(hours=1),
        '2h': timedelta(hours=2),
        '4h': timedelta(hours=4),
        '6h': timedelta(hours=6),
        '8h': timedelta(hours=8),
        '12h': timedelta(hours=12),
        '1d': timedelta(days=1),
    }

    def __init__(
        self,
        interval: str = '5m',
        config: Optional[BinanceConfig] = None,
    ):
        """
        Initialize Binance Futures collector.

        Args:
            interval: Candle interval (default: 5m)
            config: BinanceConfig instance
        """
        super().__init__(
            data_dir=DATA_DIR / 'binance',
            partition_strategy='monthly',
        )

        self.interval = interval
        self.config = config or BinanceConfig()

        # Validate interval
        if interval not in self.INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {list(self.INTERVALS.keys())}")

        self.interval_td = self.INTERVALS[interval]

        # Rate limiter - klines API weight=5, use 80% of limit
        # 2400 * 0.8 / 5 = 384 requests/min
        self.rate_limiter = SimpleRateLimiter(
            max_requests=int(self.WEIGHT_LIMIT * 0.8 / 5),  # 384 req/min (80% utilization)
            window_seconds=60
        )

        self.logger.info(
            f"Initialized BinanceFuturesLocalCollector with interval={interval}"
        )

    @property
    def collector_name(self) -> str:
        return f"binance_futures_{self.interval}"

    @property
    def schema(self) -> pa.Schema:
        return pa.schema([
            ('timestamp', pa.timestamp('ns', tz='UTC')),
            ('symbol', pa.string()),
            ('open', pa.float64()),
            ('high', pa.float64()),
            ('low', pa.float64()),
            ('close', pa.float64()),
            ('volume', pa.float64()),
            ('quote_volume', pa.float64()),
            ('trades', pa.int64()),
            ('taker_buy_volume', pa.float64()),
            ('taker_buy_quote_volume', pa.float64()),
            ('funding_rate', pa.float64()),
            ('mark_price', pa.float64()),
            ('index_price', pa.float64()),
            ('open_interest', pa.float64()),
        ])

    def _get_primary_key_columns(self) -> List[str]:
        return ['timestamp', 'symbol']

    # ========== API Methods ==========

    async def fetch_all_futures_symbols(self) -> List[str]:
        """
        Fetch all active USDT-M perpetual symbols.

        Returns:
            List of symbol strings (e.g., ['BTCUSDT', 'ETHUSDT', ...])
        """
        session = await self.get_session()
        url = f"{self.FUTURES_BASE_URL}/fapi/v1/exchangeInfo"

        await self.rate_limiter.acquire()

        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch exchange info: {response.status}")

            data = await response.json()

        # Filter for active USDT-M perpetuals
        symbols = []
        for sym in data.get('symbols', []):
            if (
                sym.get('contractType') == 'PERPETUAL' and
                sym.get('quoteAsset') == 'USDT' and
                sym.get('status') == 'TRADING'
            ):
                symbols.append(sym['symbol'])

        self.logger.info(f"Found {len(symbols)} active USDT-M perpetual symbols")
        return sorted(symbols)

    async def fetch_klines(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> List[List[Any]]:
        """
        Fetch OHLCV klines from Binance Futures API.

        Args:
            symbol: Trading pair
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds

        Returns:
            List of kline data
        """
        session = await self.get_session()
        url = f"{self.FUTURES_BASE_URL}/fapi/v1/klines"

        params = {
            'symbol': symbol,
            'interval': self.interval,
            'startTime': start_time_ms,
            'endTime': end_time_ms,
            'limit': self.MAX_LIMIT,
        }

        await self.rate_limiter.acquire()

        async with session.get(url, params=params) as response:
            if response.status != 200:
                text = await response.text()
                self.logger.error(f"Klines error for {symbol}: {response.status} - {text}")
                return []

            return await response.json()

    async def fetch_funding_rates(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> Dict[int, float]:
        """
        Fetch historical funding rates.

        Args:
            symbol: Trading pair
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds

        Returns:
            Dict mapping funding_time_ms -> funding_rate
        """
        session = await self.get_session()
        url = f"{self.FUTURES_BASE_URL}/fapi/v1/fundingRate"

        funding_rates = {}
        current_start = start_time_ms

        while current_start < end_time_ms:
            params = {
                'symbol': symbol,
                'startTime': current_start,
                'endTime': end_time_ms,
                'limit': 1000,
            }

            await self.rate_limiter.acquire()

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    break

                data = await response.json()
                if not data:
                    break

                for item in data:
                    funding_time = item['fundingTime']
                    funding_rate = float(item['fundingRate'])
                    funding_rates[funding_time] = funding_rate

                # Move to next batch
                current_start = data[-1]['fundingTime'] + 1

        return funding_rates

    async def fetch_premium_index(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Fetch current premium index (mark price, index price, funding rate).

        Args:
            symbol: Trading pair

        Returns:
            Dict with mark_price, index_price, funding_rate or None
        """
        session = await self.get_session()
        url = f"{self.FUTURES_BASE_URL}/fapi/v1/premiumIndex"

        params = {'symbol': symbol}

        await self.rate_limiter.acquire()

        async with session.get(url, params=params) as response:
            if response.status != 200:
                return None

            data = await response.json()

            return {
                'mark_price': float(data.get('markPrice', 0)),
                'index_price': float(data.get('indexPrice', 0)),
                'funding_rate': float(data.get('lastFundingRate', 0)),
            }

    async def fetch_open_interest(self, symbol: str) -> Optional[float]:
        """
        Fetch current open interest.

        Args:
            symbol: Trading pair

        Returns:
            Open interest value or None
        """
        session = await self.get_session()
        url = f"{self.FUTURES_BASE_URL}/fapi/v1/openInterest"

        params = {'symbol': symbol}

        await self.rate_limiter.acquire()

        async with session.get(url, params=params) as response:
            if response.status != 200:
                return None

            data = await response.json()
            return float(data.get('openInterest', 0))

    async def fetch_mark_price_klines(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> Dict[int, float]:
        """
        Fetch historical mark price klines.

        Args:
            symbol: Trading pair
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds

        Returns:
            Dict mapping timestamp_ms -> mark_price (close)
        """
        session = await self.get_session()
        url = f"{self.FUTURES_BASE_URL}/fapi/v1/markPriceKlines"

        mark_prices = {}
        current_start = start_time_ms

        while current_start < end_time_ms:
            params = {
                'symbol': symbol,
                'interval': self.interval,
                'startTime': current_start,
                'endTime': end_time_ms,
                'limit': self.MAX_LIMIT,
            }

            await self.rate_limiter.acquire()

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.warning(f"Mark price klines error for {symbol}: {response.status}")
                    break

                data = await response.json()
                if not data:
                    break

                for item in data:
                    # [Open time, Open, High, Low, Close, ...]
                    timestamp_ms = item[0]
                    mark_close = float(item[4])  # Close price
                    mark_prices[timestamp_ms] = mark_close

                # Move to next batch
                current_start = data[-1][0] + 1

                if len(data) < self.MAX_LIMIT:
                    break

        return mark_prices

    # ========== Data Collection ==========

    @staticmethod
    def _is_funding_time(ts: pd.Timestamp) -> bool:
        """Check if timestamp is a funding time (00:00, 08:00, 16:00 UTC)."""
        return ts.hour in (0, 8, 16) and ts.minute == 0

    async def collect_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Collect all data for a single symbol.

        Args:
            symbol: Trading pair
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV + funding + mark/index price
        """
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        all_klines = []
        current_start = start_ms

        # Fetch klines in batches
        while current_start < end_ms:
            klines = await self.fetch_klines(symbol, current_start, end_ms)

            if not klines:
                break

            all_klines.extend(klines)

            # Move to next batch
            last_close_time = klines[-1][6]  # Close time is index 6
            current_start = last_close_time + 1

            # Prevent infinite loop
            if len(klines) < self.MAX_LIMIT:
                break

        if not all_klines:
            return pd.DataFrame()

        # Parse klines into records
        records = []
        for k in all_klines:
            record = {
                'timestamp': pd.Timestamp(k[0], unit='ms', tz='UTC'),
                'symbol': symbol,
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'quote_volume': float(k[7]),
                'trades': int(k[8]),
                'taker_buy_volume': float(k[9]),
                'taker_buy_quote_volume': float(k[10]),
                'funding_rate': np.nan,  # Will be filled only at 8h intervals
                'mark_price': np.nan,    # Will be filled from mark price klines
                'index_price': np.nan,   # Not available historically
                'open_interest': np.nan, # Not available historically
            }
            records.append(record)

        df = pd.DataFrame(records)

        # ========== Fetch and merge mark prices ==========
        mark_prices = await self.fetch_mark_price_klines(symbol, start_ms, end_ms)

        if mark_prices:
            # Create lookup dict: timestamp_ms -> mark_price
            df['timestamp_ms'] = df['timestamp'].astype('int64') // 10**6
            df['mark_price'] = df['timestamp_ms'].map(mark_prices)
            df = df.drop(columns=['timestamp_ms'])

        # ========== Fetch and assign funding rates (only at 8h intervals) ==========
        funding_rates = await self.fetch_funding_rates(symbol, start_ms, end_ms)

        if funding_rates:
            # Create lookup: funding_time_ms -> rate
            # Funding is applied at exact times: 00:00, 08:00, 16:00 UTC
            for ts_ms, rate in funding_rates.items():
                # Find matching candle (within 1 minute tolerance)
                mask = (df['timestamp'].astype('int64') // 10**6 >= ts_ms - 60000) & \
                       (df['timestamp'].astype('int64') // 10**6 <= ts_ms + 60000)
                df.loc[mask, 'funding_rate'] = rate

        self.logger.info(f"Collected {len(df)} candles for {symbol}")
        return df

    async def collect(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: Optional[List[str]] = None,
        max_concurrent: int = 3,
    ) -> int:
        """
        Collect data for all symbols in the date range.

        Improved version: saves incrementally per symbol to avoid data loss.

        Args:
            start_date: Start of collection period
            end_date: End of collection period
            symbols: List of symbols to collect (None = all active)
            max_concurrent: Maximum concurrent symbol collections

        Returns:
            Total number of records collected
        """
        # Get symbols
        if symbols is None:
            symbols = await self.fetch_all_futures_symbols()

        total_symbols = len(symbols)
        self.logger.info(
            f"Starting collection for {total_symbols} symbols "
            f"from {start_date} to {end_date}"
        )

        total_records = 0
        completed_symbols = 0
        failed_symbols = []

        # Process symbols sequentially for better progress tracking
        # and immediate data persistence
        for idx, symbol in enumerate(symbols, 1):
            try:
                self.logger.info(f"[{idx}/{total_symbols}] Collecting {symbol}...")

                df = await self.collect_symbol(symbol, start_date, end_date)

                if df.empty:
                    self.logger.warning(f"[{idx}/{total_symbols}] {symbol}: No data (may not exist in this period)")
                    continue

                records_count = len(df)
                total_records += records_count

                # Group by partition and save immediately
                df['partition_key'] = df['timestamp'].apply(self.get_partition_key)

                partitions_written = 0
                for key, group in df.groupby('partition_key'):
                    self.write_parquet(group.drop(columns=['partition_key']), key, mode='append')
                    partitions_written += 1

                completed_symbols += 1
                self.logger.info(
                    f"[{idx}/{total_symbols}] {symbol}: {records_count:,} records saved to {partitions_written} partitions. "
                    f"Total: {total_records:,} records"
                )

                # Save checkpoint after each successful symbol
                self.save_checkpoint(end_date)

            except Exception as e:
                self.logger.error(f"[{idx}/{total_symbols}] Error collecting {symbol}: {e}")
                failed_symbols.append(symbol)
                continue

        self.logger.info(
            f"Collection complete: {total_records:,} records from {completed_symbols}/{total_symbols} symbols. "
            f"Failed: {len(failed_symbols)}"
        )

        if failed_symbols:
            self.logger.warning(f"Failed symbols: {failed_symbols[:20]}{'...' if len(failed_symbols) > 20 else ''}")

        return total_records

    async def collect_incremental(
        self,
        symbols: Optional[List[str]] = None,
        lookback_hours: int = 24,
    ) -> int:
        """
        Incremental collection from last checkpoint.

        Args:
            symbols: List of symbols to collect
            lookback_hours: Hours to look back if no checkpoint

        Returns:
            Number of records collected
        """
        checkpoint = self.load_checkpoint()

        if checkpoint:
            # Small overlap to ensure no gaps
            start_date = checkpoint - timedelta(hours=1)
        else:
            start_date = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

        end_date = datetime.now(timezone.utc)

        return await self.collect(start_date, end_date, symbols)


# ========== Standalone Execution ==========

async def main():
    """Run historical data collection."""
    import argparse

    parser = argparse.ArgumentParser(description='Binance Futures Data Collector')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='5m', help='Candle interval')
    parser.add_argument('--symbols', type=str, nargs='*', help='Specific symbols (default: all)')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    async with BinanceFuturesLocalCollector(interval=args.interval) as collector:
        total = await collector.collect(
            start_date=start_date,
            end_date=end_date,
            symbols=args.symbols,
        )
        print(f"Collected {total} records")

        stats = collector.get_data_stats()
        print(f"Stats: {stats}")


if __name__ == '__main__':
    asyncio.run(main())
