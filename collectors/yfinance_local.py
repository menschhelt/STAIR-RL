"""
YFinance Local Collector - Collect market indices and macro data from Yahoo Finance.

Collects:
- S&P 500 (^GSPC) - Equity index
- VIX (^VIX) - Volatility index
- DXY (DX-Y.NYB) - Dollar index
- Treasury Yields (^TNX, ^FVX) - Interest rates
- Gold (GC=F) and Oil (CL=F) - Commodities

Output: data/macro/yfinance_YYYYMM.parquet
"""

import pandas as pd
import numpy as np
import pyarrow as pa
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

from collectors.base_local_collector import BaseLocalCollector
from config.settings import DATA_DIR


class YFinanceLocalCollector(BaseLocalCollector):
    """
    Collects market indices and macro indicators from Yahoo Finance.

    Data is stored in monthly Parquet partitions.
    """

    # Default tickers to collect
    DEFAULT_TICKERS = {
        '^GSPC': {
            'name': 'S&P 500',
            'category': 'equity_index',
            'unit': 'points'
        },
        '^VIX': {
            'name': 'VIX Volatility Index',
            'category': 'volatility',
            'unit': 'index'
        },
        'DX-Y.NYB': {
            'name': 'US Dollar Index',
            'category': 'currency',
            'unit': 'index'
        },
        '^TNX': {
            'name': '10-Year Treasury Yield',
            'category': 'interest_rate',
            'unit': '%'
        },
        '^FVX': {
            'name': '5-Year Treasury Yield',
            'category': 'interest_rate',
            'unit': '%'
        },
        'GC=F': {
            'name': 'Gold Futures',
            'category': 'commodity',
            'unit': 'USD/oz'
        },
        'CL=F': {
            'name': 'Crude Oil Futures',
            'category': 'commodity',
            'unit': 'USD/barrel'
        }
    }

    # PyArrow schema for Parquet files
    SCHEMA = pa.schema([
        ('timestamp', pa.timestamp('us', tz='UTC')),
        ('indicator_name', pa.string()),
        ('indicator_category', pa.string()),
        ('value', pa.float64()),
        ('open', pa.float64()),
        ('high', pa.float64()),
        ('low', pa.float64()),
        ('close', pa.float64()),
        ('volume', pa.float64()),
        ('source', pa.string()),
        ('unit', pa.string()),
    ])

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        tickers: Optional[List[str]] = None,
    ):
        """
        Initialize YFinance Local Collector.

        Args:
            data_dir: Base data directory
            tickers: List of tickers to collect (default: all 7 tickers)
        """
        super().__init__(
            data_dir=data_dir or DATA_DIR / 'macro',
            partition_strategy='monthly',
        )

        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance is not installed. Install with: pip install yfinance"
            )

        self.tickers = tickers or list(self.DEFAULT_TICKERS.keys())
        self.logger.info(f"YFinance collector initialized with {len(self.tickers)} tickers")

    @property
    def collector_name(self) -> str:
        """Return collector name."""
        return "yfinance"

    @property
    def schema(self) -> pa.Schema:
        """Return PyArrow schema."""
        return self.SCHEMA

    async def collect(self, start_date: datetime, end_date: datetime) -> int:
        """Collect data for date range (required by base class)."""
        return await self.collect_and_save(start_date, end_date)

    async def collect_historical(
        self,
        start_date: datetime,
        end_date: datetime,
        tickers: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect historical data from Yahoo Finance.

        Args:
            start_date: Start date
            end_date: End date
            tickers: List of tickers (default: self.tickers)

        Returns:
            Dict mapping ticker to DataFrame
        """
        tickers = tickers or self.tickers
        results = {}

        self.logger.info(
            f"Collecting YFinance data from {start_date.date()} to {end_date.date()} "
            f"for {len(tickers)} tickers"
        )

        for ticker_symbol in tickers:
            try:
                df = self._fetch_ticker_data(
                    ticker_symbol,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )

                if not df.empty:
                    # Add metadata
                    ticker_info = self.DEFAULT_TICKERS.get(ticker_symbol, {})
                    df['indicator_name'] = ticker_symbol
                    df['indicator_category'] = ticker_info.get('category', 'unknown')
                    df['source'] = 'yfinance'
                    df['unit'] = ticker_info.get('unit', '')

                    results[ticker_symbol] = df
                    self.logger.info(f"Collected {len(df)} records for {ticker_symbol}")
                else:
                    self.logger.warning(f"No data for {ticker_symbol}")

            except Exception as e:
                self.logger.error(f"Error collecting {ticker_symbol}: {e}")

        return results

    def _fetch_ticker_data(
        self,
        ticker_symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.

        Args:
            ticker_symbol: Yahoo Finance ticker
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)

        Returns:
            DataFrame with timestamp and value columns
        """
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(start=start_date, end=end_date, interval='1d')

        if data.empty:
            return pd.DataFrame()

        # Reset index to get Date as column
        data = data.reset_index()

        # Standardize column names
        records = []
        for _, row in data.iterrows():
            timestamp = row['Date']
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)

            # Use Close price as the main value
            records.append({
                'timestamp': timestamp,
                'value': float(row['Close']),
                'open': float(row['Open']) if 'Open' in row else None,
                'high': float(row['High']) if 'High' in row else None,
                'low': float(row['Low']) if 'Low' in row else None,
                'close': float(row['Close']) if 'Close' in row else None,
                'volume': float(row['Volume']) if 'Volume' in row else None,
            })

        return pd.DataFrame(records)

    async def collect_and_save(
        self,
        start_date: datetime,
        end_date: datetime,
        tickers: Optional[List[str]] = None,
    ) -> int:
        """
        Collect and save data to Parquet files.

        Args:
            start_date: Start date
            end_date: End date
            tickers: List of tickers

        Returns:
            Total records saved
        """
        results = await self.collect_historical(start_date, end_date, tickers)

        if not results:
            return 0

        # Combine all ticker data
        all_data = []
        for ticker_symbol, df in results.items():
            all_data.append(df)

        if not all_data:
            return 0

        combined_df = pd.concat(all_data, ignore_index=True)

        # Group by month and save
        combined_df['partition_key'] = combined_df['timestamp'].dt.strftime('%Y%m')

        total_saved = 0
        for partition_key, group in combined_df.groupby('partition_key'):
            # Drop partition_key column before saving
            group = group.drop(columns=['partition_key'])

            # Use base class write_parquet (df, partition_key, mode)
            self.write_parquet(group, partition_key, mode='append')
            total_saved += len(group)

        self.logger.info(f"Saved {total_saved} total records")
        return total_saved

    def load_data(
        self,
        start_date: datetime,
        end_date: datetime,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load data from Parquet files.

        Args:
            start_date: Start date
            end_date: End date
            tickers: Filter by tickers (optional)

        Returns:
            DataFrame with macro data
        """
        # Get relevant partition keys
        partition_keys = []
        current = start_date.replace(day=1)

        while current <= end_date:
            partition_keys.append(current.strftime('%Y%m'))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        # Load and combine
        dfs = []
        for key in partition_keys:
            file_path = self._get_file_path(key)
            if file_path.exists():
                df = pd.read_parquet(file_path)
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)

        # Filter by time range
        result = result[
            (result['timestamp'] >= start_date) &
            (result['timestamp'] <= end_date)
        ]

        # Filter by tickers if specified
        if tickers:
            result = result[result['indicator_name'].isin(tickers)]

        return result.sort_values('timestamp').reset_index(drop=True)


# ========== Standalone Execution ==========

if __name__ == '__main__':
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description='YFinance Local Collector')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    collector = YFinanceLocalCollector()

    async def main():
        total = await collector.collect_and_save(start, end)
        print(f"Collected and saved {total} records")

    asyncio.run(main())
