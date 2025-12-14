"""
FRED Local Collector - Collect macro economic indicators from FRED API.

Collects 47 key indicators:
- Interest Rates: DFF, DGS1-DGS30, T10Y2Y, T10Y3M
- Liquidity: WALCL, RRPONTSYD, WRESBAL, M1SL, M2SL
- Inflation: CPIAUCSL, CPILFESL, PCEPI, PCEPILFE
- Employment: UNRATE, PAYEMS, ICSA
- Economic Activity: GDP, INDPRO, UMCSENT

API Requirements:
- FRED API key (free registration at https://fred.stlouisfed.org/)
- Rate limit: 120 requests/minute

Output: data/macro/fred_YYYYMM.parquet
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import aiohttp
import asyncio
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import os

from collectors.base_local_collector import BaseLocalCollector
from config.settings import DATA_DIR


class FREDLocalCollector(BaseLocalCollector):
    """
    Collects macro economic indicators from FRED API.

    Data is stored in monthly Parquet partitions.
    """

    # API Configuration
    BASE_URL = 'https://api.stlouisfed.org/fred'
    SERIES_ENDPOINT = '/series/observations'

    # Key macro indicators (47 total)
    DEFAULT_INDICATORS = {
        # ============================================================================
        # Interest Rates & Monetary Policy (11)
        # ============================================================================
        'DFF': {'name': 'Federal Funds Rate', 'category': 'interest_rate', 'unit': '%', 'frequency': 'daily'},
        'DGS1': {'name': '1-Year Treasury Rate', 'category': 'interest_rate', 'unit': '%', 'frequency': 'daily'},
        'DGS2': {'name': '2-Year Treasury Rate', 'category': 'interest_rate', 'unit': '%', 'frequency': 'daily'},
        'DGS3': {'name': '3-Year Treasury Rate', 'category': 'interest_rate', 'unit': '%', 'frequency': 'daily'},
        'DGS5': {'name': '5-Year Treasury Rate', 'category': 'interest_rate', 'unit': '%', 'frequency': 'daily'},
        'DGS7': {'name': '7-Year Treasury Rate', 'category': 'interest_rate', 'unit': '%', 'frequency': 'daily'},
        'DGS10': {'name': '10-Year Treasury Rate', 'category': 'interest_rate', 'unit': '%', 'frequency': 'daily'},
        'DGS20': {'name': '20-Year Treasury Rate', 'category': 'interest_rate', 'unit': '%', 'frequency': 'daily'},
        'DGS30': {'name': '30-Year Treasury Rate', 'category': 'interest_rate', 'unit': '%', 'frequency': 'daily'},
        'T10Y2Y': {'name': '10Y-2Y Treasury Spread', 'category': 'interest_rate', 'unit': '%', 'frequency': 'daily'},
        'T10Y3M': {'name': '10Y-3M Treasury Spread', 'category': 'interest_rate', 'unit': '%', 'frequency': 'daily'},

        # ============================================================================
        # Liquidity & Money Supply (5)
        # ============================================================================
        'WALCL': {'name': 'Fed Balance Sheet', 'category': 'liquidity', 'unit': 'billions', 'frequency': 'weekly'},
        'RRPONTSYD': {'name': 'Reverse Repo Operations', 'category': 'liquidity', 'unit': 'billions', 'frequency': 'daily'},
        'WRESBAL': {'name': 'Reserve Balances', 'category': 'liquidity', 'unit': 'billions', 'frequency': 'weekly'},
        'M1SL': {'name': 'M1 Money Supply', 'category': 'money_supply', 'unit': 'billions', 'frequency': 'monthly'},
        'M2SL': {'name': 'M2 Money Supply', 'category': 'money_supply', 'unit': 'billions', 'frequency': 'monthly'},

        # ============================================================================
        # Inflation (4)
        # ============================================================================
        'CPIAUCSL': {'name': 'Consumer Price Index', 'category': 'inflation', 'unit': 'index', 'frequency': 'monthly'},
        'CPILFESL': {'name': 'Core CPI', 'category': 'inflation', 'unit': 'index', 'frequency': 'monthly'},
        'PCEPI': {'name': 'PCE Price Index', 'category': 'inflation', 'unit': 'index', 'frequency': 'monthly'},
        'PCEPILFE': {'name': 'Core PCE', 'category': 'inflation', 'unit': 'index', 'frequency': 'monthly'},

        # ============================================================================
        # Employment (3)
        # ============================================================================
        'UNRATE': {'name': 'Unemployment Rate', 'category': 'employment', 'unit': '%', 'frequency': 'monthly'},
        'PAYEMS': {'name': 'Nonfarm Payrolls', 'category': 'employment', 'unit': 'thousands', 'frequency': 'monthly'},
        'ICSA': {'name': 'Initial Jobless Claims', 'category': 'employment', 'unit': 'thousands', 'frequency': 'weekly'},

        # ============================================================================
        # Economic Activity (3)
        # ============================================================================
        'GDP': {'name': 'Gross Domestic Product', 'category': 'gdp', 'unit': 'billions', 'frequency': 'quarterly'},
        'INDPRO': {'name': 'Industrial Production', 'category': 'production', 'unit': 'index', 'frequency': 'monthly'},
        'UMCSENT': {'name': 'Consumer Sentiment', 'category': 'sentiment', 'unit': 'index', 'frequency': 'monthly'},

        # ============================================================================
        # Risk & Credit (1)
        # ============================================================================
        'BAMLH0A0HYM2': {'name': 'High Yield Spread', 'category': 'credit', 'unit': '%', 'frequency': 'daily'},

        # ============================================================================
        # Volatility (1)
        # ============================================================================
        'VIXCLS': {'name': 'VIX Index', 'category': 'volatility', 'unit': 'index', 'frequency': 'daily'},

        # ============================================================================
        # Currency (3)
        # ============================================================================
        'DTWEXBGS': {'name': 'Trade Weighted Dollar', 'category': 'currency', 'unit': 'index', 'frequency': 'daily'},
        'DEXJPUS': {'name': 'USD/JPY', 'category': 'currency', 'unit': 'rate', 'frequency': 'daily'},
        'DEXCHUS': {'name': 'USD/CNY', 'category': 'currency', 'unit': 'rate', 'frequency': 'daily'},

        # ============================================================================
        # Commodities (3)
        # ============================================================================
        'DCOILWTICO': {'name': 'WTI Crude Oil', 'category': 'commodity', 'unit': 'USD/barrel', 'frequency': 'daily'},
        'GOLDAMGBD228NLBM': {'name': 'Gold Price', 'category': 'commodity', 'unit': 'USD/oz', 'frequency': 'daily'},
        'DCOILBRENTEU': {'name': 'Brent Crude Oil', 'category': 'commodity', 'unit': 'USD/barrel', 'frequency': 'daily'},

        # ============================================================================
        # Housing (3)
        # ============================================================================
        'MORTGAGE30US': {'name': '30-Year Mortgage Rate', 'category': 'housing', 'unit': '%', 'frequency': 'weekly'},
        'CSUSHPISA': {'name': 'Case-Shiller Home Price', 'category': 'housing', 'unit': 'index', 'frequency': 'monthly'},
        'HOUST': {'name': 'Housing Starts', 'category': 'housing', 'unit': 'thousands', 'frequency': 'monthly'},

        # ============================================================================
        # Labor (3)
        # ============================================================================
        'U6RATE': {'name': 'U6 Unemployment', 'category': 'labor', 'unit': '%', 'frequency': 'monthly'},
        'CIVPART': {'name': 'Labor Participation', 'category': 'labor', 'unit': '%', 'frequency': 'monthly'},
        'AWHAETP': {'name': 'Avg Weekly Hours', 'category': 'labor', 'unit': 'hours', 'frequency': 'monthly'},

        # ============================================================================
        # Business (2)
        # ============================================================================
        'DGORDER': {'name': 'Durable Goods Orders', 'category': 'business', 'unit': 'millions', 'frequency': 'monthly'},
        'RSAFS': {'name': 'Retail Sales', 'category': 'business', 'unit': 'millions', 'frequency': 'monthly'},

        # ============================================================================
        # Consumer (2)
        # ============================================================================
        'PCE': {'name': 'Personal Consumption', 'category': 'consumer', 'unit': 'billions', 'frequency': 'monthly'},
        'PSAVERT': {'name': 'Personal Saving Rate', 'category': 'consumer', 'unit': '%', 'frequency': 'monthly'},

        # ============================================================================
        # Debt (2)
        # ============================================================================
        'GFDEBTN': {'name': 'Federal Debt', 'category': 'debt', 'unit': 'millions', 'frequency': 'quarterly'},
        'TOTRESNS': {'name': 'Total Reserves', 'category': 'debt', 'unit': 'billions', 'frequency': 'monthly'},

        # ============================================================================
        # Trade (1)
        # ============================================================================
        'BOPGSTB': {'name': 'Trade Balance', 'category': 'trade', 'unit': 'millions', 'frequency': 'monthly'},
    }

    # PyArrow schema for Parquet files
    SCHEMA = pa.schema([
        ('timestamp', pa.timestamp('us', tz='UTC')),
        ('indicator_name', pa.string()),
        ('indicator_category', pa.string()),
        ('value', pa.float64()),
        ('source', pa.string()),
        ('unit', pa.string()),
        ('frequency', pa.string()),
    ])

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        indicators: Optional[List[str]] = None,
    ):
        """
        Initialize FRED Local Collector.

        Args:
            data_dir: Base data directory
            api_key: FRED API key (or set FRED_API_KEY env variable)
            indicators: List of series IDs to collect (default: all 47)
        """
        super().__init__(
            data_dir=data_dir or DATA_DIR / 'macro',
            partition_strategy='monthly',
        )

        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        if not self.api_key:
            self.logger.warning(
                "No FRED API key provided. Set FRED_API_KEY env variable or pass api_key parameter."
            )

        self.indicators = indicators or list(self.DEFAULT_INDICATORS.keys())
        self.logger.info(f"FRED collector initialized with {len(self.indicators)} indicators")

    @property
    def collector_name(self) -> str:
        """Return collector name."""
        return "fred"

    @property
    def schema(self) -> pa.Schema:
        """Return PyArrow schema."""
        return self.SCHEMA

    async def collect(self, start_date: datetime, end_date: datetime) -> int:
        """Collect data for date range (required by base class)."""
        return await self.collect_and_save(start_date, end_date)

    async def _fetch_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
        session: aiohttp.ClientSession,
    ) -> List[Dict]:
        """
        Fetch observations for a FRED series.

        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            session: aiohttp session

        Returns:
            List of observation dicts
        """
        url = f"{self.BASE_URL}{self.SERIES_ENDPOINT}"
        params = {
            'api_key': self.api_key,
            'file_type': 'json',
            'series_id': series_id,
            'observation_start': start_date,
            'observation_end': end_date,
            'sort_order': 'asc'
        }

        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('observations', [])
                else:
                    self.logger.error(f"FRED API error for {series_id}: {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"Error fetching {series_id}: {e}")
            return []

    async def collect_historical(
        self,
        start_date: datetime,
        end_date: datetime,
        indicators: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect historical data from FRED API.

        Args:
            start_date: Start date
            end_date: End date
            indicators: List of series IDs (default: self.indicators)

        Returns:
            Dict mapping series_id to DataFrame
        """
        if not self.api_key:
            self.logger.error("FRED API key not set")
            return {}

        indicators = indicators or self.indicators
        results = {}

        self.logger.info(
            f"Collecting FRED data from {start_date.date()} to {end_date.date()} "
            f"for {len(indicators)} indicators"
        )

        async with aiohttp.ClientSession() as session:
            # Rate limiting: 120 requests/minute = 2 requests/second
            for i, series_id in enumerate(indicators):
                if i > 0 and i % 100 == 0:
                    self.logger.info(f"Progress: {i}/{len(indicators)} indicators")

                observations = await self._fetch_series(
                    series_id,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    session
                )

                if observations:
                    records = []
                    indicator_info = self.DEFAULT_INDICATORS.get(series_id, {})

                    for obs in observations:
                        # Parse value (handle missing data marked as '.')
                        value_str = obs.get('value', '')
                        if value_str == '.' or value_str == '':
                            continue

                        try:
                            value = float(value_str)
                        except ValueError:
                            continue

                        # Parse date
                        date_str = obs.get('date', '')
                        try:
                            timestamp = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                        except ValueError:
                            continue

                        records.append({
                            'timestamp': timestamp,
                            'indicator_name': series_id,
                            'indicator_category': indicator_info.get('category', 'unknown'),
                            'value': value,
                            'source': 'fred',
                            'unit': indicator_info.get('unit', ''),
                            'frequency': indicator_info.get('frequency', 'unknown'),
                        })

                    if records:
                        results[series_id] = pd.DataFrame(records)
                        self.logger.debug(f"Collected {len(records)} records for {series_id}")

                # Rate limiting: wait 0.5s between requests
                await asyncio.sleep(0.5)

        self.logger.info(f"Collected data for {len(results)} indicators")
        return results

    async def collect_and_save(
        self,
        start_date: datetime,
        end_date: datetime,
        indicators: Optional[List[str]] = None,
    ) -> int:
        """
        Collect and save data to Parquet files.

        Args:
            start_date: Start date
            end_date: End date
            indicators: List of series IDs

        Returns:
            Total records saved
        """
        results = await self.collect_historical(start_date, end_date, indicators)

        if not results:
            return 0

        # Combine all indicator data
        all_data = []
        for series_id, df in results.items():
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
        indicators: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load data from Parquet files.

        Args:
            start_date: Start date
            end_date: End date
            indicators: Filter by series IDs (optional)

        Returns:
            DataFrame with FRED data
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

        # Filter by indicators if specified
        if indicators:
            result = result[result['indicator_name'].isin(indicators)]

        return result.sort_values('timestamp').reset_index(drop=True)

    def get_latest_value(
        self,
        indicator_name: str,
        as_of_date: datetime,
    ) -> Optional[float]:
        """
        Get the latest value for an indicator as of a given date.

        Args:
            indicator_name: FRED series ID
            as_of_date: Reference date

        Returns:
            Latest value or None
        """
        # Load recent data
        start = as_of_date - timedelta(days=365)  # Look back up to 1 year
        df = self.load_data(start, as_of_date, indicators=[indicator_name])

        if df.empty:
            return None

        # Get most recent observation
        df = df.sort_values('timestamp', ascending=False)
        return df.iloc[0]['value']


# ========== Standalone Execution ==========

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='FRED Local Collector')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--api-key', type=str, help='FRED API key')

    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    collector = FREDLocalCollector(api_key=args.api_key)

    async def main():
        total = await collector.collect_and_save(start, end)
        print(f"Collected and saved {total} records")

    asyncio.run(main())
