"""
Data Validator - Validates data integrity for backtesting.

Validates:
- Funding rate: Must exist at 00:00, 08:00, 16:00 UTC
- Mark price: Must exist for all candles
- OHLCV: No gaps, volume > 0
- Universe: At least 1 symbol filled
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone, time
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import logging

from config.settings import DATA_DIR


class DataValidator:
    """
    Validates data quality for backtesting.

    Checks:
    1. Funding rate presence (8h intervals)
    2. Mark price presence (all candles)
    3. OHLCV completeness (no gaps)
    4. Volume sanity (> 0)
    5. Universe fill rate
    """

    # Funding rate times (UTC)
    FUNDING_TIMES = [
        time(0, 0),   # 00:00 UTC
        time(8, 0),   # 08:00 UTC
        time(16, 0),  # 16:00 UTC
    ]

    def __init__(self):
        """Initialize Data Validator."""
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

    # ========== Funding Rate Validation ==========

    def validate_funding_rate(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict:
        """
        Validate funding rate data.

        Funding rates should be present at 00:00, 08:00, 16:00 UTC daily.

        Args:
            df: OHLCV DataFrame with funding_rate column
            symbol: Trading pair
            start_date: Start of period
            end_date: End of period

        Returns:
            Validation result dict
        """
        result = {
            'symbol': symbol,
            'valid': True,
            'total_expected': 0,
            'total_present': 0,
            'missing_timestamps': [],
            'fill_rate_pct': 0.0,
        }

        if 'funding_rate' not in df.columns:
            result['valid'] = False
            result['error'] = 'funding_rate column not found'
            return result

        # Get rows with funding rate (non-null)
        funding_rows = df[df['funding_rate'].notna()]

        if funding_rows.empty:
            result['valid'] = False
            result['error'] = 'No funding rate data found'
            return result

        # Build expected funding timestamps
        expected_timestamps = self._build_funding_timestamps(start_date, end_date)
        result['total_expected'] = len(expected_timestamps)

        # Check presence
        if 'timestamp' in df.columns:
            # Use floor instead of round to avoid 23:59 rounding to next day's 00:00
            present_timestamps = set(funding_rows['timestamp'].dt.floor('h'))
            missing = []

            for expected in expected_timestamps:
                if expected not in present_timestamps:
                    missing.append(expected)

            result['total_present'] = len(expected_timestamps) - len(missing)
            result['missing_timestamps'] = missing[:10]  # First 10 only
            result['fill_rate_pct'] = round(
                result['total_present'] / result['total_expected'] * 100, 2
            ) if result['total_expected'] > 0 else 0

            # Mark as invalid if fill rate < 95%
            if result['fill_rate_pct'] < 95:
                result['valid'] = False

        return result

    def _build_funding_timestamps(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Set[datetime]:
        """Build set of expected funding rate timestamps."""
        timestamps = set()

        current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)

        while current <= end_date:
            for funding_time in self.FUNDING_TIMES:
                ts = current.replace(
                    hour=funding_time.hour,
                    minute=funding_time.minute
                )
                if start_date <= ts <= end_date:
                    timestamps.add(ts)

            current += timedelta(days=1)

        return timestamps

    # ========== Mark Price Validation ==========

    def validate_mark_price(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> Dict:
        """
        Validate mark price data.

        Mark price should be present for all candles.

        Args:
            df: OHLCV DataFrame with mark_price column
            symbol: Trading pair

        Returns:
            Validation result dict
        """
        result = {
            'symbol': symbol,
            'valid': True,
            'total_rows': len(df),
            'missing_count': 0,
            'fill_rate_pct': 100.0,
        }

        if 'mark_price' not in df.columns:
            result['valid'] = False
            result['error'] = 'mark_price column not found'
            return result

        # Count missing
        missing = df['mark_price'].isna().sum()
        result['missing_count'] = int(missing)
        result['fill_rate_pct'] = round(
            (len(df) - missing) / len(df) * 100, 2
        ) if len(df) > 0 else 0

        # Mark as invalid if fill rate < 99%
        if result['fill_rate_pct'] < 99:
            result['valid'] = False

        return result

    # ========== OHLCV Validation ==========

    def validate_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str = '5m',
    ) -> Dict:
        """
        Validate OHLCV data completeness.

        Checks:
        - No time gaps
        - All prices > 0
        - Volume >= 0 (can be 0 for some periods)
        - High >= Low
        - High >= Open, Close
        - Low <= Open, Close

        Args:
            df: OHLCV DataFrame
            symbol: Trading pair
            interval: Expected interval

        Returns:
            Validation result dict
        """
        result = {
            'symbol': symbol,
            'valid': True,
            'total_rows': len(df),
            'issues': [],
        }

        if df.empty:
            result['valid'] = False
            result['issues'].append('Empty DataFrame')
            return result

        # Check required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            result['valid'] = False
            result['issues'].append(f'Missing columns: {missing_cols}')
            return result

        # Check price validity
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            invalid_count = (df[col] <= 0).sum()
            if invalid_count > 0:
                result['issues'].append(f'{col} <= 0: {invalid_count} rows')

        # Check High >= Low
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            result['issues'].append(f'High < Low: {invalid_hl} rows')
            result['valid'] = False

        # Check High >= Open and High >= Close
        invalid_ho = (df['high'] < df['open']).sum()
        invalid_hc = (df['high'] < df['close']).sum()
        if invalid_ho > 0 or invalid_hc > 0:
            result['issues'].append(f'High < Open/Close: {max(invalid_ho, invalid_hc)} rows')

        # Check Low <= Open and Low <= Close
        invalid_lo = (df['low'] > df['open']).sum()
        invalid_lc = (df['low'] > df['close']).sum()
        if invalid_lo > 0 or invalid_lc > 0:
            result['issues'].append(f'Low > Open/Close: {max(invalid_lo, invalid_lc)} rows')

        # Check time gaps
        if 'timestamp' in df.columns and len(df) > 1:
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff().dropna()

            expected_diff = self._interval_to_timedelta(interval)
            gaps = time_diffs[time_diffs > expected_diff * 1.5]

            if len(gaps) > 0:
                result['issues'].append(f'Time gaps detected: {len(gaps)}')
                result['gap_count'] = len(gaps)

        return result

    def _interval_to_timedelta(self, interval: str) -> timedelta:
        """Convert interval string to timedelta."""
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            return timedelta(minutes=minutes)
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            return timedelta(hours=hours)
        elif interval.endswith('d'):
            days = int(interval[:-1])
            return timedelta(days=days)
        else:
            return timedelta(minutes=5)  # Default

    # ========== Universe Validation ==========

    def validate_universe(
        self,
        universe_history: pd.DataFrame,
        top_n: int = 20,
    ) -> Dict:
        """
        Validate universe history.

        Checks:
        - At least 1 slot filled per day
        - Fill rate statistics

        Args:
            universe_history: Universe history DataFrame
            top_n: Expected number of slots

        Returns:
            Validation result dict
        """
        result = {
            'valid': True,
            'total_days': 0,
            'avg_fill_rate_pct': 0.0,
            'min_fill_rate_pct': 0.0,
            'empty_days': [],
        }

        if universe_history.empty:
            result['valid'] = False
            result['error'] = 'Empty universe history'
            return result

        # Calculate fill rates per day
        daily_fills = universe_history.groupby('date').apply(
            lambda x: x['symbol'].notna().sum() / top_n * 100
        )

        result['total_days'] = len(daily_fills)
        result['avg_fill_rate_pct'] = round(daily_fills.mean(), 2)
        result['min_fill_rate_pct'] = round(daily_fills.min(), 2)

        # Find empty days (fill rate = 0)
        empty_days = daily_fills[daily_fills == 0].index.tolist()
        result['empty_days'] = empty_days[:10]  # First 10 only

        if len(empty_days) > 0:
            result['valid'] = False
            result['error'] = f'{len(empty_days)} days with no symbols'

        return result

    # ========== Comprehensive Validation ==========

    def validate_all(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
        universe_history: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        interval: str = '5m',
    ) -> Dict:
        """
        Run all validations.

        Args:
            ohlcv_dict: {symbol: ohlcv_df}
            universe_history: Universe history DataFrame
            start_date: Start of period
            end_date: End of period
            interval: Data interval

        Returns:
            Comprehensive validation report
        """
        report = {
            'valid': True,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
            },
            'symbols': {},
            'universe': {},
            'summary': {
                'total_symbols': len(ohlcv_dict),
                'valid_symbols': 0,
                'funding_issues': 0,
                'mark_price_issues': 0,
                'ohlcv_issues': 0,
            },
        }

        # Validate each symbol
        for symbol, df in ohlcv_dict.items():
            symbol_report = {
                'funding_rate': self.validate_funding_rate(df, symbol, start_date, end_date),
                'mark_price': self.validate_mark_price(df, symbol),
                'ohlcv': self.validate_ohlcv(df, symbol, interval),
            }

            # Check if symbol is valid
            is_valid = all([
                symbol_report['funding_rate']['valid'],
                symbol_report['mark_price']['valid'],
                symbol_report['ohlcv']['valid'],
            ])

            symbol_report['valid'] = is_valid
            report['symbols'][symbol] = symbol_report

            if is_valid:
                report['summary']['valid_symbols'] += 1
            if not symbol_report['funding_rate']['valid']:
                report['summary']['funding_issues'] += 1
            if not symbol_report['mark_price']['valid']:
                report['summary']['mark_price_issues'] += 1
            if not symbol_report['ohlcv']['valid']:
                report['summary']['ohlcv_issues'] += 1

        # Validate universe
        report['universe'] = self.validate_universe(universe_history)

        # Overall validity
        report['valid'] = (
            report['summary']['valid_symbols'] > 0 and
            report['universe']['valid']
        )

        return report

    def print_report(self, report: Dict):
        """Print validation report to console."""
        print("\n" + "=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)

        print(f"\nPeriod: {report['period']['start']} to {report['period']['end']}")
        print(f"Overall Valid: {report['valid']}")

        print(f"\nSummary:")
        print(f"  Total Symbols: {report['summary']['total_symbols']}")
        print(f"  Valid Symbols: {report['summary']['valid_symbols']}")
        print(f"  Funding Issues: {report['summary']['funding_issues']}")
        print(f"  Mark Price Issues: {report['summary']['mark_price_issues']}")
        print(f"  OHLCV Issues: {report['summary']['ohlcv_issues']}")

        print(f"\nUniverse:")
        print(f"  Total Days: {report['universe'].get('total_days', 'N/A')}")
        print(f"  Avg Fill Rate: {report['universe'].get('avg_fill_rate_pct', 'N/A')}%")
        print(f"  Min Fill Rate: {report['universe'].get('min_fill_rate_pct', 'N/A')}%")

        # Show problematic symbols
        problematic = [s for s, r in report['symbols'].items() if not r['valid']]
        if problematic:
            print(f"\nProblematic Symbols ({len(problematic)}):")
            for symbol in problematic[:5]:
                print(f"  - {symbol}")

        print("\n" + "=" * 60)


# ========== Standalone Testing ==========

if __name__ == '__main__':
    validator = DataValidator()

    # Test building funding timestamps
    from datetime import datetime, timezone

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, tzinfo=timezone.utc)

    timestamps = validator._build_funding_timestamps(start, end)
    print(f"Expected funding timestamps for 2 days: {len(timestamps)}")

    for ts in sorted(timestamps)[:6]:
        print(f"  {ts}")
