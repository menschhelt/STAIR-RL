"""
Alpha Calculator - Pre-calculate ALL indicators for the entire symbol pool.

Strategy:
1. Load all Parquet data
2. For each symbol: calculate all alphas + factors
3. Cache results for fast backtesting

This pre-calculation approach enables:
- Fast backtesting (no repeated calculations)
- Consistent feature values across experiments
- Memory-efficient streaming during training
"""

import asyncio
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from config.settings import DATA_DIR
from features.alpha_adapter import AlphaAdapter


class AlphaCalculator:
    """
    Pre-calculates alpha factors for all symbols.

    Features:
    - Batch processing for efficiency
    - Parallel symbol processing (multiprocessing)
    - Incremental updates (only new data)
    - Parquet caching
    """

    def __init__(
        self,
        binance_data_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        alpha_adapter: Optional[AlphaAdapter] = None,
    ):
        """
        Initialize Alpha Calculator.

        Args:
            binance_data_dir: Directory with Binance Parquet files
            cache_dir: Directory to cache calculated features
            alpha_adapter: AlphaAdapter instance
        """
        self.binance_data_dir = binance_data_dir or DATA_DIR / 'binance'
        self.cache_dir = cache_dir or DATA_DIR / 'features' / 'alpha_cache'
        self.alpha_adapter = alpha_adapter or AlphaAdapter()

        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

        # Get available alphas
        self.alpha_names = self.alpha_adapter.list_alphas()
        self.logger.info(f"AlphaCalculator initialized with {len(self.alpha_names)} alphas")

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

    def get_cache_path(self, symbol: str) -> Path:
        """Get cache file path for a symbol."""
        return self.cache_dir / f"{symbol}_features.parquet"

    # ========== Data Loading ==========

    def load_symbol_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '5m',
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a symbol from Parquet files.

        Args:
            symbol: Trading pair
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            DataFrame with OHLCV data
        """
        # Find relevant monthly partitions
        dfs = []
        current = start_date

        while current <= end_date:
            partition_key = current.strftime('%Y%m')
            file_path = self.binance_data_dir / f"binance_futures_{interval}_{partition_key}.parquet"

            if file_path.exists():
                df = pq.read_table(file_path).to_pandas()
                # Filter to symbol
                df = df[df['symbol'] == symbol]
                if not df.empty:
                    dfs.append(df)

            # Move to next month
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                current = datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)

        # Filter to exact date range
        if 'timestamp' in result.columns:
            result = result[
                (result['timestamp'] >= start_date) &
                (result['timestamp'] <= end_date)
            ]
            result = result.sort_values('timestamp').reset_index(drop=True)

        return result

    def get_all_symbols(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str = '5m',
    ) -> Set[str]:
        """
        Get all symbols present in the data range.

        Args:
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            Set of symbol names
        """
        symbols = set()
        current = start_date

        while current <= end_date:
            partition_key = current.strftime('%Y%m')
            file_path = self.binance_data_dir / f"binance_futures_{interval}_{partition_key}.parquet"

            if file_path.exists():
                df = pq.read_table(file_path, columns=['symbol']).to_pandas()
                symbols.update(df['symbol'].unique())

            # Move to next month
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                current = datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)

        return symbols

    # ========== Feature Calculation ==========

    def calculate_symbol_features(
        self,
        symbol: str,
        dataframe: pd.DataFrame,
        btc_dataframe: Optional[pd.DataFrame] = None,
        alpha_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate all features for a single symbol.

        Args:
            symbol: Trading pair
            dataframe: OHLCV DataFrame
            btc_dataframe: BTC data for market factor
            alpha_names: Specific alphas to calculate (None = all)

        Returns:
            DataFrame with all features
        """
        if dataframe.empty:
            return pd.DataFrame()

        alphas = alpha_names or self.alpha_names
        features = {'timestamp': dataframe['timestamp'].values}

        # Calculate all alphas
        for alpha_name in alphas:
            try:
                alpha_values = self.alpha_adapter.calculate_alpha(
                    alpha_name, dataframe, symbol
                )
                features[alpha_name] = alpha_values.values
            except Exception as e:
                self.logger.debug(f"Error calculating {alpha_name} for {symbol}: {e}")
                features[alpha_name] = np.zeros(len(dataframe))

        # Calculate 9 factors
        try:
            factor_df = self.alpha_adapter.calculate_factors(
                dataframe, symbol, btc_dataframe
            )
            for col in factor_df.columns:
                features[f'factor_{col}'] = factor_df[col].values
        except Exception as e:
            self.logger.debug(f"Error calculating factors for {symbol}: {e}")

        # Add original OHLCV for reference
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            if col in dataframe.columns:
                features[col] = dataframe[col].values

        # Add futures-specific fields if present
        for col in ['funding_rate', 'mark_price', 'open_interest']:
            if col in dataframe.columns:
                features[col] = dataframe[col].values

        result = pd.DataFrame(features)
        result['symbol'] = symbol

        return result

    # ========== Batch Processing ==========

    def precalculate_all(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: Optional[List[str]] = None,
        interval: str = '5m',
        parallel: bool = True,
        max_workers: int = None,
    ) -> Dict[str, Path]:
        """
        Pre-calculate all features for all symbols.

        Args:
            start_date: Start date
            end_date: End date
            symbols: List of symbols (None = all)
            interval: Data interval
            parallel: Use parallel processing
            max_workers: Number of parallel workers

        Returns:
            Dict mapping symbol -> cache file path
        """
        # Get symbols
        if symbols is None:
            symbols = sorted(self.get_all_symbols(start_date, end_date, interval))

        self.logger.info(
            f"Pre-calculating features for {len(symbols)} symbols "
            f"from {start_date} to {end_date}"
        )

        # Load BTC data for market factor (all symbols need this)
        btc_df = self.load_symbol_data('BTCUSDT', start_date, end_date, interval)
        if btc_df.empty:
            self.logger.warning("No BTC data found, market factor will be zero")
            btc_df = None

        cache_paths = {}

        if parallel and len(symbols) > 1:
            max_workers = max_workers or min(mp.cpu_count() - 1, 8)
            self.logger.info(f"Using {max_workers} parallel workers")

            # Process in parallel (using threads for I/O bound work)
            from concurrent.futures import ThreadPoolExecutor

            def process_symbol(sym):
                return self._process_single_symbol(
                    sym, start_date, end_date, interval, btc_df
                )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_symbol, symbols))

            for sym, path in zip(symbols, results):
                if path:
                    cache_paths[sym] = path

        else:
            # Sequential processing
            for sym in symbols:
                path = self._process_single_symbol(
                    sym, start_date, end_date, interval, btc_df
                )
                if path:
                    cache_paths[sym] = path

        self.logger.info(f"Completed pre-calculation for {len(cache_paths)} symbols")
        return cache_paths

    def _process_single_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        btc_df: Optional[pd.DataFrame],
    ) -> Optional[Path]:
        """
        Process a single symbol (for parallel execution).

        Returns:
            Cache file path or None
        """
        try:
            # Load data
            df = self.load_symbol_data(symbol, start_date, end_date, interval)
            if df.empty:
                return None

            # Calculate features
            features = self.calculate_symbol_features(symbol, df, btc_df)
            if features.empty:
                return None

            # Save to cache
            cache_path = self.get_cache_path(symbol)
            features.to_parquet(cache_path, compression='snappy')

            self.logger.debug(f"Cached {len(features)} rows for {symbol}")
            return cache_path

        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            return None

    # ========== Cache Management ==========

    def load_cached_features(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load pre-calculated features from cache.

        Args:
            symbol: Trading pair
            start_date: Filter start date
            end_date: Filter end date
            columns: Specific columns to load

        Returns:
            DataFrame with features
        """
        cache_path = self.get_cache_path(symbol)

        if not cache_path.exists():
            self.logger.warning(f"No cache found for {symbol}")
            return pd.DataFrame()

        df = pq.read_table(cache_path, columns=columns).to_pandas()

        # Filter by date range
        if 'timestamp' in df.columns:
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]

        return df

    def list_cached_symbols(self) -> List[str]:
        """List all symbols with cached features."""
        symbols = []
        for path in self.cache_dir.glob('*_features.parquet'):
            symbol = path.stem.replace('_features', '')
            symbols.append(symbol)
        return sorted(symbols)

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear feature cache.

        Args:
            symbol: Specific symbol to clear (None = all)
        """
        if symbol:
            path = self.get_cache_path(symbol)
            if path.exists():
                path.unlink()
                self.logger.info(f"Cleared cache for {symbol}")
        else:
            for path in self.cache_dir.glob('*_features.parquet'):
                path.unlink()
            self.logger.info("Cleared all feature cache")

    def get_cache_stats(self) -> Dict:
        """Get statistics about cached features."""
        files = list(self.cache_dir.glob('*_features.parquet'))
        total_size = sum(f.stat().st_size for f in files)

        return {
            'cached_symbols': len(files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_dir': str(self.cache_dir),
        }


# ========== Standalone Execution ==========

if __name__ == '__main__':
    import argparse
    from datetime import timezone

    parser = argparse.ArgumentParser(description='Alpha Calculator')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, nargs='*', help='Specific symbols')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')

    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    calculator = AlphaCalculator()

    paths = calculator.precalculate_all(
        start_date=start,
        end_date=end,
        symbols=args.symbols,
        parallel=args.parallel,
        max_workers=args.workers,
    )

    print(f"Calculated features for {len(paths)} symbols")
    print(f"Cache stats: {calculator.get_cache_stats()}")
