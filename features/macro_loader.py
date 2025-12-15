"""
MacroDataLoader - Load macro indicators with forward-fill strategy.

Loads macro indicators from FRED and yfinance monthly parquet files.
Supports forward-fill for 5-minute timestamps (use last observed value).

Integrates Fama-French 5-factor model for cryptocurrency markets.

Data files:
- FRED: /home/work/data/stair-local/macro/fred_YYYYMM.parquet
- yfinance: /home/work/data/stair-local/macro/yfinance_YYYYMM.parquet

Usage:
    loader = MacroDataLoader()
    # Option 1: Macro only (23-dim)
    features = loader.get_global_features(timestamp)

    # Option 2: Macro + Fama-French (28-dim)
    features = loader.get_global_features(
        timestamp,
        universe_returns=returns,  # (N,) array
        universe_market_caps=market_caps,  # (N,) array
        universe_volumes=volumes,  # (N,) array
        universe_volatilities=volatilities,  # (N,) array
    )
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging
from tqdm import tqdm

from config.settings import DATA_DIR
from features.fama_french import FamaFrenchCalculator

logger = logging.getLogger(__name__)


class MacroDataLoader:
    """
    Load macro indicators with forward-fill for 5-minute timestamps.

    Handles different update frequencies:
    - Daily: VIX, DFF, equity indices
    - Weekly: Fed balance sheet, initial claims
    - Monthly: CPI, employment, GDP components
    - Quarterly: GDP

    Forward-fill strategy: For any 5-minute timestamp, use the last observed value.
    """

    # Selected 23 indicators (recommended default)
    DEFAULT_INDICATORS = [
        # Interest rates (5)
        'DFF', 'DGS2', 'DGS10', 'T10Y2Y', 'T10Y3M',

        # Risk & volatility (3)
        '^VIX', 'VIXCLS', 'BAMLH0A0HYM2',

        # Equity (1)
        '^GSPC',

        # Monetary (3)
        'WALCL', 'M1SL', 'M2SL',

        # Inflation (2)
        'CPIAUCSL', 'PCEPI',

        # Employment (2)
        'UNRATE', 'PAYEMS',

        # Economic activity (3)
        'GDP', 'INDPRO', 'UMCSENT',

        # Commodities (3)
        'DCOILWTICO', 'GC=F', 'DX-Y.NYB',

        # Housing (1)
        'CSUSHPISA',
    ]  # Total: 23 indicators

    def __init__(
        self,
        macro_data_dir: Optional[Path] = None,
        indicators: Optional[List[str]] = None,
        cache_months: int = 12,
        include_fama_french: bool = True,
    ):
        """
        Initialize MacroDataLoader.

        Args:
            macro_data_dir: Path to macro data directory (default: DATA_DIR/macro)
            indicators: List of indicator names to load (default: DEFAULT_INDICATORS)
            cache_months: Number of months to keep in memory cache
            include_fama_french: Whether to include Fama-French factors (default: True)
        """
        self.macro_data_dir = macro_data_dir or (DATA_DIR / 'macro')
        self.indicators = indicators or self.DEFAULT_INDICATORS
        self.cache_months = cache_months
        self.include_fama_french = include_fama_french

        # Feature count: 23 macro + 5 Fama-French (if enabled)
        self.n_macro_features = len(self.indicators)
        self.n_ff_features = 5 if include_fama_french else 0
        self.n_features = self.n_macro_features + self.n_ff_features

        # Cache: {(year, month): DataFrame}
        self._fred_cache: Dict[tuple, pd.DataFrame] = {}
        self._yfinance_cache: Dict[tuple, pd.DataFrame] = {}

        # Fast numpy preload (initialized by preload_all())
        self._preloaded_data: Optional[np.ndarray] = None
        self._timestamp_to_idx: Optional[Dict] = None

        # Initialize Fama-French calculator
        if include_fama_french:
            self.ff_calculator = FamaFrenchCalculator()
        else:
            self.ff_calculator = None

        logger.info(
            f"MacroDataLoader initialized with {self.n_macro_features} macro indicators "
            f"+ {self.n_ff_features} Fama-French factors = {self.n_features} total features"
        )

    def _load_month_fred(self, year: int, month: int) -> pd.DataFrame:
        """Load FRED data for a specific month with caching."""
        key = (year, month)

        if key in self._fred_cache:
            return self._fred_cache[key]

        # Load from file
        file_path = self.macro_data_dir / f"fred_{year:04d}{month:02d}.parquet"

        if not file_path.exists():
            logger.warning(f"FRED file not found: {file_path}")
            return pd.DataFrame()

        df = pd.read_parquet(file_path)

        # Filter to selected indicators
        df = df[df['indicator_name'].isin(self.indicators)]

        # Ensure timestamp is timezone-aware
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Cache management (keep only recent months)
        if len(self._fred_cache) >= self.cache_months:
            # Remove oldest entry
            oldest_key = min(self._fred_cache.keys())
            del self._fred_cache[oldest_key]

        self._fred_cache[key] = df
        return df

    def _load_month_yfinance(self, year: int, month: int) -> pd.DataFrame:
        """Load yfinance data for a specific month with caching."""
        key = (year, month)

        if key in self._yfinance_cache:
            return self._yfinance_cache[key]

        # Load from file
        file_path = self.macro_data_dir / f"yfinance_{year:04d}{month:02d}.parquet"

        if not file_path.exists():
            logger.warning(f"yfinance file not found: {file_path}")
            return pd.DataFrame()

        df = pd.read_parquet(file_path)

        # Filter to selected indicators
        df = df[df['indicator_name'].isin(self.indicators)]

        # Ensure timestamp is timezone-aware
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # For yfinance, use 'close' as the primary value
        if 'close' in df.columns:
            df['value'] = df['close']

        # Cache management
        if len(self._yfinance_cache) >= self.cache_months:
            oldest_key = min(self._yfinance_cache.keys())
            del self._yfinance_cache[oldest_key]

        self._yfinance_cache[key] = df
        return df

    def get_global_features(
        self,
        timestamp: pd.Timestamp,
        universe_returns: Optional[np.ndarray] = None,
        universe_market_caps: Optional[np.ndarray] = None,
        universe_volumes: Optional[np.ndarray] = None,
        universe_volatilities: Optional[np.ndarray] = None,
        universe_momentum: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract global features with forward-fill.

        Args:
            timestamp: Target timestamp (5-minute bar)
            universe_returns: (N,) array of returns for Top N universe (optional, for Fama-French)
            universe_market_caps: (N,) array of market caps (optional, for Fama-French)
            universe_volumes: (N,) array of volumes (optional, for Fama-French)
            universe_volatilities: (N,) array of volatilities (optional, for Fama-French)
            universe_momentum: (N,) array of momentum (optional, for Fama-French)

        Returns:
            np.ndarray of shape (n_features,) with:
            - [0:23]: Macro indicator values
            - [23:28]: Fama-French factors (if include_fama_french=True and universe data provided)
        """
        # Ensure timestamp is timezone-aware
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')

        # Load data for current month
        year, month = timestamp.year, timestamp.month

        fred_df = self._load_month_fred(year, month)
        yfinance_df = self._load_month_yfinance(year, month)

        # Combine data
        combined_df = pd.concat([fred_df, yfinance_df], ignore_index=True)

        # Filter to data <= timestamp (forward-fill strategy)
        combined_df = combined_df[combined_df['timestamp'] <= timestamp]

        # Initialize feature array: macro + Fama-French
        features = np.zeros(self.n_features, dtype=np.float32)

        # Part 1: Macro indicators (23-dim)
        for i, indicator in enumerate(self.indicators):
            indicator_data = combined_df[combined_df['indicator_name'] == indicator]

            if not indicator_data.empty:
                # Get last value (forward-fill)
                last_value = indicator_data.iloc[-1]['value']

                # Normalize based on indicator type
                features[i] = self._normalize_indicator(indicator, last_value)
            else:
                # No data available - try previous month
                features[i] = self._get_from_previous_month(indicator, year, month, timestamp)

        # Part 2: Fama-French factors (5-dim) if enabled and universe data provided
        if self.include_fama_french and universe_returns is not None:
            # Check if all required data is provided
            if (universe_market_caps is not None and
                universe_volumes is not None and
                universe_volatilities is not None):

                # Calculate Fama-French factors
                ff_factors = self.ff_calculator.calculate_factors(
                    returns=universe_returns,
                    market_caps=universe_market_caps,
                    volumes=universe_volumes,
                    volatilities=universe_volatilities,
                    price_momentum=universe_momentum,
                )

                # Add to features array: [MKT_RF, SMB, HML, RMW, CMA]
                ff_offset = self.n_macro_features
                features[ff_offset + 0] = ff_factors['MKT_RF']
                features[ff_offset + 1] = ff_factors['SMB']
                features[ff_offset + 2] = ff_factors['HML']
                features[ff_offset + 3] = ff_factors['RMW']
                features[ff_offset + 4] = ff_factors['CMA']
            else:
                # Missing universe data - set Fama-French to zero
                logger.warning(f"Missing universe data for Fama-French calculation at {timestamp}")

        return features

    def _normalize_indicator(self, indicator: str, value: float) -> float:
        """
        Normalize indicator value based on its type.

        Different indicators have different scales:
        - Rates: percentage (0-10)
        - Indices: points (1000-40000)
        - Commodities: price (20-2000)
        """
        # Interest rates (percentage → decimal)
        if indicator in ['DFF', 'DGS2', 'DGS10', 'T10Y2Y', 'T10Y3M', '^TNX', '^FVX']:
            return value / 100.0

        # VIX (volatility index)
        elif indicator in ['^VIX', 'VIXCLS']:
            return value / 100.0  # Normalize to 0-1 range

        # Equity indices (normalize by typical range)
        elif indicator == '^GSPC':
            return value / 10000.0  # S&P 500: 1000-5000 → 0.1-0.5

        # Dollar index
        elif indicator == 'DX-Y.NYB':
            return value / 100.0

        # GDP (normalize by trillion)
        elif indicator == 'GDP':
            return value / 1e12

        # Unemployment rate
        elif indicator == 'UNRATE':
            return value / 100.0

        # Money supply (normalize by trillion)
        elif indicator in ['M1SL', 'M2SL', 'WALCL']:
            return value / 1e12

        # CPI (normalize by 100)
        elif indicator in ['CPIAUCSL', 'CPILFESL', 'PCEPI', 'PCEPILFE']:
            return value / 100.0

        # Industrial production, sentiment (already normalized)
        elif indicator in ['INDPRO', 'UMCSENT']:
            return value / 100.0

        # Commodities (normalize by typical range)
        elif indicator == 'DCOILWTICO':  # Oil price
            return value / 100.0  # 20-120 → 0.2-1.2

        elif indicator == 'GC=F':  # Gold price
            return value / 2000.0  # 1000-2000 → 0.5-1.0

        # High yield spread
        elif indicator == 'BAMLH0A0HYM2':
            return value / 100.0

        # Housing
        elif indicator == 'CSUSHPISA':
            return value / 300.0

        # Default: use raw value
        else:
            return value

    def _get_from_previous_month(
        self,
        indicator: str,
        year: int,
        month: int,
        timestamp: pd.Timestamp,
    ) -> float:
        """
        Get indicator value from previous month if current month has no data.

        This handles cases where:
        - Indicator is quarterly (e.g., GDP)
        - Data for current month is not yet available
        """
        # Try previous month
        prev_month = month - 1
        prev_year = year

        if prev_month < 1:
            prev_month = 12
            prev_year = year - 1

        fred_df = self._load_month_fred(prev_year, prev_month)
        yfinance_df = self._load_month_yfinance(prev_year, prev_month)

        combined_df = pd.concat([fred_df, yfinance_df], ignore_index=True)
        indicator_data = combined_df[combined_df['indicator_name'] == indicator]

        if not indicator_data.empty:
            last_value = indicator_data.iloc[-1]['value']
            return self._normalize_indicator(indicator, last_value)

        # Still no data - return 0
        logger.warning(f"No data for indicator {indicator} at {timestamp}")
        return 0.0

    def clear_cache(self):
        """Clear all cached data."""
        self._fred_cache.clear()
        self._yfinance_cache.clear()
        self._preloaded_data = None
        self._timestamp_to_idx = None
        logger.info("Cleared macro data cache")

    def preload_all(
        self,
        start_date: str,
        end_date: str,
        freq: str = '5min',
    ) -> None:
        """
        Preload all macro data for a date range into numpy array.

        OPTIMIZED: Uses vectorized pandas operations instead of per-timestamp loops.
        Macro data (daily/weekly/monthly) is forward-filled to 5min frequency.

        Args:
            start_date: Start date (e.g., '2021-01-01')
            end_date: End date (e.g., '2022-12-31')
            freq: Timestamp frequency (default: '5min')

        After calling this, use get_features_fast() for O(1) lookup.
        """
        logger.info(f"Preloading macro data: {start_date} to {end_date}")
        total_start = time.time()

        # Step 1: Generate all timestamps
        logger.info("[1/4] Generating timestamp range...")
        timestamps = pd.date_range(start_date, end_date, freq=freq, tz='UTC')
        T = len(timestamps)
        logger.info(f"  {T:,} timestamps × {self.n_features} features")

        # Step 2: Create numpy array (T, n_features)
        logger.info("[2/4] Allocating numpy array...")
        self._preloaded_data = np.zeros((T, self.n_features), dtype=np.float32)
        self._timestamp_to_idx = {ts: i for i, ts in enumerate(timestamps)}

        # Step 3: Load all months needed
        logger.info("[3/4] Loading monthly parquet files...")
        months_to_load = set()
        for ts in timestamps:
            months_to_load.add((ts.year, ts.month))

        for year, month in tqdm(sorted(months_to_load), desc="Loading months", unit="month"):
            self._load_month_fred(year, month)
            self._load_month_yfinance(year, month)

        logger.info(f"  Loaded {len(months_to_load)} months of data")

        # Combine ALL cached data into single DataFrame (vectorized)
        all_dfs = []
        for (year, month), df in self._fred_cache.items():
            if not df.empty:
                all_dfs.append(df)
        for (year, month), df in self._yfinance_cache.items():
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            logger.warning("No macro data found, using zeros")
            return

        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"  Combined {len(combined_df):,} macro data points")

        # Step 4: Process each indicator ONCE with vectorized forward-fill
        logger.info("[4/4] Building feature matrix (vectorized forward-fill)...")
        ffill_start = time.time()

        for j, indicator in enumerate(tqdm(self.indicators, desc="Forward-fill indicators", unit="ind")):
            indicator_data = combined_df[combined_df['indicator_name'] == indicator].copy()

            if indicator_data.empty:
                continue

            # Sort by timestamp and remove duplicates (keep last)
            indicator_data = indicator_data.sort_values('timestamp')
            indicator_data = indicator_data.drop_duplicates(subset='timestamp', keep='last')

            # Normalize values
            indicator_data['normalized'] = indicator_data['value'].apply(
                lambda v: self._normalize_indicator(indicator, v)
            )

            # Create a Series with the indicator's native timestamps
            native_series = pd.Series(
                indicator_data['normalized'].values,
                index=indicator_data['timestamp']
            )

            # Reindex to target 5min timestamps with forward-fill
            # This is the KEY optimization: one reindex + ffill per indicator
            filled_series = native_series.reindex(timestamps, method='ffill')

            # Fill any remaining NaN at the start with 0
            filled_series = filled_series.fillna(0)

            # Store in numpy array
            self._preloaded_data[:, j] = filled_series.values

        ffill_elapsed = time.time() - ffill_start
        total_elapsed = time.time() - total_start
        memory_mb = self._preloaded_data.nbytes / 1e6

        logger.info(f"Macro preload complete!")
        logger.info(f"  Shape: {self._preloaded_data.shape}")
        logger.info(f"  Memory: {memory_mb:.1f} MB")
        logger.info(f"  Forward-fill time: {ffill_elapsed:.1f}s")
        logger.info(f"  Total time: {total_elapsed:.1f}s")

    def get_features_fast(self, timestamp: pd.Timestamp) -> np.ndarray:
        """
        Fast O(1) lookup of macro features using preloaded numpy array.

        Args:
            timestamp: Target timestamp

        Returns:
            np.ndarray of shape (n_features,)

        Raises:
            ValueError if preload_all() was not called first
        """
        if self._preloaded_data is None:
            raise ValueError("Must call preload_all() before get_features_fast()")

        # Ensure timestamp is timezone-aware
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')

        idx = self._timestamp_to_idx.get(timestamp)
        if idx is not None:
            return self._preloaded_data[idx]
        else:
            # Timestamp not in preloaded data, return zeros
            return np.zeros(self.n_features, dtype=np.float32)

    def is_preloaded(self) -> bool:
        """Check if numpy preload is ready."""
        return self._preloaded_data is not None

    def get_features_batch(self, timestamps: list) -> np.ndarray:
        """
        Fast O(1) batch lookup of macro features using preloaded numpy array.

        OPTIMIZED: Uses vectorized numpy indexing for T timestamps at once.

        Args:
            timestamps: List of timestamps (length T)

        Returns:
            np.ndarray of shape (T, n_features)

        Raises:
            ValueError if preload_all() was not called first
        """
        if self._preloaded_data is None:
            raise ValueError("Must call preload_all() before get_features_batch()")

        T = len(timestamps)
        result = np.zeros((T, self.n_features), dtype=np.float32)

        for i, ts in enumerate(timestamps):
            # Handle numpy.str_ and other string-like types
            if isinstance(ts, (np.str_, np.bytes_)):
                ts = str(ts)

            # Ensure timestamp is timezone-aware
            if hasattr(ts, 'tz') and ts.tz is None:
                ts = ts.tz_localize('UTC')
            elif isinstance(ts, str):
                ts = pd.Timestamp(ts)
                if ts.tz is None:
                    ts = ts.tz_localize('UTC')

            idx = self._timestamp_to_idx.get(ts)
            if idx is not None:
                result[i] = self._preloaded_data[idx]

        return result
