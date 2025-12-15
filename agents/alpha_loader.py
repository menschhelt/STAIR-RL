"""
Alpha101 Loader for HierarchicalStateBuilder

Efficiently loads pre-computed alpha factors from parquet cache.
Supports batch loading for training and real-time loading for inference.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)


class AlphaLoader:
    """
    Loads Alpha101 factors from parquet cache.

    Alpha cache structure:
        alpha_cache/
            alpha_101_000.parquet  # Alpha 0: (timestamp, symbol) -> value
            alpha_101_001.parquet  # Alpha 1
            ...
            alpha_101_100.parquet  # Alpha 100

    Each parquet file:
        - Index: timestamp (datetime)
        - Columns: symbol names (BTCUSDT, ETHUSDT, ...)
        - Values: float32 alpha values
    """

    def __init__(
        self,
        alpha_cache_dir: Optional[Path] = None,
        n_alphas: int = 101,
        device: str = 'cpu',
    ):
        """
        Initialize alpha loader.

        Args:
            alpha_cache_dir: Path to alpha cache directory
            n_alphas: Number of alphas to load (default: 101)
            device: Device for tensors
        """
        if alpha_cache_dir is None:
            self.alpha_cache_dir = DATA_DIR / 'features' / 'alpha_cache'
        else:
            self.alpha_cache_dir = Path(alpha_cache_dir)

        self.n_alphas = n_alphas
        self.device = device

        # Cache for loaded alphas (alpha_idx -> DataFrame)
        self._alpha_cache: Dict[int, pd.DataFrame] = {}

        # Verify cache directory exists
        if not self.alpha_cache_dir.exists():
            logger.warning(f"Alpha cache directory not found: {self.alpha_cache_dir}")
        else:
            # Count available alpha files
            alpha_files = list(self.alpha_cache_dir.glob('alpha_101_*.parquet'))
            logger.info(f"AlphaLoader initialized: {len(alpha_files)} alpha files found")

    def load_alpha_file(self, alpha_idx: int) -> pd.DataFrame:
        """
        Load a single alpha file into cache.

        Args:
            alpha_idx: Alpha index (0-100)

        Returns:
            DataFrame with timestamp index and symbol columns
        """
        if alpha_idx in self._alpha_cache:
            return self._alpha_cache[alpha_idx]

        alpha_file = self.alpha_cache_dir / f'alpha_101_{alpha_idx:03d}.parquet'

        if not alpha_file.exists():
            logger.warning(f"Alpha file not found: {alpha_file}")
            return pd.DataFrame()

        try:
            df = pq.read_table(alpha_file).to_pandas()
            self._alpha_cache[alpha_idx] = df
            return df
        except Exception as e:
            logger.error(f"Failed to load alpha {alpha_idx}: {e}")
            return pd.DataFrame()

    def preload_all_alphas(self) -> None:
        """Preload all alpha files into memory for faster access."""
        logger.info(f"Preloading {self.n_alphas} alpha files...")

        for alpha_idx in range(self.n_alphas):
            self.load_alpha_file(alpha_idx)

        logger.info(f"Preloaded {len(self._alpha_cache)} alpha files")

    def get_alphas_for_symbols(
        self,
        symbols: List[str],
        timestamps: List[datetime],
    ) -> np.ndarray:
        """
        Get alpha values for specified symbols and timestamps.

        Args:
            symbols: List of symbol names (length N)
            timestamps: List of timestamps (length T)

        Returns:
            np.ndarray of shape (T, N, n_alphas) with alpha values
        """
        T = len(timestamps)
        N = len(symbols)

        # Initialize output array
        alphas = np.zeros((T, N, self.n_alphas), dtype=np.float32)

        # Convert timestamps to pandas Timestamps for indexing
        ts_index = pd.to_datetime(timestamps)

        for alpha_idx in range(self.n_alphas):
            df = self.load_alpha_file(alpha_idx)

            if df.empty:
                continue

            # Get values for each symbol
            for n_idx, symbol in enumerate(symbols):
                if symbol not in df.columns:
                    continue

                # Get values at specified timestamps
                try:
                    # Use reindex to handle missing timestamps gracefully
                    values = df[symbol].reindex(ts_index)
                    alphas[:, n_idx, alpha_idx] = values.fillna(0).values
                except Exception as e:
                    logger.debug(f"Failed to get alpha {alpha_idx} for {symbol}: {e}")
                    continue

        return alphas

    def get_alphas_batch(
        self,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> Tuple[np.ndarray, List[datetime]]:
        """
        Get all alpha values for a time range (for batch training).

        Args:
            symbols: List of symbol names (length N)
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Tuple of:
                - np.ndarray of shape (T, N, n_alphas)
                - List of timestamps
        """
        # Load first alpha to get timestamp range
        df0 = self.load_alpha_file(0)

        if df0.empty:
            logger.warning("No alpha data available")
            return np.array([]), []

        # Filter timestamps
        mask = (df0.index >= start_time) & (df0.index <= end_time)
        timestamps = df0.index[mask].tolist()

        if not timestamps:
            logger.warning(f"No timestamps in range {start_time} to {end_time}")
            return np.array([]), []

        T = len(timestamps)
        N = len(symbols)

        logger.info(f"Loading alphas: T={T}, N={N}, alphas={self.n_alphas}")

        # Initialize output array
        alphas = np.zeros((T, N, self.n_alphas), dtype=np.float32)

        # Load each alpha
        for alpha_idx in range(self.n_alphas):
            df = self.load_alpha_file(alpha_idx)

            if df.empty:
                continue

            # Filter to time range
            df_filtered = df.loc[mask]

            # Get values for each symbol
            for n_idx, symbol in enumerate(symbols):
                if symbol not in df_filtered.columns:
                    continue

                values = df_filtered[symbol].values
                alphas[:, n_idx, alpha_idx] = np.nan_to_num(values, nan=0.0)

        logger.info(f"Loaded alphas: shape={alphas.shape}, non-zero={np.count_nonzero(alphas)}")

        return alphas, timestamps

    def get_alpha_stats(self) -> Dict[str, any]:
        """Get statistics about loaded alphas."""
        if not self._alpha_cache:
            return {"loaded": 0, "total": self.n_alphas}

        stats = {
            "loaded": len(self._alpha_cache),
            "total": self.n_alphas,
            "memory_mb": sum(
                df.memory_usage(deep=True).sum() / 1e6
                for df in self._alpha_cache.values()
            ),
        }

        # Sample one alpha for shape info
        if self._alpha_cache:
            sample_df = next(iter(self._alpha_cache.values()))
            stats["timestamps"] = len(sample_df)
            stats["symbols"] = len(sample_df.columns)

        return stats

    def clear_cache(self) -> None:
        """Clear loaded alpha cache to free memory."""
        self._alpha_cache.clear()
        logger.info("Alpha cache cleared")
