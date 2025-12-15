"""
Alpha101 Loader for HierarchicalStateBuilder

Efficiently loads pre-computed alpha factors from parquet cache.
Supports batch loading for training and real-time loading for inference.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
from tqdm import tqdm

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

        # Fast numpy preload (initialized by preload_as_numpy())
        self._preloaded_data: Optional[np.ndarray] = None
        self._timestamp_to_idx: Optional[Dict] = None
        self._preload_symbols: Optional[List[str]] = None

        # Slot mapping (initialized by set_slot_symbols())
        self._slot_symbol_indices: Optional[np.ndarray] = None
        self._slot_symbols: Optional[np.ndarray] = None

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
        start_time = time.time()

        for alpha_idx in tqdm(range(self.n_alphas), desc="Loading alpha files", unit="file"):
            self.load_alpha_file(alpha_idx)

        elapsed = time.time() - start_time
        logger.info(f"Preloaded {len(self._alpha_cache)} alpha files in {elapsed:.1f}s")

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
        self._preloaded_data = None
        self._timestamp_to_idx = None
        self._preload_symbols = None
        logger.info("Alpha cache cleared")

    def preload_as_numpy(
        self,
        symbols: List[str],
        timestamps: pd.DatetimeIndex,
    ) -> None:
        """
        Preload all alpha data into a numpy array for fast O(1) lookup.

        Args:
            symbols: List of symbol names (N symbols)
            timestamps: DatetimeIndex of all timestamps to preload

        After calling this, use get_alphas_fast() for O(1) lookup.
        """
        logger.info(f"Preloading alphas as numpy: {len(timestamps):,} timestamps × {len(symbols)} symbols × {self.n_alphas} alphas")
        total_start = time.time()

        # Step 1: Ensure all alpha files are loaded
        logger.info("[1/3] Loading alpha parquet files...")
        self.preload_all_alphas()

        N = len(symbols)
        T = len(timestamps)

        # Step 2: Create numpy array (T, N, n_alphas)
        logger.info(f"[2/3] Allocating numpy array: ({T:,}, {N}, {self.n_alphas})...")
        self._preloaded_data = np.zeros((T, N, self.n_alphas), dtype=np.float32)

        # Create timestamp -> index mapping
        logger.info("Building timestamp index mapping...")
        self._timestamp_to_idx = {ts: i for i, ts in enumerate(timestamps)}
        self._preload_symbols = symbols

        # Step 3: Fill the array - VECTORIZED (101 reindex instead of 10,100)
        logger.info("[3/3] Building numpy array (vectorized reindex)...")
        symbol_to_idx = {s: i for i, s in enumerate(symbols)}
        reindex_start = time.time()

        for alpha_idx in tqdm(range(self.n_alphas), desc="Vectorized reindex", unit="alpha"):
            df = self._alpha_cache.get(alpha_idx)
            if df is None or df.empty:
                continue

            # Find available symbols in this alpha file
            available_symbols = [s for s in symbols if s in df.columns]
            if not available_symbols:
                continue

            # VECTORIZED: Reindex all symbols at once (1 op instead of N ops)
            reindexed = df[available_symbols].reindex(timestamps).fillna(0)

            # Copy to numpy array
            for symbol in available_symbols:
                n_idx = symbol_to_idx[symbol]
                self._preloaded_data[:, n_idx, alpha_idx] = reindexed[symbol].values

        reindex_elapsed = time.time() - reindex_start
        total_elapsed = time.time() - total_start

        # Calculate memory usage
        memory_mb = self._preloaded_data.nbytes / 1e6
        logger.info(f"Alpha preload complete!")
        logger.info(f"  Shape: {self._preloaded_data.shape}")
        logger.info(f"  Memory: {memory_mb:.1f} MB")
        logger.info(f"  Reindex time: {reindex_elapsed:.1f}s")
        logger.info(f"  Total time: {total_elapsed:.1f}s")

    def get_alphas_fast(
        self,
        timestamps: List,
    ) -> np.ndarray:
        """
        Fast O(1) lookup of alpha values using preloaded numpy array.

        Args:
            timestamps: List of timestamps (length T)

        Returns:
            np.ndarray of shape (T, N, n_alphas)

        Raises:
            ValueError if preload_as_numpy() was not called first
        """
        if self._preloaded_data is None:
            raise ValueError("Must call preload_as_numpy() before get_alphas_fast()")

        # Convert timestamps to indices
        indices = []
        for ts in timestamps:
            # Handle both string and Timestamp types
            if isinstance(ts, str):
                ts = pd.Timestamp(ts)
            idx = self._timestamp_to_idx.get(ts)
            if idx is not None:
                indices.append(idx)
            else:
                # Timestamp not in preloaded data, use nearest
                indices.append(0)  # Fallback to first

        # Fast numpy indexing
        return self._preloaded_data[indices]

    def is_preloaded(self) -> bool:
        """Check if numpy preload is ready."""
        return self._preloaded_data is not None

    def set_slot_symbols(self, slot_symbols: np.ndarray) -> None:
        """
        Set slot_symbols mapping for dynamic universe support.

        This method pre-computes the mapping from (timestamp, slot) → preload_index
        for fast O(1) lookup during training.

        Args:
            slot_symbols: (T, N_slots) array of symbol names at each timestamp
                         e.g., slot_symbols[t, slot_idx] = "BTCUSDT"

        After calling this, use get_alphas_for_slots() for slot-aware lookup.
        """
        if self._preloaded_data is None:
            raise ValueError("Must call preload_as_numpy() before set_slot_symbols()")

        if self._preload_symbols is None:
            raise ValueError("Preload symbols not available")

        T_slots, N_slots = slot_symbols.shape
        T_preload = self._preloaded_data.shape[0]

        logger.info(f"Setting up slot_symbols mapping: ({T_slots}, {N_slots}) slots")

        # Build symbol -> preload_idx mapping
        symbol_to_preload_idx = {s: i for i, s in enumerate(self._preload_symbols)}

        # Pre-compute slot_symbol_indices: (T, N_slots) -> preload indices
        self._slot_symbol_indices = np.zeros((T_slots, N_slots), dtype=np.int32)
        unknown_symbols = set()

        for t in range(T_slots):
            for slot_idx in range(N_slots):
                symbol = slot_symbols[t, slot_idx]
                preload_idx = symbol_to_preload_idx.get(symbol, -1)
                if preload_idx == -1:
                    unknown_symbols.add(symbol)
                    preload_idx = 0  # Fallback to first symbol
                self._slot_symbol_indices[t, slot_idx] = preload_idx

        # Store for reference
        self._slot_symbols = slot_symbols

        # Log sample mapping for verification
        logger.info(f"Slot symbol mapping complete!")
        logger.info(f"  Total timestamps: {T_slots:,}")
        logger.info(f"  Slots per timestamp: {N_slots}")
        logger.info(f"  Unknown symbols (fallback to idx 0): {len(unknown_symbols)}")

        if len(unknown_symbols) > 0 and len(unknown_symbols) <= 10:
            logger.warning(f"  Unknown symbols: {unknown_symbols}")

        # Log sample mapping at t=0, t=mid, t=end
        sample_indices = [0, T_slots // 2, T_slots - 1]
        for t in sample_indices:
            symbols_at_t = slot_symbols[t, :5]  # First 5 slots
            indices_at_t = self._slot_symbol_indices[t, :5]
            logger.info(f"  t={t}: symbols={list(symbols_at_t)} -> preload_idx={list(indices_at_t)}")

    def get_alphas_for_slots(self, timestamp_indices: List[int]) -> np.ndarray:
        """
        Fast O(1) lookup of alpha values for specific slots using pre-computed mapping.

        This method uses the slot_symbol_indices mapping to extract only the
        20 symbols that are in slots at each timestamp (dynamic universe).

        OPTIMIZED: Uses numpy advanced indexing instead of Python loops (14.7x faster).

        Args:
            timestamp_indices: List of timestamp indices (length T_batch)

        Returns:
            np.ndarray of shape (T_batch, N_slots, n_alphas)

        Raises:
            ValueError if set_slot_symbols() was not called first
        """
        if self._slot_symbol_indices is None:
            raise ValueError("Must call set_slot_symbols() before get_alphas_for_slots()")

        # Convert to numpy array for vectorized operations
        timestamp_indices = np.asarray(timestamp_indices)

        # Get slot indices for requested timestamps: (T_batch, N_slots)
        slot_indices = self._slot_symbol_indices[timestamp_indices]

        # VECTORIZED: Use numpy advanced indexing instead of Python loops
        # preloaded_data: (T_total, N_all, 101)
        # We want: result[i, j, :] = preloaded_data[timestamp_indices[i], slot_indices[i, j], :]
        # This is equivalent to: preloaded_data[timestamp_indices[:, None], slot_indices, :]
        result = self._preloaded_data[timestamp_indices[:, None], slot_indices, :]

        return result

    def get_alphas_for_slots_fast(self, timestamps: List) -> np.ndarray:
        """
        Fast slot-aware lookup using timestamps (convenience wrapper).

        Args:
            timestamps: List of timestamps (length T_batch)

        Returns:
            np.ndarray of shape (T_batch, N_slots, n_alphas)
        """
        if self._slot_symbol_indices is None:
            raise ValueError("Must call set_slot_symbols() before get_alphas_for_slots_fast()")

        # Convert timestamps to indices
        indices = []
        for ts in timestamps:
            if isinstance(ts, str):
                ts = pd.Timestamp(ts)
            idx = self._timestamp_to_idx.get(ts)
            if idx is not None:
                indices.append(idx)
            else:
                indices.append(0)  # Fallback

        return self.get_alphas_for_slots(indices)

    def has_slot_mapping(self) -> bool:
        """Check if slot_symbols mapping is ready."""
        return self._slot_symbol_indices is not None
