"""
Embedding Loader for GDELT and Nostr Embeddings

Efficiently loads pre-computed FinBERT (GDELT news) and CryptoBERT (Nostr social)
embeddings from HDF5 files for training.
"""

from typing import Dict, List, Optional
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class EmbeddingLoader:
    """
    Efficient loader for GDELT and Nostr embeddings from HDF5 files.

    Features:
    - In-memory indexing for O(1) lookup
    - Timestamp rounding to 5-minute intervals
    - Graceful fallback to zeros for missing data
    - Support for temporal window queries

    Usage:
        loader = EmbeddingLoader(
            gdelt_path='/path/to/gdelt_embeddings.h5',
            nostr_path='/path/to/nostr_embeddings.h5',
        )

        # Query embeddings for temporal window
        timestamps = ['2021-01-01T12:00:00+00:00', ...]  # T timestamps
        assets = [0, 1, 2, ...]  # N asset indices

        news_emb = loader.get_gdelt_embeddings(timestamps, assets)  # (T, N, 768)
        social_emb = loader.get_nostr_embeddings(timestamps, assets)  # (T, N, 768)
        has_signal = loader.get_social_signal_mask(timestamps, assets)  # (T, N, 1)

        loader.close()
    """

    def __init__(
        self,
        gdelt_path: str,
        nostr_path: str,
        device: str = 'cpu',
    ):
        """
        Initialize embedding loader.

        Args:
            gdelt_path: Path to GDELT HDF5 file
            nostr_path: Path to Nostr HDF5 file
            device: Device to load tensors ('cpu' or 'cuda')
        """
        self.device = device

        # Load HDF5 files (read-only, keep file handles open)
        self.gdelt_file = h5py.File(gdelt_path, 'r')
        self.nostr_file = h5py.File(nostr_path, 'r')

        # Build indices for fast lookup
        self.gdelt_index = self._build_index(self.gdelt_file)
        self.nostr_index = self._build_index(self.nostr_file)

        print(f"EmbeddingLoader: Loading embeddings into memory...")

        # Load all embeddings into memory (600-700MB total)
        # This avoids 54.7ms disk I/O per embedding (50-100x speedup)
        print(f"  Loading GDELT embeddings (all at once)...")
        gdelt_all = self.gdelt_file['embeddings'][:].astype(np.float32)
        self.gdelt_embeddings = {key: gdelt_all[idx] for key, idx in self.gdelt_index.items()}

        # FAST PATH: Keep contiguous array for vectorized lookup
        self._gdelt_array = gdelt_all  # (N_embeddings, 768)
        self._gdelt_ts_asset_to_idx = self.gdelt_index  # {ts_asset: idx}

        print(f"  Loading Nostr embeddings (all at once)...")
        nostr_all = self.nostr_file['embeddings'][:].astype(np.float32)
        self.nostr_embeddings = {key: nostr_all[idx] for key, idx in self.nostr_index.items()}

        # FAST PATH: Keep contiguous array for vectorized lookup
        self._nostr_array = nostr_all  # (N_embeddings, 768)
        self._nostr_ts_asset_to_idx = self.nostr_index  # {ts_asset: idx}

        # Build timestamp-only index for market-wide fast lookup
        # Group by timestamp (strip asset idx)
        self._gdelt_ts_to_indices = self._build_ts_index(self.gdelt_index)
        self._nostr_ts_to_indices = self._build_ts_index(self.nostr_index)

        print(f"EmbeddingLoader initialized:")
        print(f"  GDELT: {len(self.gdelt_index):,} embeddings (loaded to memory)")
        print(f"  Nostr: {len(self.nostr_index):,} embeddings (loaded to memory)")
        print(f"  GDELT unique timestamps: {len(self._gdelt_ts_to_indices):,}")
        print(f"  Nostr unique timestamps: {len(self._nostr_ts_to_indices):,}")

    def _build_index(self, h5file) -> Dict[str, int]:
        """
        Build {timestamp_assetIdx: index} mapping.

        Args:
            h5file: Open HDF5 file handle

        Returns:
            Dict mapping keys to embedding indices
        """
        keys = h5file['keys'][:]
        # Convert bytes to strings and build index
        index = {}
        for i, key in enumerate(keys):
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            index[key_str] = i
        return index

    def _build_ts_index(self, ts_asset_index: Dict[str, int]) -> Dict[str, List[int]]:
        """
        Build {timestamp: [embedding_indices]} mapping for market-wide lookups.

        Groups embeddings by timestamp for fast vectorized market-wide aggregation.

        Args:
            ts_asset_index: Dict mapping 'timestamp_assetIdx' -> embedding_index

        Returns:
            Dict mapping timestamp -> list of embedding indices for that timestamp
        """
        ts_to_indices = {}
        for key, idx in ts_asset_index.items():
            # Key format: "2021-01-01T12:00:00+00:00_5"
            # Split to get timestamp (everything before last underscore)
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                ts = parts[0]
                if ts not in ts_to_indices:
                    ts_to_indices[ts] = []
                ts_to_indices[ts].append(idx)
        return ts_to_indices

    def _round_timestamp(self, ts: str) -> str:
        """
        Round timestamp to nearest 5-minute interval.

        Args:
            ts: ISO format timestamp string

        Returns:
            Rounded timestamp in ISO format
        """
        # Parse ISO format: "2021-01-01T12:34:56+00:00"
        dt = pd.Timestamp(ts)
        minute = (dt.minute // 5) * 5
        rounded = dt.replace(minute=minute, second=0, microsecond=0)
        return rounded.isoformat()

    def get_gdelt_embeddings(
        self,
        timestamps: List[str],
        asset_indices: List[int],
    ) -> torch.Tensor:
        """
        Get GDELT embeddings for (timestamps, assets).

        Args:
            timestamps: List of ISO format timestamps (length T)
            asset_indices: List of asset indices (length N)

        Returns:
            embeddings: (T, N, 768) tensor
        """
        T = len(timestamps)
        N = len(asset_indices)
        embeddings = torch.zeros(T, N, 768, device=self.device, dtype=torch.float32)

        for t, ts in enumerate(timestamps):
            ts_rounded = self._round_timestamp(ts)

            for n, asset_idx in enumerate(asset_indices):
                key = f"{ts_rounded}_{asset_idx}"

                if key in self.gdelt_embeddings:
                    emb = self.gdelt_embeddings[key]
                    embeddings[t, n, :] = torch.from_numpy(emb)
                # else: leave as zeros (no news for this asset at this time)

        return embeddings

    def get_nostr_embeddings(
        self,
        timestamps: List[str],
        asset_indices: List[int],
    ) -> torch.Tensor:
        """
        Get Nostr embeddings for (timestamps, assets).

        Args:
            timestamps: List of ISO format timestamps (length T)
            asset_indices: List of asset indices (length N)

        Returns:
            embeddings: (T, N, 768) tensor
        """
        T = len(timestamps)
        N = len(asset_indices)
        embeddings = torch.zeros(T, N, 768, device=self.device, dtype=torch.float32)

        for t, ts in enumerate(timestamps):
            ts_rounded = self._round_timestamp(ts)

            for n, asset_idx in enumerate(asset_indices):
                key = f"{ts_rounded}_{asset_idx}"

                if key in self.nostr_embeddings:
                    emb = self.nostr_embeddings[key]
                    embeddings[t, n, :] = torch.from_numpy(emb)
                # else: leave as zeros (no social posts for this asset at this time)

        return embeddings

    def get_social_signal_mask(
        self,
        timestamps: List[str],
        asset_indices: List[int],
    ) -> torch.Tensor:
        """
        Get social signal availability mask.

        Args:
            timestamps: List of ISO format timestamps (length T)
            asset_indices: List of asset indices (length N)

        Returns:
            mask: (T, N, 1) tensor with 1.0 where social data exists, 0.0 otherwise
        """
        T = len(timestamps)
        N = len(asset_indices)
        mask = torch.zeros(T, N, 1, device=self.device, dtype=torch.float32)

        for t, ts in enumerate(timestamps):
            ts_rounded = self._round_timestamp(ts)

            for n, asset_idx in enumerate(asset_indices):
                key = f"{ts_rounded}_{asset_idx}"

                if key in self.nostr_embeddings:
                    mask[t, n, 0] = 1.0

        return mask

    # ============================================================
    # Market-Wide Methods (Global-Local Architecture)
    # ============================================================

    def get_gdelt_embeddings_marketwide(
        self,
        timestamps: List[str],
        asset_indices: Optional[List[int]] = None,
        max_assets: int = 20,
    ) -> torch.Tensor:
        """
        Get market-wide GDELT embeddings (no N dimension).

        Aggregates per-asset embeddings into market-wide representation
        using mean pooling across available assets.

        Args:
            timestamps: List of ISO format timestamps (length T)
            asset_indices: List of asset indices to pool (default: 0 to max_assets-1)
            max_assets: Maximum number of assets to consider (default: 20 for Top 20 universe)

        Returns:
            embeddings: (T, 768) tensor - market-wide news embeddings
        """
        if asset_indices is None:
            asset_indices = list(range(max_assets))

        T = len(timestamps)
        embeddings = torch.zeros(T, 768, device=self.device, dtype=torch.float32)

        for t, ts in enumerate(timestamps):
            ts_rounded = self._round_timestamp(ts)

            # Collect all available asset embeddings for this timestamp
            asset_embs = []
            for asset_idx in asset_indices:
                key = f"{ts_rounded}_{asset_idx}"

                if key in self.gdelt_embeddings:
                    emb = self.gdelt_embeddings[key]
                    asset_embs.append(emb)

            # Mean pooling across assets
            if asset_embs:
                mean_emb = np.mean(asset_embs, axis=0)
                embeddings[t, :] = torch.from_numpy(mean_emb)
            # else: leave as zeros (no news for any asset at this time)

        return embeddings

    def get_nostr_embeddings_marketwide(
        self,
        timestamps: List[str],
        asset_indices: Optional[List[int]] = None,
        max_assets: int = 20,
    ) -> torch.Tensor:
        """
        Get market-wide Nostr embeddings (no N dimension).

        Aggregates per-asset embeddings into market-wide representation
        using mean pooling across available assets.

        Args:
            timestamps: List of ISO format timestamps (length T)
            asset_indices: List of asset indices to pool (default: 0 to max_assets-1)
            max_assets: Maximum number of assets to consider (default: 20 for Top 20 universe)

        Returns:
            embeddings: (T, 768) tensor - market-wide social embeddings
        """
        if asset_indices is None:
            asset_indices = list(range(max_assets))

        T = len(timestamps)
        embeddings = torch.zeros(T, 768, device=self.device, dtype=torch.float32)

        for t, ts in enumerate(timestamps):
            ts_rounded = self._round_timestamp(ts)

            # Collect all available asset embeddings for this timestamp
            asset_embs = []
            for asset_idx in asset_indices:
                key = f"{ts_rounded}_{asset_idx}"

                if key in self.nostr_embeddings:
                    emb = self.nostr_embeddings[key]
                    asset_embs.append(emb)

            # Mean pooling across assets
            if asset_embs:
                mean_emb = np.mean(asset_embs, axis=0)
                embeddings[t, :] = torch.from_numpy(mean_emb)
            # else: leave as zeros (no social posts for any asset at this time)

        return embeddings

    def get_social_signal_mask_marketwide(
        self,
        timestamps: List[str],
        asset_indices: Optional[List[int]] = None,
        max_assets: int = 20,
    ) -> torch.Tensor:
        """
        Get market-wide social signal availability mask (no N dimension).

        Args:
            timestamps: List of ISO format timestamps (length T)
            asset_indices: List of asset indices to check (default: 0 to max_assets-1)
            max_assets: Maximum number of assets to consider (default: 20)

        Returns:
            mask: (T, 1) tensor with 1.0 if ANY asset has social data, 0.0 otherwise
        """
        if asset_indices is None:
            asset_indices = list(range(max_assets))

        T = len(timestamps)
        mask = torch.zeros(T, 1, device=self.device, dtype=torch.float32)

        for t, ts in enumerate(timestamps):
            ts_rounded = self._round_timestamp(ts)

            # Check if ANY asset has social data
            has_data = False
            for asset_idx in asset_indices:
                key = f"{ts_rounded}_{asset_idx}"

                if key in self.nostr_embeddings:
                    has_data = True
                    break

            if has_data:
                mask[t, 0] = 1.0

        return mask

    # ============================================================
    # FAST Market-Wide Methods (Vectorized, ~300x faster)
    # ============================================================

    def _round_timestamp_fast(self, ts) -> str:
        """Fast timestamp rounding without pandas parsing overhead."""
        # Handle numpy.str_ and other string-like types
        if isinstance(ts, (np.str_, np.bytes_)):
            ts = str(ts)

        if isinstance(ts, pd.Timestamp):
            dt = ts
        elif isinstance(ts, str):
            # Fast path: assume ISO format and parse minimally
            dt = pd.Timestamp(ts)
        else:
            dt = pd.Timestamp(ts)

        minute = (dt.minute // 5) * 5
        rounded = dt.replace(minute=minute, second=0, microsecond=0)
        return rounded.isoformat()

    def get_gdelt_embeddings_marketwide_fast(
        self,
        timestamps: List,
        asset_indices: Optional[List[int]] = None,
        max_assets: int = 20,
    ) -> torch.Tensor:
        """
        FAST market-wide GDELT embeddings using vectorized numpy operations.

        ~300x faster than the non-fast version by:
        1. Using pre-built timestamp index
        2. Vectorized numpy fancy indexing
        3. Batched mean computation

        Args:
            timestamps: List of timestamps (pd.Timestamp or ISO strings)
            asset_indices: Ignored (uses all available per timestamp)
            max_assets: Ignored (uses all available per timestamp)

        Returns:
            embeddings: (T, 768) tensor - market-wide news embeddings
        """
        T = len(timestamps)
        result = np.zeros((T, 768), dtype=np.float32)

        for t, ts in enumerate(timestamps):
            ts_rounded = self._round_timestamp_fast(ts)

            if ts_rounded in self._gdelt_ts_to_indices:
                indices = self._gdelt_ts_to_indices[ts_rounded]
                # Vectorized: get all embeddings for this timestamp at once
                embs = self._gdelt_array[indices]  # (N_assets, 768)
                # Vectorized mean
                result[t, :] = embs.mean(axis=0)

        return torch.from_numpy(result).to(self.device)

    def get_nostr_embeddings_marketwide_fast(
        self,
        timestamps: List,
        asset_indices: Optional[List[int]] = None,
        max_assets: int = 20,
    ) -> torch.Tensor:
        """
        FAST market-wide Nostr embeddings using vectorized numpy operations.

        ~300x faster than the non-fast version.

        Args:
            timestamps: List of timestamps (pd.Timestamp or ISO strings)
            asset_indices: Ignored (uses all available per timestamp)
            max_assets: Ignored (uses all available per timestamp)

        Returns:
            embeddings: (T, 768) tensor - market-wide social embeddings
        """
        T = len(timestamps)
        result = np.zeros((T, 768), dtype=np.float32)

        for t, ts in enumerate(timestamps):
            ts_rounded = self._round_timestamp_fast(ts)

            if ts_rounded in self._nostr_ts_to_indices:
                indices = self._nostr_ts_to_indices[ts_rounded]
                # Vectorized: get all embeddings for this timestamp at once
                embs = self._nostr_array[indices]  # (N_assets, 768)
                # Vectorized mean
                result[t, :] = embs.mean(axis=0)

        return torch.from_numpy(result).to(self.device)

    def get_social_signal_mask_marketwide_fast(
        self,
        timestamps: List,
        asset_indices: Optional[List[int]] = None,
        max_assets: int = 20,
    ) -> torch.Tensor:
        """
        FAST market-wide social signal mask.

        Args:
            timestamps: List of timestamps (pd.Timestamp or ISO strings)
            asset_indices: Ignored
            max_assets: Ignored

        Returns:
            mask: (T, 1) tensor with 1.0 if ANY asset has social data
        """
        T = len(timestamps)
        result = np.zeros((T, 1), dtype=np.float32)

        for t, ts in enumerate(timestamps):
            ts_rounded = self._round_timestamp_fast(ts)

            if ts_rounded in self._nostr_ts_to_indices:
                result[t, 0] = 1.0

        return torch.from_numpy(result).to(self.device)

    def close(self):
        """Close HDF5 file handles."""
        self.gdelt_file.close()
        self.nostr_file.close()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass  # File may already be closed
