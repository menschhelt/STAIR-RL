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

        print(f"EmbeddingLoader initialized:")
        print(f"  GDELT: {len(self.gdelt_index):,} embeddings")
        print(f"  Nostr: {len(self.nostr_index):,} embeddings")

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

                if key in self.gdelt_index:
                    idx = self.gdelt_index[key]
                    emb = self.gdelt_file['embeddings'][idx]
                    embeddings[t, n, :] = torch.from_numpy(emb.astype(np.float32))
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

                if key in self.nostr_index:
                    idx = self.nostr_index[key]
                    emb = self.nostr_file['embeddings'][idx]
                    embeddings[t, n, :] = torch.from_numpy(emb.astype(np.float32))
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

                if key in self.nostr_index:
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

                if key in self.gdelt_index:
                    idx = self.gdelt_index[key]
                    emb = self.gdelt_file['embeddings'][idx]
                    asset_embs.append(emb.astype(np.float32))

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

                if key in self.nostr_index:
                    idx = self.nostr_index[key]
                    emb = self.nostr_file['embeddings'][idx]
                    asset_embs.append(emb.astype(np.float32))

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

                if key in self.nostr_index:
                    has_data = True
                    break

            if has_data:
                mask[t, 0] = 1.0

        return mask

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
