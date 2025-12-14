"""
Text Embedding Utilities for STAIR-RL

This module provides utilities for:
1. Pre-computing BERT embeddings from text processors
2. Loading and managing pre-computed embeddings during training
3. Efficient batch retrieval of embeddings

Usage:
    # Pre-compute embeddings (one-time offline process)
    precomputer = EmbeddingPrecomputer(processor, output_dir)
    precomputer.compute_and_save(df, timestamps, assets)

    # Load embeddings during training
    loader = PrecomputedTextEmbedding(embedding_path)
    embeddings, has_signal = loader.get_batch_embedding(timestamps, n_assets=20)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import json
from datetime import datetime
import h5py

logger = logging.getLogger(__name__)


class PrecomputedTextEmbedding:
    """
    Load and manage pre-computed BERT embeddings.

    Embeddings are stored per (timestamp, asset) pair.
    FinBERT/CryptoBERT are frozen during RL training,
    so embeddings are pre-computed to save GPU memory and time.

    Storage format: HDF5 for efficient I/O
    - embeddings: (N, 768) float16 array
    - keys: list of "timestamp_assetIdx" strings
    - metadata: model info, creation date, etc.
    """

    EMBEDDING_DIM = 768  # BERT hidden size

    def __init__(
        self,
        embedding_path: Union[str, Path],
        device: str = 'cpu',
        cache_size: int = 10000,
    ):
        """
        Initialize embedding loader.

        Args:
            embedding_path: Path to HDF5 embedding file
            device: Device for tensors ('cpu' or 'cuda')
            cache_size: Number of embeddings to cache in memory
        """
        self.embedding_path = Path(embedding_path)
        self.device = device
        self.cache_size = cache_size

        # Load embeddings
        self._embeddings: Dict[str, torch.Tensor] = {}
        self._load_embeddings()

        logger.info(f"Loaded {len(self._embeddings)} embeddings from {embedding_path}")

    def _load_embeddings(self):
        """Load embeddings from HDF5 file."""
        if not self.embedding_path.exists():
            logger.warning(f"Embedding file not found: {self.embedding_path}")
            return

        with h5py.File(self.embedding_path, 'r') as f:
            # Load keys
            keys = [k.decode('utf-8') for k in f['keys'][:]]

            # Load embeddings as float32
            embeddings = f['embeddings'][:].astype(np.float32)

            # Store in dictionary
            for key, emb in zip(keys, embeddings):
                self._embeddings[key] = torch.tensor(emb, device=self.device)

    def get_embedding(
        self,
        timestamp: str,
        asset_idx: int,
    ) -> torch.Tensor:
        """
        Get embedding for a specific timestamp and asset.

        Args:
            timestamp: Timestamp string (format: "2024-01-15T10:30:00")
            asset_idx: Asset index (0-19)

        Returns:
            Embedding tensor (768,) or zeros if not found
        """
        key = f"{timestamp}_{asset_idx}"

        if key in self._embeddings:
            return self._embeddings[key]
        else:
            # No embedding available - return zeros
            return torch.zeros(self.EMBEDDING_DIM, device=self.device)

    def get_batch_embedding(
        self,
        timestamps: List[str],
        n_assets: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get batch of embeddings for multiple timestamps.

        Args:
            timestamps: List of timestamp strings
            n_assets: Number of assets

        Returns:
            embeddings: (T, n_assets, 768) tensor
            has_signal: (T, n_assets, 1) binary tensor indicating data availability
        """
        T = len(timestamps)
        embeddings = torch.zeros(T, n_assets, self.EMBEDDING_DIM, device=self.device)
        has_signal = torch.zeros(T, n_assets, 1, device=self.device)

        for t, ts in enumerate(timestamps):
            for a in range(n_assets):
                emb = self.get_embedding(ts, a)
                embeddings[t, a] = emb

                # Check if embedding exists (non-zero)
                if emb.abs().sum() > 0:
                    has_signal[t, a] = 1.0

        return embeddings, has_signal

    def get_lookback_embedding(
        self,
        end_timestamp: str,
        timestamps_list: List[str],
        lookback: int,
        n_assets: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get embeddings for a lookback window ending at end_timestamp.

        Args:
            end_timestamp: End timestamp
            timestamps_list: Ordered list of all timestamps
            lookback: Number of timesteps to look back
            n_assets: Number of assets

        Returns:
            embeddings: (lookback, n_assets, 768)
            has_signal: (lookback, n_assets, 1)
        """
        # Find index of end_timestamp
        try:
            end_idx = timestamps_list.index(end_timestamp)
        except ValueError:
            logger.warning(f"Timestamp not found: {end_timestamp}")
            return (
                torch.zeros(lookback, n_assets, self.EMBEDDING_DIM, device=self.device),
                torch.zeros(lookback, n_assets, 1, device=self.device),
            )

        # Get lookback timestamps
        start_idx = max(0, end_idx - lookback + 1)
        lookback_timestamps = timestamps_list[start_idx:end_idx + 1]

        # Pad if needed
        if len(lookback_timestamps) < lookback:
            padding = lookback - len(lookback_timestamps)
            lookback_timestamps = [lookback_timestamps[0]] * padding + lookback_timestamps

        return self.get_batch_embedding(lookback_timestamps, n_assets)

    def __len__(self) -> int:
        """Return number of stored embeddings."""
        return len(self._embeddings)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._embeddings


class EmbeddingPrecomputer:
    """
    Pre-compute embeddings from text using BERT models.

    Use this class to pre-compute embeddings before training.
    This is a one-time offline process.
    """

    def __init__(
        self,
        processor,  # FinBERTProcessor or CryptoBERTProcessor
        output_dir: Union[str, Path],
        batch_size: int = 256,
    ):
        """
        Initialize pre-computer.

        Args:
            processor: Text processor instance (FinBERT or CryptoBERT)
            output_dir: Directory to save embeddings
            batch_size: Batch size for processing
        """
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

    def compute_and_save(
        self,
        texts_by_key: Dict[str, str],  # {timestamp_assetIdx: text}
        output_name: str = "embeddings.h5",
        show_progress: bool = True,
    ):
        """
        Compute embeddings and save to HDF5.

        Args:
            texts_by_key: Dictionary mapping keys to texts
            output_name: Output filename
            show_progress: Show progress bar
        """
        from tqdm import tqdm

        keys = list(texts_by_key.keys())
        texts = [texts_by_key[k] for k in keys]

        # Process in batches
        all_embeddings = []

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing embeddings")

        for i in iterator:
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.processor.get_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)

        # Concatenate
        embeddings = np.concatenate(all_embeddings, axis=0)

        # Save to HDF5
        output_path = self.output_dir / output_name
        with h5py.File(output_path, 'w') as f:
            # Store embeddings as float16 to save space
            f.create_dataset(
                'embeddings',
                data=embeddings.astype(np.float16),
                compression='gzip',
                compression_opts=9,
            )

            # Store keys
            f.create_dataset(
                'keys',
                data=np.array(keys, dtype='S50'),  # Max 50 chars per key
            )

            # Store metadata
            f.attrs['model_name'] = self.processor.MODEL_NAME
            f.attrs['embedding_dim'] = 768
            f.attrs['num_embeddings'] = len(keys)
            f.attrs['created_at'] = datetime.now().isoformat()

        logger.info(f"Saved {len(keys)} embeddings to {output_path}")


class MultiSourceEmbeddingLoader:
    """
    Load embeddings from multiple sources (news + social).

    Combines FinBERT (GDELT news) and CryptoBERT (Nostr social) embeddings.
    """

    def __init__(
        self,
        news_embedding_path: Optional[Union[str, Path]] = None,
        social_embedding_path: Optional[Union[str, Path]] = None,
        device: str = 'cpu',
    ):
        """
        Initialize multi-source loader.

        Args:
            news_embedding_path: Path to news embeddings (FinBERT)
            social_embedding_path: Path to social embeddings (CryptoBERT)
            device: Device for tensors
        """
        self.device = device

        # Load news embeddings
        self.news_loader = None
        if news_embedding_path and Path(news_embedding_path).exists():
            self.news_loader = PrecomputedTextEmbedding(
                news_embedding_path, device=device
            )
            logger.info(f"Loaded news embeddings: {len(self.news_loader)} entries")

        # Load social embeddings
        self.social_loader = None
        if social_embedding_path and Path(social_embedding_path).exists():
            self.social_loader = PrecomputedTextEmbedding(
                social_embedding_path, device=device
            )
            logger.info(f"Loaded social embeddings: {len(self.social_loader)} entries")

    def get_all_embeddings(
        self,
        timestamps: List[str],
        n_assets: int = 20,
    ) -> Dict[str, torch.Tensor]:
        """
        Get all embeddings for timestamps.

        Args:
            timestamps: List of timestamp strings
            n_assets: Number of assets

        Returns:
            Dictionary with:
                - 'news_embedding': (T, n_assets, 768)
                - 'social_embedding': (T, n_assets, 768)
                - 'has_news_signal': (T, n_assets, 1)
                - 'has_social_signal': (T, n_assets, 1)
        """
        T = len(timestamps)
        embedding_dim = PrecomputedTextEmbedding.EMBEDDING_DIM

        # Initialize outputs
        result = {
            'news_embedding': torch.zeros(T, n_assets, embedding_dim, device=self.device),
            'social_embedding': torch.zeros(T, n_assets, embedding_dim, device=self.device),
            'has_news_signal': torch.zeros(T, n_assets, 1, device=self.device),
            'has_social_signal': torch.zeros(T, n_assets, 1, device=self.device),
        }

        # Load news embeddings
        if self.news_loader:
            news_emb, has_news = self.news_loader.get_batch_embedding(timestamps, n_assets)
            result['news_embedding'] = news_emb
            result['has_news_signal'] = has_news

        # Load social embeddings
        if self.social_loader:
            social_emb, has_social = self.social_loader.get_batch_embedding(timestamps, n_assets)
            result['social_embedding'] = social_emb
            result['has_social_signal'] = has_social

        return result


# ========== Standalone Testing ==========

if __name__ == '__main__':
    import tempfile

    # Test PrecomputedTextEmbedding
    print("Testing PrecomputedTextEmbedding...")

    # Create dummy embeddings
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = f.name

    # Create test data
    keys = ['2024-01-15T10:30:00_0', '2024-01-15T10:30:00_1', '2024-01-15T10:35:00_0']
    embeddings = np.random.randn(3, 768).astype(np.float16)

    # Save to HDF5
    with h5py.File(temp_path, 'w') as f:
        f.create_dataset('embeddings', data=embeddings)
        f.create_dataset('keys', data=np.array(keys, dtype='S50'))

    # Load and test
    loader = PrecomputedTextEmbedding(temp_path)
    print(f"Loaded {len(loader)} embeddings")

    # Test single embedding
    emb = loader.get_embedding('2024-01-15T10:30:00', 0)
    print(f"Single embedding shape: {emb.shape}")

    # Test batch embedding
    timestamps = ['2024-01-15T10:30:00', '2024-01-15T10:35:00']
    batch_emb, has_signal = loader.get_batch_embedding(timestamps, n_assets=2)
    print(f"Batch embedding shape: {batch_emb.shape}")
    print(f"Has signal shape: {has_signal.shape}")
    print(f"Has signal values:\n{has_signal.squeeze()}")

    # Clean up
    Path(temp_path).unlink()
    print("\nTest completed successfully!")
