#!/usr/bin/env python3
"""
Test embedding loading speed with memory caching optimization.

Expected results:
- Before: 54.7ms per embedding (disk I/O)
- After: <1ms per embedding (in-memory lookup)
- Speedup: 50-100x
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
from datetime import datetime, timedelta

from agents.embedding_loader import EmbeddingLoader


def test_embedding_loading_speed():
    """Test embedding loading performance."""
    print("=" * 80)
    print("EMBEDDING LOADING SPEED TEST")
    print("=" * 80)

    # Check if embedding files exist
    gdelt_path = "/home/work/data/stair-local/embeddings/gdelt_embeddings.h5"
    nostr_path = "/home/work/data/stair-local/embeddings/nostr_embeddings.h5"

    if not Path(gdelt_path).exists():
        print(f"❌ GDELT embeddings not found: {gdelt_path}")
        return False

    if not Path(nostr_path).exists():
        print(f"❌ Nostr embeddings not found: {nostr_path}")
        return False

    # Initialize loader (will load all embeddings into memory)
    print("\nInitializing EmbeddingLoader...")
    start = time.time()
    loader = EmbeddingLoader(
        gdelt_path=gdelt_path,
        nostr_path=nostr_path,
        device='cpu',
    )
    init_time = time.time() - start
    print(f"Initialization time: {init_time:.2f}s")

    # Test 1: Single embedding lookup (warmup)
    print("\n--- Test 1: Single Embedding Lookup ---")
    timestamps = ["2024-06-15T12:00:00+00:00"]
    assets = [0]

    start = time.time()
    gdelt_emb = loader.get_gdelt_embeddings(timestamps, assets)
    single_time = (time.time() - start) * 1000  # ms
    print(f"Single GDELT embedding: {single_time:.3f}ms")

    # Test 2: Batch loading (realistic scenario)
    print("\n--- Test 2: Batch Loading (Realistic Scenario) ---")
    # Simulate batch_size=384, temporal_window=20, n_assets=20
    B = 384
    T = 20
    N = 20
    total_embeddings = B * T * N  # 153,600 embeddings

    # Generate timestamps (spaced 5 minutes apart)
    base_time = datetime(2024, 6, 15, 12, 0, 0)
    timestamps = []
    for i in range(B * T):
        ts = base_time + timedelta(minutes=5 * i)
        timestamps.append(ts.isoformat() + "+00:00")

    assets = list(range(N))

    print(f"Batch size: {B}")
    print(f"Temporal window: {T}")
    print(f"Assets: {N}")
    print(f"Total embeddings to load: {total_embeddings:,}")

    # Test GDELT loading
    print("\nLoading GDELT embeddings...")
    start = time.time()
    gdelt_batch = loader.get_gdelt_embeddings(timestamps, assets)
    gdelt_time = time.time() - start
    print(f"  Time: {gdelt_time:.2f}s")
    print(f"  Per embedding: {(gdelt_time / total_embeddings) * 1000:.3f}ms")
    print(f"  Shape: {gdelt_batch.shape}")

    # Test Nostr loading
    print("\nLoading Nostr embeddings...")
    start = time.time()
    nostr_batch = loader.get_nostr_embeddings(timestamps, assets)
    nostr_time = time.time() - start
    print(f"  Time: {nostr_time:.2f}s")
    print(f"  Per embedding: {(nostr_time / total_embeddings) * 1000:.3f}ms")
    print(f"  Shape: {nostr_batch.shape}")

    # Combined time
    total_time = gdelt_time + nostr_time
    print(f"\nTotal loading time: {total_time:.2f}s")
    print(f"Per embedding (average): {(total_time / (total_embeddings * 2)) * 1000:.3f}ms")

    # Calculate training speedup
    print("\n" + "=" * 80)
    print("TRAINING SPEEDUP ESTIMATION")
    print("=" * 80)

    # Before optimization: 54.7ms per embedding (disk I/O)
    old_time_per_embedding = 54.7  # ms
    old_batch_time = (total_embeddings * 2) * old_time_per_embedding / 1000 / 60  # minutes

    # After optimization
    new_time_per_embedding = (total_time / (total_embeddings * 2)) * 1000  # ms
    new_batch_time = total_time / 60  # minutes

    speedup = old_batch_time / new_batch_time

    print(f"\nBefore optimization (disk I/O):")
    print(f"  Per embedding: {old_time_per_embedding}ms")
    print(f"  Batch time: {old_batch_time:.1f} minutes")

    print(f"\nAfter optimization (in-memory):")
    print(f"  Per embedding: {new_time_per_embedding:.3f}ms")
    print(f"  Batch time: {new_batch_time:.2f} minutes")

    print(f"\n🚀 Speedup: {speedup:.1f}x faster!")

    # Training time estimation
    print("\n" + "=" * 80)
    print("TRAINING TIME ESTIMATION (500K steps)")
    print("=" * 80)

    total_steps = 500_000

    old_training_time = (old_batch_time * total_steps) / 60 / 24  # days
    new_training_time = (new_batch_time * total_steps) / 60 / 24  # days

    print(f"Before optimization: {old_training_time:.1f} days")
    print(f"After optimization: {new_training_time:.1f} days")
    print(f"Time saved: {old_training_time - new_training_time:.1f} days")

    if new_training_time < 1:
        print(f"\n✅ Training time reduced to {new_training_time * 24:.1f} hours!")

    loader.close()

    return True


if __name__ == '__main__':
    success = test_embedding_loading_speed()
    sys.exit(0 if success else 1)
