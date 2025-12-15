#!/usr/bin/env python3
"""
Profile embedding lookup performance to identify bottlenecks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
import numpy as np
import torch
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def measure(name, times_dict):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    times_dict[name] = times_dict.get(name, []) + [elapsed]


def test_real_embedding_loader():
    """Test the actual EmbeddingLoader with real H5 files."""
    from agents.embedding_loader import EmbeddingLoader

    # Check if real embedding files exist
    gdelt_path = Path("/home/work/RL/stair-local/data/embeddings/gdelt_embeddings.h5")
    nostr_path = Path("/home/work/RL/stair-local/data/embeddings/nostr_embeddings.h5")

    if not gdelt_path.exists() or not nostr_path.exists():
        print("Real embedding files not found, skipping real EmbeddingLoader test")
        return

    print("\n" + "=" * 70)
    print("REAL EmbeddingLoader Test")
    print("=" * 70)

    loader = EmbeddingLoader(str(gdelt_path), str(nostr_path))

    T = 24
    n_iterations = 100
    current_time = pd.Timestamp('2021-06-15T12:00:00+00:00')

    # Build temporal timestamps
    temporal_timestamps = []
    for i in range(T):
        ts = current_time - pd.Timedelta(minutes=5 * (T - 1 - i))
        temporal_timestamps.append(ts.isoformat())

    asset_indices = list(range(20))

    times = {}

    # Test slow method
    print(f"\nProfiling SLOW method (T={T}, iters={n_iterations})...")
    for _ in range(n_iterations):
        with measure("slow_gdelt", times):
            _ = loader.get_gdelt_embeddings_marketwide(temporal_timestamps, asset_indices)
        with measure("slow_nostr", times):
            _ = loader.get_nostr_embeddings_marketwide(temporal_timestamps, asset_indices)
        with measure("slow_mask", times):
            _ = loader.get_social_signal_mask_marketwide(temporal_timestamps, asset_indices)

    # Test fast method
    print(f"Profiling FAST method (T={T}, iters={n_iterations})...")
    for _ in range(n_iterations):
        with measure("fast_gdelt", times):
            _ = loader.get_gdelt_embeddings_marketwide_fast(temporal_timestamps, asset_indices)
        with measure("fast_nostr", times):
            _ = loader.get_nostr_embeddings_marketwide_fast(temporal_timestamps, asset_indices)
        with measure("fast_mask", times):
            _ = loader.get_social_signal_mask_marketwide_fast(temporal_timestamps, asset_indices)

    # Report
    print(f"\n{'Method':<25} {'Avg(ms)':<12} {'Total(s)':<10}")
    print("-" * 50)
    for name in ["slow_gdelt", "slow_nostr", "slow_mask", "fast_gdelt", "fast_nostr", "fast_mask"]:
        if name in times:
            avg_ms = np.mean(times[name]) * 1000
            total = sum(times[name])
            print(f"{name:<25} {avg_ms:<12.3f} {total:<10.3f}")

    slow_total = sum(times.get("slow_gdelt", [0])) + sum(times.get("slow_nostr", [0])) + sum(times.get("slow_mask", [0]))
    fast_total = sum(times.get("fast_gdelt", [0])) + sum(times.get("fast_nostr", [0])) + sum(times.get("fast_mask", [0]))

    print("-" * 50)
    print(f"Speedup: {slow_total / fast_total:.1f}x")
    print(f"\nEstimated savings per training step:")
    slow_per_step = (slow_total / n_iterations) * 4  # 4 encode_state calls
    fast_per_step = (fast_total / n_iterations) * 4
    print(f"  Slow: {slow_per_step * 1000:.1f}ms")
    print(f"  Fast: {fast_per_step * 1000:.1f}ms")
    print(f"  Savings: {(slow_per_step - fast_per_step) * 1000:.1f}ms/step")

    loader.close()


def profile_embedding_lookup():
    """Profile the slow parts of embedding lookup."""

    # Simulate embedding loader with preloaded data
    print("Creating mock preloaded embeddings...")

    # Mock ~100k embeddings (typical for 18 months of data)
    n_embeddings = 100000
    embedding_dim = 768

    # Create mock index and embeddings
    mock_timestamps = pd.date_range('2021-01-01', periods=5000, freq='5min')
    mock_assets = list(range(20))

    # Build mock index like the real EmbeddingLoader does
    mock_index = {}
    for ts in mock_timestamps:
        ts_str = ts.isoformat()
        for asset in mock_assets:
            key = f"{ts_str}_{asset}"
            mock_index[key] = len(mock_index) % n_embeddings

    # Mock preloaded embeddings (in-memory numpy arrays)
    mock_embeddings = {key: np.random.randn(embedding_dim).astype(np.float32)
                       for key in list(mock_index.keys())[:n_embeddings]}

    print(f"  Mock index size: {len(mock_index):,}")
    print(f"  Mock embeddings: {len(mock_embeddings):,}")

    # Parameters
    T = 24  # temporal window
    N = 20  # assets
    n_iterations = 100

    times = {}

    # Test 1: Current approach (Python for loops)
    print(f"\nProfiling current approach (T={T}, N={N}, iters={n_iterations})...")

    current_time = pd.Timestamp('2021-06-15T12:00:00')

    for i in range(n_iterations):
        with measure("total_lookup", times):
            # Step 1: Build temporal timestamps (like build_state_dict line 250-254)
            with measure("build_temporal_timestamps", times):
                temporal_timestamps = []
                for j in range(T):
                    ts = current_time - pd.Timedelta(minutes=5 * (T - 1 - j))
                    temporal_timestamps.append(ts.isoformat())

            # Step 2: Round timestamps (like _round_timestamp)
            with measure("round_timestamps", times):
                rounded_timestamps = []
                for ts in temporal_timestamps:
                    dt = pd.Timestamp(ts)
                    minute = (dt.minute // 5) * 5
                    rounded = dt.replace(minute=minute, second=0, microsecond=0)
                    rounded_timestamps.append(rounded.isoformat())

            # Step 3: Loop through T x N and look up embeddings
            with measure("embedding_loop", times):
                result = torch.zeros(T, embedding_dim)
                for t, ts_rounded in enumerate(rounded_timestamps):
                    asset_embs = []
                    for asset_idx in range(N):
                        with measure("key_format", times):
                            key = f"{ts_rounded}_{asset_idx}"

                        with measure("dict_lookup", times):
                            if key in mock_embeddings:
                                emb = mock_embeddings[key]
                                asset_embs.append(emb)

                    # Mean pooling
                    with measure("mean_pool", times):
                        if asset_embs:
                            mean_emb = np.mean(asset_embs, axis=0)
                            result[t, :] = torch.from_numpy(mean_emb)

    # Report
    print("\n" + "=" * 70)
    print("PROFILING REPORT")
    print("=" * 70)
    print(f"{'Component':<35} {'Total(s)':<10} {'Avg(ms)':<12} {'Per-step(ms)':<15}")
    print("-" * 70)

    for name in ["total_lookup", "build_temporal_timestamps", "round_timestamps",
                 "embedding_loop", "key_format", "dict_lookup", "mean_pool"]:
        if name in times:
            total = sum(times[name])
            avg_ms = np.mean(times[name]) * 1000
            per_step_ms = total / n_iterations * 1000
            print(f"{name:<35} {total:<10.3f} {avg_ms:<12.4f} {per_step_ms:<15.4f}")

    print("-" * 70)

    # Compute estimate for 4 encode_state calls per step
    total_lookup_avg = np.mean(times["total_lookup"]) * 1000
    est_per_step = total_lookup_avg * 4 * 3  # 4 encode calls × 3 embedding types
    print(f"\nEstimated per training step: {est_per_step:.1f}ms")
    print(f"  (4 encode_state calls × 3 embedding types × {total_lookup_avg:.2f}ms/lookup)")

    if est_per_step > 100:
        print(f"\n⚠️  Embedding lookup is a major bottleneck!")
        print(f"    930ms/step observed, embedding lookup est: {est_per_step:.0f}ms ({est_per_step/930*100:.0f}%)")


def profile_optimized_approach():
    """Profile an optimized approach using vectorized operations."""

    print("\n\n" + "=" * 70)
    print("OPTIMIZED APPROACH (Vectorized)")
    print("=" * 70)

    # Pre-compute all possible keys and their indices
    n_embeddings = 100000
    embedding_dim = 768
    T = 24
    N = 20
    n_iterations = 100

    # Preloaded embeddings as contiguous numpy array
    embeddings_array = np.random.randn(n_embeddings, embedding_dim).astype(np.float32)

    # Pre-computed timestamp → base_index mapping
    mock_timestamps = pd.date_range('2021-01-01', periods=5000, freq='5min')
    ts_to_idx = {ts.isoformat(): i * N for i, ts in enumerate(mock_timestamps)}

    current_time = pd.Timestamp('2021-06-15T12:00:00')

    times = {}

    for i in range(n_iterations):
        with measure("total_optimized", times):
            # Step 1: Compute all T timestamps at once
            with measure("compute_timestamps", times):
                base_ts = current_time.floor('5min')
                timestamps = [base_ts - pd.Timedelta(minutes=5*j) for j in range(T-1, -1, -1)]

            # Step 2: Look up base indices
            with measure("lookup_indices", times):
                indices = []
                for ts in timestamps:
                    ts_str = ts.isoformat()
                    if ts_str in ts_to_idx:
                        base_idx = ts_to_idx[ts_str]
                        # Get all N asset embeddings
                        indices.extend(range(base_idx, min(base_idx + N, n_embeddings)))

            # Step 3: Batch numpy indexing
            with measure("numpy_fancy_index", times):
                if indices:
                    selected = embeddings_array[indices[:T*N]]
                    selected = selected.reshape(T, N, embedding_dim)
                    # Mean pool across assets
                    result = selected.mean(axis=1)  # (T, embedding_dim)

    print(f"\n{'Component':<35} {'Total(s)':<10} {'Avg(ms)':<12}")
    print("-" * 70)

    for name in ["total_optimized", "compute_timestamps", "lookup_indices", "numpy_fancy_index"]:
        if name in times:
            total = sum(times[name])
            avg_ms = np.mean(times[name]) * 1000
            print(f"{name:<35} {total:<10.3f} {avg_ms:<12.4f}")

    total_optimized_avg = np.mean(times["total_optimized"]) * 1000
    print(f"\nOptimized per training step est: {total_optimized_avg * 4 * 3:.1f}ms")


if __name__ == '__main__':
    # Test with mock data first
    profile_embedding_lookup()
    profile_optimized_approach()

    # Test with real EmbeddingLoader if available
    test_real_embedding_loader()
