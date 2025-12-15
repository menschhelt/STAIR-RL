#!/usr/bin/env python3
"""
Profile build_state_dict to find remaining bottlenecks.
Target: Identify components taking >10ms to optimize.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import torch
from contextlib import contextmanager
from collections import defaultdict

@contextmanager
def measure(name, times_dict):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    times_dict[name] = times_dict.get(name, []) + [elapsed]


def profile_alpha_loader():
    """Profile AlphaLoader performance."""
    from agents.alpha_loader import AlphaLoader
    from config.settings import DATA_DIR

    print("\n" + "=" * 70)
    print("ALPHA LOADER PROFILING")
    print("=" * 70)

    alpha_cache_dir = DATA_DIR / 'features' / 'alpha_cache_normalized'
    if not alpha_cache_dir.exists():
        print("Alpha cache not found, skipping")
        return

    loader = AlphaLoader(alpha_cache_dir=alpha_cache_dir)

    # Check if preloaded
    print(f"Preloaded: {loader.is_preloaded()}")
    print(f"Has slot mapping: {loader.has_slot_mapping()}")

    if not loader.is_preloaded():
        print("Alpha loader not preloaded, skipping detailed profiling")
        return

    # Profile get_alphas_for_slots_fast
    T = 24
    n_iters = 100
    times = {}

    # Create test timestamps
    base_ts = pd.Timestamp('2021-06-15T12:00:00', tz='UTC')
    timestamps = [base_ts - pd.Timedelta(minutes=5*i) for i in range(T-1, -1, -1)]

    print(f"\nProfiling get_alphas_for_slots_fast (T={T}, iters={n_iters})...")

    for _ in range(n_iters):
        with measure("get_alphas_for_slots_fast", times):
            _ = loader.get_alphas_for_slots_fast(timestamps)

    avg_ms = np.mean(times["get_alphas_for_slots_fast"]) * 1000
    print(f"  Avg: {avg_ms:.3f}ms per call")


def profile_macro_loader():
    """Profile MacroDataLoader performance."""
    from features.macro_loader import MacroDataLoader
    from config.settings import DATA_DIR

    print("\n" + "=" * 70)
    print("MACRO LOADER PROFILING")
    print("=" * 70)

    macro_dir = DATA_DIR / 'macro'
    if not macro_dir.exists():
        print("Macro dir not found, skipping")
        return

    loader = MacroDataLoader(macro_data_dir=macro_dir)

    print(f"Preloaded: {loader.is_preloaded()}")

    if not loader.is_preloaded():
        print("Macro loader not preloaded, skipping fast method profiling")
        # Profile slow method
        T = 24
        n_iters = 20
        times = {}

        base_ts = pd.Timestamp('2021-06-15T12:00:00', tz='UTC')
        timestamps = [base_ts - pd.Timedelta(minutes=5*i) for i in range(T-1, -1, -1)]

        print(f"\nProfiling get_global_features SLOW (T={T}, iters={n_iters})...")

        for _ in range(n_iters):
            with measure("macro_slow_total", times):
                for ts in timestamps:
                    with measure("macro_slow_single", times):
                        _ = loader.get_global_features(ts)

        avg_total = np.mean(times["macro_slow_total"]) * 1000
        avg_single = np.mean(times["macro_slow_single"]) * 1000
        print(f"  Total (24 calls): {avg_total:.3f}ms")
        print(f"  Per call: {avg_single:.3f}ms")
        return

    # Profile fast method
    T = 24
    n_iters = 100
    times = {}

    base_ts = pd.Timestamp('2021-06-15T12:00:00', tz='UTC')
    timestamps = [base_ts - pd.Timedelta(minutes=5*i) for i in range(T-1, -1, -1)]

    print(f"\nProfiling get_features_fast (T={T}, iters={n_iters})...")

    for _ in range(n_iters):
        with measure("macro_fast_total", times):
            for ts in timestamps:
                with measure("macro_fast_single", times):
                    _ = loader.get_features_fast(ts)

    avg_total = np.mean(times["macro_fast_total"]) * 1000
    avg_single = np.mean(times["macro_fast_single"]) * 1000
    print(f"  Total (24 calls): {avg_total:.3f}ms")
    print(f"  Per call: {avg_single:.3f}ms")


def profile_timestamp_operations():
    """Profile timestamp parsing and arithmetic."""
    print("\n" + "=" * 70)
    print("TIMESTAMP OPERATIONS PROFILING")
    print("=" * 70)

    T = 24
    n_iters = 1000
    times = {}

    base_ts = pd.Timestamp('2021-06-15T12:00:00', tz='UTC')

    # Test 1: pd.Timedelta in loop (current approach)
    print(f"\nProfiling temporal_timestamps construction (T={T}, iters={n_iters})...")

    for _ in range(n_iters):
        with measure("timedelta_loop", times):
            temporal_timestamps = []
            for i in range(T):
                ts = base_ts - pd.Timedelta(minutes=5 * (T - 1 - i))
                temporal_timestamps.append(ts.isoformat())

    # Test 2: Vectorized with date_range
    for _ in range(n_iters):
        with measure("date_range", times):
            end_ts = base_ts
            start_ts = base_ts - pd.Timedelta(minutes=5 * (T - 1))
            temporal_timestamps = pd.date_range(start_ts, end_ts, periods=T)

    # Test 3: Pre-computed offsets
    offsets = np.arange(T-1, -1, -1) * 5  # minutes
    for _ in range(n_iters):
        with measure("precomputed_offsets", times):
            temporal_timestamps = [
                (base_ts - pd.Timedelta(minutes=int(offset))).isoformat()
                for offset in offsets
            ]

    # Test 4: numpy timedelta
    for _ in range(n_iters):
        with measure("numpy_timedelta", times):
            base_np = np.datetime64(base_ts.value, 'ns')
            offsets_ns = np.arange(T-1, -1, -1) * 5 * 60 * 1e9  # minutes to ns
            timestamps_np = base_np - offsets_ns.astype('timedelta64[ns]')

    print(f"\n{'Method':<25} {'Avg(ms)':<12}")
    print("-" * 40)
    for name in ["timedelta_loop", "date_range", "precomputed_offsets", "numpy_timedelta"]:
        avg_ms = np.mean(times[name]) * 1000
        print(f"{name:<25} {avg_ms:.4f}")


def profile_build_state_dict_components():
    """Profile individual components of build_state_dict."""
    print("\n" + "=" * 70)
    print("BUILD_STATE_DICT COMPONENT PROFILING")
    print("=" * 70)

    # Try to import and test with real data
    try:
        from agents.hierarchical_state_builder import HierarchicalStateBuilder
        from config.settings import DATA_DIR

        print("Initializing HierarchicalStateBuilder...")

        builder = HierarchicalStateBuilder(
            n_assets=20,
            n_alphas=101,
            temporal_window=24,
            gdelt_embeddings_path=str(DATA_DIR / 'embeddings' / 'gdelt_embeddings.h5'),
            nostr_embeddings_path=str(DATA_DIR / 'embeddings' / 'nostr_embeddings.h5'),
            device='cpu',
        )

        # Set symbols
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
                   'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT', 'DOTUSDT', 'LINKUSDT',
                   'MATICUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'XLMUSDT',
                   'NEARUSDT', 'ICPUSDT', 'FILUSDT', 'APTUSDT', 'ARBUSDT']
        builder.set_symbols(symbols)

        # Create test inputs
        B, N = 1, 20
        market_state = torch.randn(B, N, 36)
        portfolio_state = torch.randn(B, 22)

        # Test timestamp
        timestamps = [pd.Timestamp('2021-06-15T12:00:00', tz='UTC').isoformat()]

        times = {}
        n_iters = 50

        print(f"\nProfiling build_state_dict (B={B}, iters={n_iters})...")

        for _ in range(n_iters):
            with measure("build_state_dict_total", times):
                _ = builder.build_state_dict(market_state, portfolio_state, timestamps)

        avg_ms = np.mean(times["build_state_dict_total"]) * 1000
        print(f"  Total: {avg_ms:.2f}ms per call")
        print(f"  Per training step (×4 calls): {avg_ms * 4:.2f}ms")

    except Exception as e:
        print(f"Could not test with real data: {e}")
        import traceback
        traceback.print_exc()


def profile_alpha_slots_inner_loop():
    """Profile the inner loop of get_alphas_for_slots (the slow part)."""
    print("\n" + "=" * 70)
    print("ALPHA SLOTS INNER LOOP PROFILING")
    print("=" * 70)

    # Simulate the data structures
    T_total = 200000
    N_all = 100
    N_slots = 20
    n_alphas = 101

    print(f"Simulating: T_total={T_total:,}, N_all={N_all}, N_slots={N_slots}, n_alphas={n_alphas}")

    # Preloaded data
    preloaded_data = np.random.randn(T_total, N_all, n_alphas).astype(np.float32)
    slot_symbol_indices = np.random.randint(0, N_all, (T_total, N_slots)).astype(np.int32)

    # Test case: T_batch = 24
    T_batch = 24
    timestamp_indices = np.random.randint(0, T_total, T_batch)

    n_iters = 100
    times = {}

    # Method 1: Current (Python double loop)
    print(f"\nProfiling inner loop methods (T_batch={T_batch}, iters={n_iters})...")

    for _ in range(n_iters):
        with measure("python_double_loop", times):
            slot_indices = slot_symbol_indices[timestamp_indices]
            result = np.zeros((T_batch, N_slots, n_alphas), dtype=np.float32)
            for i, t_idx in enumerate(timestamp_indices):
                for slot_idx in range(N_slots):
                    preload_idx = slot_indices[i, slot_idx]
                    result[i, slot_idx, :] = preloaded_data[t_idx, preload_idx, :]

    # Method 2: Numpy advanced indexing
    for _ in range(n_iters):
        with measure("numpy_advanced_indexing", times):
            slot_indices = slot_symbol_indices[timestamp_indices]  # (T_batch, N_slots)
            # Create index arrays for advanced indexing
            t_indices = np.arange(T_batch)[:, None].repeat(N_slots, axis=1)  # (T_batch, N_slots)
            result = preloaded_data[timestamp_indices[:, None], slot_indices, :]

    # Method 3: np.take_along_axis (if applicable)
    for _ in range(n_iters):
        with measure("numpy_take", times):
            # Get batch data first: (T_batch, N_all, n_alphas)
            batch_data = preloaded_data[timestamp_indices]
            # Get slot indices: (T_batch, N_slots)
            slot_indices = slot_symbol_indices[timestamp_indices]
            # Expand for gather: (T_batch, N_slots, n_alphas)
            slot_indices_expanded = slot_indices[:, :, None].repeat(n_alphas, axis=2)
            # Gather
            result = np.take_along_axis(batch_data, slot_indices_expanded, axis=1)

    print(f"\n{'Method':<30} {'Avg(ms)':<12} {'Speedup':<10}")
    print("-" * 55)

    baseline = np.mean(times["python_double_loop"]) * 1000
    for name in ["python_double_loop", "numpy_advanced_indexing", "numpy_take"]:
        avg_ms = np.mean(times[name]) * 1000
        speedup = baseline / avg_ms
        print(f"{name:<30} {avg_ms:.4f}      {speedup:.1f}x")


if __name__ == '__main__':
    profile_timestamp_operations()
    profile_alpha_slots_inner_loop()
    profile_macro_loader()
    profile_alpha_loader()
    profile_build_state_dict_components()
