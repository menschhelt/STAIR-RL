#!/usr/bin/env python3
"""
Quick test for vectorization changes.
Tests data_loader.py and cql_sac.py ReplayBuffer without full training.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datetime import datetime
from config.settings import Config, DATA_DIR
from training.data_loader import TrainingDataLoader

def test_data_loader():
    """Test vectorized data loading."""
    print("=" * 60)
    print("Testing vectorized data_loader.py")
    print("=" * 60)

    config = Config()

    loader = TrainingDataLoader(
        data_dir=DATA_DIR,
        n_assets=config.universe.top_n,
    )

    # Use short period for testing (1 week)
    start = datetime(2022, 1, 1)
    end = datetime(2022, 1, 7)

    print(f"Loading data: {start} to {end}")
    start_time = time.time()

    data = loader.load_period_dynamic(
        start_date=start,
        end_date=end,
    )

    elapsed = time.time() - start_time

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  States shape: {data['states'].shape}")
    print(f"  Returns shape: {data['returns'].shape}")
    print(f"  Prices shape: {data['prices'].shape}")
    print(f"  Slot changes: {data['slot_changes'].sum()}")
    print(f"  Rebalances: {data['rebalance_mask'].sum()}")

    # Sanity checks
    assert not np.isnan(data['states']).any(), "NaN in states!"
    assert not np.isnan(data['returns']).any(), "NaN in returns!"
    assert not np.isnan(data['prices']).any(), "NaN in prices!"

    print("\n[PASS] data_loader.py vectorization works!")
    return data


def test_replay_buffer(data):
    """Test vectorized ReplayBuffer loading."""
    print("\n" + "=" * 60)
    print("Testing vectorized ReplayBuffer.load_from_data()")
    print("=" * 60)

    from agents.cql_sac import ReplayBuffer

    T, N, D = data['states'].shape
    portfolio_dim = N + 2  # weights + leverage + cash

    # Create buffer
    buffer = ReplayBuffer(
        capacity=T + 1000,
        n_assets=N,
        state_dim=D,
        portfolio_dim=portfolio_dim,
        device='cpu',
    )

    # Create dummy portfolio states and actions
    portfolio_states = np.zeros((T, portfolio_dim), dtype=np.float32)
    actions = np.random.randn(T, N).astype(np.float32) * 0.1
    rewards = data['returns'].sum(axis=1)  # Simple reward
    dones = np.zeros(T, dtype=np.float32)
    dones[-1] = 1.0  # Episode ends

    print(f"Loading {T} transitions into buffer...")
    start_time = time.time()

    buffer.load_from_data(
        market_states=data['states'],
        portfolio_states=portfolio_states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        timestamps=data['timestamps'],
    )

    elapsed = time.time() - start_time

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Buffer size: {len(buffer)}")
    print(f"  Expected: {T - 1}")

    # Sample a batch to verify
    batch = buffer.sample(32)
    print(f"  Sample batch market_states: {batch['market_states'].shape}")
    print(f"  Sample batch rewards: {batch['rewards'].shape}")

    assert len(buffer) == T - 1, f"Buffer size mismatch: {len(buffer)} != {T-1}"

    print("\n[PASS] ReplayBuffer vectorization works!")


def main():
    print("\n" + "=" * 60)
    print("VECTORIZATION TEST")
    print("=" * 60 + "\n")

    # Test 1: Data loader
    data = test_data_loader()

    # Test 2: Replay buffer
    test_replay_buffer(data)

    print("\n" + "=" * 60)
    print("[ALL TESTS PASSED]")
    print("Vectorization is working correctly!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
