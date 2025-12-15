#!/usr/bin/env python3
"""
Test script for vectorized environment.

Tests:
1. Single environment (DummyVecEnv)
2. Multiple parallel environments (SubprocVecEnv)
3. Performance comparison
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.vectorized_env import make_vec_env
from environments.trading_env import EnvConfig
from config.settings import DATA_DIR

def test_single_env():
    """Test single environment."""
    print("\n" + "=" * 60)
    print("Test 1: Single environment (DummyVecEnv)")
    print("=" * 60)

    env_config = {
        'n_assets': 20,
        'target_leverage': 1.0,
        'transaction_cost_rate': 0.0004,
    }

    env = make_vec_env(env_config, n_envs=1, seed=42)

    print(f"✓ Created single environment")
    print(f"  n_envs: {env.n_envs}")
    print(f"  observation_space: {env.observation_space}")
    print(f"  action_space: {env.action_space}")

    # Test reset
    obs = env.reset()
    print(f"\n✓ Reset environment")
    print(f"  obs shape: {obs.shape} (expected: (1, 20, 36))")
    assert obs.shape[0] == 1, f"Expected n_envs=1, got {obs.shape[0]}"

    # Test step
    actions = np.random.randn(1, 20)  # Random actions for 1 env, 20 assets
    next_obs, rewards, dones, infos = env.step(actions)

    print(f"\n✓ Step environment")
    print(f"  next_obs shape: {next_obs.shape}")
    print(f"  rewards shape: {rewards.shape}")
    print(f"  dones shape: {dones.shape}")
    print(f"  rewards: {rewards}")

    env.close()
    print(f"\n✓ Closed environment")
    return True


def test_multi_env():
    """Test multiple parallel environments."""
    print("\n" + "=" * 60)
    print("Test 2: Multiple environments (SubprocVecEnv)")
    print("=" * 60)

    n_envs = 4
    env_config = {
        'n_assets': 20,
        'target_leverage': 1.0,
        'transaction_cost_rate': 0.0004,
    }

    env = make_vec_env(env_config, n_envs=n_envs, seed=42)

    print(f"✓ Created {n_envs} parallel environments")
    print(f"  n_envs: {env.n_envs}")

    # Test reset
    obs = env.reset()
    print(f"\n✓ Reset all environments")
    print(f"  obs shape: {obs.shape} (expected: ({n_envs}, 20, 36))")
    assert obs.shape[0] == n_envs, f"Expected n_envs={n_envs}, got {obs.shape[0]}"

    # Test step
    actions = np.random.randn(n_envs, 20)  # Random actions for n_envs, 20 assets
    next_obs, rewards, dones, infos = env.step(actions)

    print(f"\n✓ Step all environments")
    print(f"  next_obs shape: {next_obs.shape}")
    print(f"  rewards shape: {rewards.shape}")
    print(f"  dones shape: {dones.shape}")
    print(f"  rewards: {rewards}")
    print(f"  infos length: {len(infos)}")

    env.close()
    print(f"\n✓ Closed all environments")
    return True


def test_performance():
    """Compare performance of single vs vectorized environments."""
    print("\n" + "=" * 60)
    print("Test 3: Performance comparison")
    print("=" * 60)

    n_steps = 100
    n_envs = 8

    env_config = {
        'n_assets': 20,
        'target_leverage': 1.0,
        'transaction_cost_rate': 0.0004,
    }

    # Single environment
    print(f"\nRunning {n_steps} steps with 1 environment...")
    env1 = make_vec_env(env_config, n_envs=1, seed=42)
    obs1 = env1.reset()

    start_time = time.time()
    for _ in range(n_steps):
        actions = np.random.randn(1, 20)
        obs1, rewards, dones, infos = env1.step(actions)
    time1 = time.time() - start_time
    env1.close()

    print(f"  Time: {time1:.3f}s ({time1/n_steps*1000:.2f} ms/step)")

    # Vectorized environment
    print(f"\nRunning {n_steps} steps with {n_envs} environments...")
    envN = make_vec_env(env_config, n_envs=n_envs, seed=42)
    obsN = envN.reset()

    start_time = time.time()
    for _ in range(n_steps):
        actions = np.random.randn(n_envs, 20)
        obsN, rewards, dones, infos = envN.step(actions)
    timeN = time.time() - start_time
    envN.close()

    print(f"  Time: {timeN:.3f}s ({timeN/n_steps*1000:.2f} ms/step)")

    # Calculate speedup
    total_steps1 = n_steps * 1
    total_stepsN = n_steps * n_envs
    speedup = (time1 / total_steps1) / (timeN / total_stepsN)

    print(f"\n✓ Performance summary:")
    print(f"  Single env: {total_steps1} total steps in {time1:.3f}s")
    print(f"  {n_envs} envs: {total_stepsN} total steps in {timeN:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  (Ideal speedup: {n_envs}x)")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("VECTORIZED ENVIRONMENT TESTS")
    print("=" * 80)

    try:
        # Test 1: Single environment
        test_single_env()

        # Test 2: Multiple environments
        test_multi_env()

        # Test 3: Performance comparison
        test_performance()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print("\nVectorized environment is ready for training!")
        print("Usage:")
        print("  python scripts/run_training.py --phase 2 --n-envs 64")

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
