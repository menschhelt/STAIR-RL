#!/usr/bin/env python3
"""
Example: Collecting Offline RL Data with Timestamps

This script demonstrates the correct format for offline RL data collection.
The collected data can be used for CQL-SAC Phase 1 pre-training.

Required format:
    {
        'market_states': np.ndarray (T, n_assets, state_dim),
        'portfolio_states': np.ndarray (T, portfolio_dim),
        'actions': np.ndarray (T, n_assets),
        'rewards': np.ndarray (T,),
        'dones': np.ndarray (T,),
        'timestamps': np.ndarray (T,) - ISO format strings e.g. "2024-01-01T12:00:00+00:00"
    }

IMPORTANT: Timestamps are required for embedding lookup during training!
Without timestamps, all text embeddings will be zero vectors.
"""

import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.trading_env import TradingEnv, TradingEnvConfig


def collect_trajectories_with_random_policy(
    env: TradingEnv,
    num_episodes: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Collect trajectories using a random policy.

    Args:
        env: Trading environment
        num_episodes: Number of episodes to collect

    Returns:
        Dict with market_states, portfolio_states, actions, rewards, dones, timestamps
    """
    # Storage lists
    market_states_list = []
    portfolio_states_list = []
    actions_list = []
    rewards_list = []
    dones_list = []
    timestamps_list = []

    for ep in range(num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}")

        obs, info = env.reset()
        done = False
        step = 0

        while not done:
            # Random action (for demonstration)
            action = env.action_space.sample()

            # Get current timestamp
            current_step = env._step_idx
            if 'timestamps' in env.data and current_step < len(env.data['timestamps']):
                timestamp = env.data['timestamps'][current_step]
                # Convert to ISO string if it's datetime
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.isoformat()
            else:
                # Fallback: generate timestamp from step index
                timestamp = None

            # Store transition
            market_states_list.append(obs['market'])
            portfolio_states_list.append(obs['portfolio'])
            actions_list.append(action)

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rewards_list.append(reward)
            dones_list.append(float(done))
            if timestamp is not None:
                timestamps_list.append(timestamp)

            obs = next_obs
            step += 1

        print(f"  Collected {step} steps")

    # Convert to arrays
    market_states = np.array(market_states_list, dtype=np.float32)
    portfolio_states = np.array(portfolio_states_list, dtype=np.float32)
    actions = np.array(actions_list, dtype=np.float32)
    rewards = np.array(rewards_list, dtype=np.float32)
    dones = np.array(dones_list, dtype=np.float32)
    timestamps = np.array(timestamps_list, dtype='U32')  # ISO string format

    print(f"\nCollected data shapes:")
    print(f"  market_states: {market_states.shape}")
    print(f"  portfolio_states: {portfolio_states.shape}")
    print(f"  actions: {actions.shape}")
    print(f"  rewards: {rewards.shape}")
    print(f"  dones: {dones.shape}")
    print(f"  timestamps: {timestamps.shape}")

    return {
        'market_states': market_states,
        'portfolio_states': portfolio_states,
        'actions': actions,
        'rewards': rewards,
        'dones': dones,
        'timestamps': timestamps,
    }


def save_offline_data(data: Dict[str, np.ndarray], output_path: Path):
    """
    Save offline data in the correct format for CQL-SAC training.

    Args:
        data: Dict with offline RL trajectories
        output_path: Path to save .npz file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_path, **data)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved offline data to {output_path}")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Total transitions: {len(data['rewards'])}")

    # Verify timestamps
    if 'timestamps' in data and len(data['timestamps']) > 0:
        print(f"  Timestamp range: {data['timestamps'][0]} to {data['timestamps'][-1]}")
    else:
        print("  WARNING: No timestamps in data! Embeddings will be zero vectors during training.")


def main():
    """Example usage."""
    print("=" * 80)
    print("OFFLINE DATA COLLECTION EXAMPLE")
    print("=" * 80)

    # Create environment (you need actual data loaded)
    config = TradingEnvConfig(
        n_assets=20,
        state_dim=36,
        portfolio_dim=22,
    )

    env = TradingEnv(config)

    # Load data into environment
    # env.set_data(your_data)  # You need to provide actual market data

    # NOTE: This is just an example skeleton.
    # In practice, you would:
    # 1. Load real market data into the environment
    # 2. Run a heuristic policy (e.g., momentum, mean-reversion)
    # 3. Collect trajectories with timestamps
    # 4. Save to NPZ file

    print("\nTo collect real offline data:")
    print("1. Load market data into TradingEnv using env.set_data()")
    print("2. Ensure data includes 'timestamps' field")
    print("3. Run collect_trajectories_with_random_policy() or your own policy")
    print("4. Save using save_offline_data()")
    print("\nExample:")
    print("  data = collect_trajectories_with_random_policy(env, num_episodes=100)")
    print("  save_offline_data(data, Path('data/offline_rl_data.npz'))")

    print("\n" + "=" * 80)
    print("IMPORTANT: Always include timestamps in offline data!")
    print("=" * 80)


if __name__ == '__main__':
    main()
