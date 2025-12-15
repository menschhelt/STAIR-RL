#!/usr/bin/env python3
"""
Minimal Training Test for STAIR-RL

Tests both Phase 1 (CQL-SAC) and Phase 2 (PPO-CVaR) training
with minimal steps to validate the training pipeline.

Usage:
    # Phase 1 (CQL-SAC offline)
    python scripts/test_training_minimal.py \
        --phase 1 \
        --data /home/work/data/stair-local/test_mock/tensors/train_data.npz \
        --steps 1000 \
        --checkpoint-dir /tmp/stair_test_checkpoints/phase1

    # Phase 2 (PPO-CVaR online)
    python scripts/test_training_minimal.py \
        --phase 2 \
        --data /home/work/data/stair-local/test_mock/tensors/val_data.npz \
        --steps 500 \
        --pretrained /tmp/stair_test_checkpoints/phase1/cql_sac_final.pt \
        --checkpoint-dir /tmp/stair_test_checkpoints/phase2
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.trading_env import TradingEnv, EnvConfig
from agents.cql_sac import CQLSACAgent, CQLSACConfig, ReplayBuffer
from agents.ppo_cvar import PPOCVaRAgent, PPOCVaRConfig, RolloutBuffer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinimalPhase1Tester:
    """
    Minimal Phase 1 (CQL-SAC) training tester.
    """

    def __init__(
        self,
        data_path: Path,
        checkpoint_dir: Path,
        n_assets: int = 10,
        state_dim: int = 36,
        steps: int = 1000,
        batch_size: int = 64,
        device: str = 'cpu',
    ):
        """
        Initialize Phase 1 tester.

        Args:
            data_path: Path to training data .npz file
            checkpoint_dir: Directory for checkpoints
            n_assets: Number of assets
            state_dim: State dimension per asset
            steps: Number of training steps
            batch_size: Batch size
            device: Device ('cpu' or 'cuda')
        """
        self.data_path = data_path
        self.checkpoint_dir = checkpoint_dir
        self.n_assets = n_assets
        self.state_dim = state_dim
        self.steps = steps
        self.batch_size = batch_size
        self.device = device

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized Phase 1 Tester")
        logger.info(f"  Data: {data_path}")
        logger.info(f"  Steps: {steps}")
        logger.info(f"  Device: {device}")

    def load_data(self) -> Dict[str, np.ndarray]:
        """
        Load training data from .npz file.

        Returns:
            Data dict with states, returns, prices, timestamps
        """
        logger.info(f"Loading data from {self.data_path}...")

        data = np.load(self.data_path, allow_pickle=True)

        # Convert timestamp strings back to datetime if needed
        timestamps = data['timestamps']
        if timestamps.dtype.kind == 'U':  # Unicode strings
            from datetime import datetime
            timestamps = [datetime.fromisoformat(ts) for ts in timestamps]

        loaded_data = {
            'states': data['states'],
            'returns': data['returns'],
            'prices': data['prices'],
            'funding_rates': data['funding_rates'],
            'timestamps': timestamps,
        }

        T, N, D = loaded_data['states'].shape
        logger.info(f"  Loaded: T={T}, N={N}, D={D}")

        # Validate
        assert N == self.n_assets, f"Asset mismatch: {N} vs {self.n_assets}"
        assert D == self.state_dim, f"State dim mismatch: {D} vs {self.state_dim}"

        return loaded_data

    def collect_offline_data(
        self,
        env: TradingEnv,
        replay_buffer: ReplayBuffer,
        n_transitions: int = 10000,
    ):
        """
        Collect offline data with random policy.

        Args:
            env: Trading environment
            replay_buffer: Replay buffer to fill
            n_transitions: Number of transitions to collect
        """
        logger.info(f"Collecting {n_transitions} transitions with random policy...")

        obs, _ = env.reset()
        transitions_collected = 0

        pbar = tqdm(total=n_transitions, desc="Collecting data")

        while transitions_collected < n_transitions:
            # Random action
            action = env.action_space.sample()

            # Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            replay_buffer.add(
                obs['market'],
                obs['portfolio'],
                action,
                reward,
                next_obs['market'],
                next_obs['portfolio'],
                done,
            )

            transitions_collected += 1
            pbar.update(1)

            # Move to next state
            obs = next_obs

            # Reset if done
            if done:
                obs, _ = env.reset()

        pbar.close()
        logger.info(f"  Collected {transitions_collected} transitions")

    def train(self) -> Dict[str, float]:
        """
        Run Phase 1 training.

        Returns:
            Final metrics dict
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: CQL-SAC OFFLINE TRAINING")
        logger.info("=" * 60)

        # Load data
        data = self.load_data()

        # Create environment
        env_config = EnvConfig(
            n_assets=self.n_assets,
            state_dim=self.state_dim,
        )
        env = TradingEnv(config=env_config)
        env.set_data(data)

        # Reset environment to validate
        obs, info = env.reset()
        logger.info(f"Environment initialized:")
        logger.info(f"  Market shape: {obs['market'].shape}")
        logger.info(f"  Portfolio shape: {obs['portfolio'].shape}")

        # Create agent
        agent_config = CQLSACConfig(
            n_assets=self.n_assets,
            state_dim=self.state_dim,
            portfolio_dim=self.n_assets + 2,
            batch_size=self.batch_size,
        )
        agent = CQLSACAgent(agent_config, device=self.device)

        # Create replay buffer
        replay_buffer = ReplayBuffer(
            capacity=50000,  # Reduced for testing
            n_assets=self.n_assets,
            state_dim=self.state_dim,
            portfolio_dim=self.n_assets + 2,
            device=self.device,
        )

        # Collect offline data
        self.collect_offline_data(env, replay_buffer, n_transitions=10000)

        # Training loop
        logger.info(f"Training for {self.steps} steps...")

        metrics_history = []

        for step in tqdm(range(self.steps), desc="Training"):
            # Sample batch
            batch = replay_buffer.sample(self.batch_size)

            # Train step
            metrics = agent.update(batch)

            # Validate metrics
            assert np.isfinite(metrics['critic_loss']), f"Critic loss is NaN at step {step}"
            assert np.isfinite(metrics['actor_loss']), f"Actor loss is NaN at step {step}"
            assert np.isfinite(metrics['cql_loss']), f"CQL loss is NaN at step {step}"

            metrics_history.append(metrics)

            # Log periodically
            if step % 100 == 0 and step > 0:
                avg_metrics = {
                    key: np.mean([m[key] for m in metrics_history[-100:]])
                    for key in metrics.keys()
                }
                logger.info(
                    f"Step {step}/{self.steps} | "
                    f"Critic: {avg_metrics['critic_loss']:.4f} | "
                    f"Actor: {avg_metrics['actor_loss']:.4f} | "
                    f"CQL: {avg_metrics['cql_loss']:.4f}"
                )

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / 'cql_sac_final.pt'
        agent.save(checkpoint_path)
        logger.info(f"✓ Saved checkpoint: {checkpoint_path}")

        # Return final metrics
        final_metrics = {
            key: np.mean([m[key] for m in metrics_history[-100:]])
            for key in metrics_history[-1].keys()
        }

        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1 COMPLETE")
        logger.info("=" * 60)
        for key, value in final_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        return final_metrics


class MinimalPhase2Tester:
    """
    Minimal Phase 2 (PPO-CVaR) training tester.
    """

    def __init__(
        self,
        data_path: Path,
        checkpoint_dir: Path,
        pretrained_path: Path = None,
        n_assets: int = 10,
        state_dim: int = 36,
        steps: int = 500,
        horizon: int = 512,
        device: str = 'cpu',
    ):
        """
        Initialize Phase 2 tester.

        Args:
            data_path: Path to validation data .npz file
            checkpoint_dir: Directory for checkpoints
            pretrained_path: Path to pretrained Phase 1 checkpoint (optional)
            n_assets: Number of assets
            state_dim: State dimension per asset
            steps: Number of training steps
            horizon: Rollout horizon
            device: Device ('cpu' or 'cuda')
        """
        self.data_path = data_path
        self.checkpoint_dir = checkpoint_dir
        self.pretrained_path = pretrained_path
        self.n_assets = n_assets
        self.state_dim = state_dim
        self.steps = steps
        self.horizon = horizon
        self.device = device

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized Phase 2 Tester")
        logger.info(f"  Data: {data_path}")
        logger.info(f"  Steps: {steps}")
        logger.info(f"  Horizon: {horizon}")
        logger.info(f"  Device: {device}")

    def load_data(self) -> Dict[str, np.ndarray]:
        """Load validation data."""
        logger.info(f"Loading data from {self.data_path}...")

        data = np.load(self.data_path, allow_pickle=True)

        # Convert timestamp strings
        timestamps = data['timestamps']
        if timestamps.dtype.kind == 'U':
            from datetime import datetime
            timestamps = [datetime.fromisoformat(ts) for ts in timestamps]

        loaded_data = {
            'states': data['states'],
            'returns': data['returns'],
            'prices': data['prices'],
            'funding_rates': data['funding_rates'],
            'timestamps': timestamps,
        }

        T, N, D = loaded_data['states'].shape
        logger.info(f"  Loaded: T={T}, N={N}, D={D}")

        return loaded_data

    def collect_rollout(
        self,
        env: TradingEnv,
        agent: PPOCVaRAgent,
        rollout_buffer: RolloutBuffer,
    ) -> Tuple[float, Dict]:
        """
        Collect one rollout.

        Args:
            env: Trading environment
            agent: PPO-CVaR agent
            rollout_buffer: Rollout buffer

        Returns:
            Tuple of (episode_reward, episode_info)
        """
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_returns = []

        for _ in range(self.horizon):
            # Select action
            action, log_prob, value = agent.select_action(obs, deterministic=False)

            # Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            rollout_buffer.add(
                obs['market'],
                obs['portfolio'],
                action,
                reward,
                done,
                value,
                log_prob,
            )

            episode_reward += reward
            episode_returns.append(info.get('port_return', 0.0))

            obs = next_obs

            if done:
                break

        episode_info = {
            'total_reward': episode_reward,
            'sharpe': np.mean(episode_returns) / (np.std(episode_returns) + 1e-8) * np.sqrt(252),
        }

        return episode_reward, episode_info

    def train(self) -> Dict[str, float]:
        """
        Run Phase 2 training.

        Returns:
            Final metrics dict
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: PPO-CVaR ONLINE TRAINING")
        logger.info("=" * 60)

        # Load data
        data = self.load_data()

        # Create environment
        env_config = EnvConfig(
            n_assets=self.n_assets,
            state_dim=self.state_dim,
        )
        env = TradingEnv(config=env_config)
        env.set_data(data)

        # Reset environment
        obs, info = env.reset()
        logger.info(f"Environment initialized:")
        logger.info(f"  Market shape: {obs['market'].shape}")
        logger.info(f"  Portfolio shape: {obs['portfolio'].shape}")

        # Create agent
        agent_config = PPOCVaRConfig(
            n_assets=self.n_assets,
            state_dim=self.state_dim,
            portfolio_dim=self.n_assets + 2,
        )
        agent = PPOCVaRAgent(agent_config, device=self.device)

        # Load pretrained weights if provided
        if self.pretrained_path and self.pretrained_path.exists():
            logger.info(f"Loading pretrained weights from {self.pretrained_path}")
            # Note: This would need proper weight transfer logic
            # For now, we skip it in minimal test

        # Create rollout buffer
        rollout_buffer = RolloutBuffer(
            capacity=self.horizon * 2,
            n_assets=self.n_assets,
            state_dim=self.state_dim,
            portfolio_dim=self.n_assets + 2,
            device=self.device,
        )

        # Training loop
        logger.info(f"Training for {self.steps} steps...")

        metrics_history = []
        episode_count = 0

        for step in tqdm(range(self.steps), desc="Training"):
            # Collect rollout
            episode_reward, episode_info = self.collect_rollout(env, agent, rollout_buffer)
            episode_count += 1

            # Train on collected data
            if rollout_buffer.size() >= self.horizon:
                # Get rollout data
                rollout_data = rollout_buffer.get()

                # Compute returns and advantages
                agent.compute_returns_and_advantages(
                    rollout_data['rewards'],
                    rollout_data['dones'],
                    rollout_data['values'],
                )

                # Update policy
                metrics = agent.update(rollout_data)

                # Validate metrics
                assert np.isfinite(metrics['policy_loss']), f"Policy loss is NaN at step {step}"
                assert np.isfinite(metrics['value_loss']), f"Value loss is NaN at step {step}"

                metrics['episode_reward'] = episode_reward
                metrics['sharpe'] = episode_info['sharpe']
                metrics_history.append(metrics)

                # Clear buffer
                rollout_buffer.clear()

            # Log periodically
            if step % 50 == 0 and len(metrics_history) > 0:
                recent_metrics = metrics_history[-10:]
                avg_reward = np.mean([m['episode_reward'] for m in recent_metrics])
                avg_sharpe = np.mean([m['sharpe'] for m in recent_metrics])
                logger.info(
                    f"Step {step}/{self.steps} | "
                    f"Episodes: {episode_count} | "
                    f"Reward: {avg_reward:.4f} | "
                    f"Sharpe: {avg_sharpe:.2f}"
                )

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / 'ppo_cvar_final.pt'
        agent.save(checkpoint_path)
        logger.info(f"✓ Saved checkpoint: {checkpoint_path}")

        # Return final metrics
        if metrics_history:
            final_metrics = {
                key: np.mean([m[key] for m in metrics_history[-10:]])
                for key in metrics_history[-1].keys()
            }
        else:
            final_metrics = {}

        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2 COMPLETE")
        logger.info("=" * 60)
        for key, value in final_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        return final_metrics


def main():
    parser = argparse.ArgumentParser(description='Test training pipeline')
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2],
                        help='Training phase (1=CQL-SAC, 2=PPO-CVaR)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data .npz file')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Checkpoint directory')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained checkpoint (Phase 2 only)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of training steps')
    parser.add_argument('--n-assets', type=int, default=10,
                        help='Number of assets')
    parser.add_argument('--state-dim', type=int, default=36,
                        help='State dimension per asset')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')

    args = parser.parse_args()

    data_path = Path(args.data)
    checkpoint_dir = Path(args.checkpoint_dir)

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return 1

    if args.phase == 1:
        # Phase 1: CQL-SAC
        tester = MinimalPhase1Tester(
            data_path=data_path,
            checkpoint_dir=checkpoint_dir,
            n_assets=args.n_assets,
            state_dim=args.state_dim,
            steps=args.steps,
            device=args.device,
        )
        metrics = tester.train()

    else:
        # Phase 2: PPO-CVaR
        pretrained_path = Path(args.pretrained) if args.pretrained else None

        tester = MinimalPhase2Tester(
            data_path=data_path,
            checkpoint_dir=checkpoint_dir,
            pretrained_path=pretrained_path,
            n_assets=args.n_assets,
            state_dim=args.state_dim,
            steps=args.steps,
            device=args.device,
        )
        metrics = tester.train()

    logger.info("\n✓ Test completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
