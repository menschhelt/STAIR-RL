"""
Training Loop - Orchestrates the 3-phase training pipeline.

Phase 1: CQL-SAC Offline Pre-training (36 months historical data)
Phase 2: PPO-CVaR Online Fine-tuning (18 months recent data)
Phase 3: Event-Driven Adaptive Retraining (production)

This module provides:
- Data loading and preprocessing
- Training loop management
- Evaluation and logging
- Checkpointing
"""

import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from config.settings import Config, RLConfig
from environments.trading_env import TradingEnv, EnvConfig
from agents.cql_sac import CQLSACAgent, CQLSACConfig, ReplayBuffer
from agents.ppo_cvar import PPOCVaRAgent, PPOCVaRConfig, RolloutBuffer


@dataclass
class TrainerConfig:
    """Trainer configuration."""
    # Paths
    data_dir: Path = Path('data')
    log_dir: Path = Path('logs')
    checkpoint_dir: Path = Path('checkpoints')

    # Training
    seed: int = 42
    device: str = 'cuda'

    # Logging
    log_interval: int = 1000
    eval_interval: int = 10000
    save_interval: int = 50000

    # Evaluation
    eval_episodes: int = 10


class Trainer:
    """
    Base trainer class with common functionality.
    """

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        rl_config: Optional[RLConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Trainer configuration
            rl_config: RL training configuration
        """
        self.config = config or TrainerConfig()
        self.rl_config = rl_config or RLConfig()

        # Set up directories
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set random seeds
        self._set_seeds(self.config.seed)

        # Device
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

        # TensorBoard
        self.writer: Optional[SummaryWriter] = None

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_logging(self):
        """Set up logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _init_tensorboard(self, run_name: str):
        """Initialize TensorBoard writer."""
        log_path = self.config.log_dir / run_name
        self.writer = SummaryWriter(str(log_path))
        self.logger.info(f"TensorBoard logs: {log_path}")

    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """Log metrics to TensorBoard and console."""
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f'{prefix}/{key}', value, step)

        if step % self.config.log_interval == 0:
            metrics_str = ', '.join([f'{k}={v:.4f}' for k, v in metrics.items()])
            self.logger.info(f"Step {step}: {metrics_str}")


class Phase1Trainer(Trainer):
    """
    Phase 1: CQL-SAC Offline Pre-training.

    Trains the agent on historical data using Conservative Q-Learning
    to prevent overestimation on out-of-distribution actions.
    """

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        rl_config: Optional[RLConfig] = None,
    ):
        super().__init__(config, rl_config)

        # CQL-SAC specific config
        self.cql_config = CQLSACConfig(
            n_assets=20,
            state_dim=35,
            portfolio_dim=22,
            lr_actor=self.rl_config.cql_sac.learning_rate_actor,
            lr_critic=self.rl_config.cql_sac.learning_rate_critic,
            gamma=self.rl_config.cql_sac.gamma,
            tau=self.rl_config.cql_sac.tau,
            alpha_init=self.rl_config.cql_sac.alpha,
            lambda_cql=self.rl_config.cql_sac.lambda_cql,
            lambda_gp=self.rl_config.cql_sac.lambda_gp,
            batch_size=self.rl_config.cql_sac.batch_size,
        )

        # Initialize agent
        self.agent = CQLSACAgent(self.cql_config, device=str(self.device))

        # Replay buffer
        self.replay_buffer: Optional[ReplayBuffer] = None

    def load_offline_data(
        self,
        data_path: Path,
    ) -> int:
        """
        Load offline data into replay buffer.

        Args:
            data_path: Path to preprocessed data

        Returns:
            Number of transitions loaded
        """
        self.logger.info(f"Loading offline data from {data_path}")

        # Load data (expected format: numpy arrays)
        data = np.load(data_path, allow_pickle=True)

        market_states = data['market_states']  # (T, n_assets, state_dim)
        portfolio_states = data['portfolio_states']  # (T, portfolio_dim)
        actions = data['actions']  # (T, n_assets)
        rewards = data['rewards']  # (T,)
        dones = data['dones']  # (T,)

        T = len(rewards)
        self.logger.info(f"Loaded {T} transitions")

        # Initialize replay buffer
        buffer_size = min(T, self.rl_config.cql_sac.replay_buffer_size)
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            n_assets=self.cql_config.n_assets,
            state_dim=self.cql_config.state_dim,
            portfolio_dim=self.cql_config.portfolio_dim,
            device=str(self.device),
        )

        # Load data into buffer
        self.replay_buffer.load_from_data(
            market_states, portfolio_states, actions, rewards, dones
        )

        return len(self.replay_buffer)

    def train(
        self,
        total_steps: int,
        run_name: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Run Phase 1 offline training.

        Args:
            total_steps: Total training steps
            run_name: Name for this training run

        Returns:
            Dict of training metrics history
        """
        if self.replay_buffer is None or len(self.replay_buffer) == 0:
            raise RuntimeError("No data loaded. Call load_offline_data() first.")

        run_name = run_name or f"phase1_cql_sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._init_tensorboard(run_name)

        self.logger.info(f"Starting Phase 1 training: {total_steps} steps")
        self.logger.info(f"Replay buffer size: {len(self.replay_buffer)}")

        # Training history
        history = {
            'critic_loss': [],
            'actor_loss': [],
            'cql_loss': [],
            'alpha': [],
        }

        start_time = time.time()

        for step in range(1, total_steps + 1):
            # Sample batch and update
            batch = self.replay_buffer.sample(self.cql_config.batch_size)
            metrics = self.agent.update(batch)

            # Record metrics
            for key in history:
                if key in metrics:
                    history[key].append(metrics[key])

            # Log
            self._log_metrics(metrics, step, prefix='train')

            # Save checkpoint
            if step % self.config.save_interval == 0:
                self._save_checkpoint(step, run_name)

        # Final save
        self._save_checkpoint(total_steps, run_name, final=True)

        elapsed = time.time() - start_time
        self.logger.info(f"Phase 1 training completed in {elapsed/3600:.2f} hours")

        if self.writer:
            self.writer.close()

        return history

    def _save_checkpoint(self, step: int, run_name: str, final: bool = False):
        """Save training checkpoint."""
        suffix = 'final' if final else f'step_{step}'
        path = self.config.checkpoint_dir / f'{run_name}_{suffix}.pt'
        self.agent.save(str(path))
        self.logger.info(f"Saved checkpoint: {path}")


class Phase2Trainer(Trainer):
    """
    Phase 2: PPO-CVaR Online Fine-tuning.

    Fine-tunes the pre-trained agent using online interaction
    with CVaR constraint for risk management.
    """

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        rl_config: Optional[RLConfig] = None,
    ):
        super().__init__(config, rl_config)

        # PPO-CVaR specific config
        self.ppo_config = PPOCVaRConfig(
            n_assets=20,
            state_dim=35,
            portfolio_dim=22,
            lr=self.rl_config.ppo_cvar.learning_rate,
            gamma=self.rl_config.ppo_cvar.gamma,
            gae_lambda=self.rl_config.ppo_cvar.gae_lambda,
            clip_epsilon=self.rl_config.ppo_cvar.clip_epsilon,
            ppo_epochs=self.rl_config.ppo_cvar.ppo_epochs,
            batch_size=self.rl_config.ppo_cvar.batch_size,
            horizon=self.rl_config.ppo_cvar.horizon,
            alpha_cvar=self.rl_config.ppo_cvar.alpha_cvar,
            kappa=self.rl_config.ppo_cvar.kappa,
        )

        # Initialize agent
        self.agent = PPOCVaRAgent(self.ppo_config, device=str(self.device))

        # Environment
        self.env: Optional[TradingEnv] = None

        # Rollout buffer
        self.rollout_buffer: Optional[RolloutBuffer] = None

    def load_pretrained(self, path: Path):
        """Load pre-trained CQL-SAC agent weights."""
        self.logger.info(f"Loading pre-trained weights from {path}")

        # Load CQL-SAC agent
        cql_agent = CQLSACAgent(device=str(self.device))
        cql_agent.load(str(path))

        # Transfer weights
        self.agent.load_from_cql_sac(cql_agent)

    def set_environment(self, env: TradingEnv):
        """Set the training environment."""
        self.env = env

    def train(
        self,
        total_steps: int,
        run_name: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Run Phase 2 online training.

        Args:
            total_steps: Total training steps
            run_name: Name for this training run

        Returns:
            Dict of training metrics history
        """
        if self.env is None:
            raise RuntimeError("No environment set. Call set_environment() first.")

        run_name = run_name or f"phase2_ppo_cvar_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._init_tensorboard(run_name)

        self.logger.info(f"Starting Phase 2 training: {total_steps} steps")

        # Initialize rollout buffer
        self.rollout_buffer = RolloutBuffer(
            horizon=self.ppo_config.horizon,
            n_assets=self.ppo_config.n_assets,
            state_dim=self.ppo_config.state_dim,
            portfolio_dim=self.ppo_config.portfolio_dim,
            device=str(self.device),
        )

        # Training history
        history = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'cvar': [],
            'lambda_cvar': [],
            'episode_return': [],
        }

        # Reset environment
        obs, info = self.env.reset()
        market_state = obs['market']
        portfolio_state = obs['portfolio']

        episode_return = 0
        n_episodes = 0
        total_steps_done = 0

        start_time = time.time()

        while total_steps_done < total_steps:
            # Collect rollout
            self.rollout_buffer.reset()

            for _ in range(self.ppo_config.horizon):
                # Select action
                action, log_prob, value = self.agent.select_action(
                    market_state, portfolio_state
                )

                # Environment step
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # Store transition
                self.rollout_buffer.add(
                    market_state=market_state,
                    portfolio_state=portfolio_state,
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_prob,
                    done=terminated,
                )

                episode_return += reward
                total_steps_done += 1

                if terminated or truncated:
                    # Log episode
                    history['episode_return'].append(episode_return)
                    self._log_metrics({'episode_return': episode_return}, total_steps_done, 'episode')

                    # Reset
                    obs, info = self.env.reset()
                    episode_return = 0
                    n_episodes += 1
                else:
                    next_market_state = next_obs['market']
                    next_portfolio_state = next_obs['portfolio']

                market_state = next_obs['market']
                portfolio_state = next_obs['portfolio']

            # Compute advantages
            last_value = self.agent.get_value(market_state, portfolio_state)
            self.rollout_buffer.compute_gae(
                last_value,
                gamma=self.ppo_config.gamma,
                gae_lambda=self.ppo_config.gae_lambda,
            )

            # Update agent
            metrics = self.agent.update(self.rollout_buffer)

            # Record metrics
            for key in ['policy_loss', 'value_loss', 'entropy', 'cvar', 'lambda_cvar']:
                if key in metrics:
                    history[key].append(metrics[key])

            self._log_metrics(metrics, total_steps_done, 'train')

            # Save checkpoint
            if total_steps_done % self.config.save_interval == 0:
                self._save_checkpoint(total_steps_done, run_name)

        # Final save
        self._save_checkpoint(total_steps_done, run_name, final=True)

        elapsed = time.time() - start_time
        self.logger.info(f"Phase 2 training completed in {elapsed/3600:.2f} hours")
        self.logger.info(f"Episodes completed: {n_episodes}")

        if self.writer:
            self.writer.close()

        return history

    def evaluate(
        self,
        n_episodes: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate current policy.

        Args:
            n_episodes: Number of evaluation episodes

        Returns:
            Dict of evaluation metrics
        """
        if self.env is None:
            raise RuntimeError("No environment set.")

        returns = []
        sharpe_ratios = []
        max_drawdowns = []

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            done = False

            while not done:
                market_state = obs['market']
                portfolio_state = obs['portfolio']

                action, _, _ = self.agent.select_action(
                    market_state, portfolio_state, deterministic=True
                )

                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

            # Get episode stats
            stats = self.env.get_episode_stats()
            returns.append(stats.get('total_return', 0))
            sharpe_ratios.append(stats.get('sharpe_ratio', 0))
            max_drawdowns.append(stats.get('max_drawdown', 0))

        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_sharpe': np.mean(sharpe_ratios),
            'mean_max_dd': np.mean(max_drawdowns),
        }

    def _save_checkpoint(self, step: int, run_name: str, final: bool = False):
        """Save training checkpoint."""
        suffix = 'final' if final else f'step_{step}'
        path = self.config.checkpoint_dir / f'{run_name}_{suffix}.pt'
        self.agent.save(str(path))
        self.logger.info(f"Saved checkpoint: {path}")


# ========== Utility Functions ==========

def create_data_splits(
    data: Dict[str, np.ndarray],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[Dict, Dict, Dict]:
    """
    Split data into train/validation/test sets.

    Args:
        data: Dict of numpy arrays
        train_ratio: Fraction for training
        val_ratio: Fraction for validation

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    T = len(data['rewards'])
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    def slice_data(start: int, end: int) -> Dict:
        return {k: v[start:end] for k, v in data.items()}

    return (
        slice_data(0, train_end),
        slice_data(train_end, val_end),
        slice_data(val_end, T),
    )


# ========== Standalone Testing ==========

if __name__ == '__main__':
    print("Testing Training Module...")

    # Test configuration
    trainer_config = TrainerConfig(
        log_dir=Path('/tmp/stair_logs'),
        checkpoint_dir=Path('/tmp/stair_checkpoints'),
        seed=42,
        device='cpu',
    )

    rl_config = RLConfig()

    # Test Phase 1 Trainer initialization
    phase1 = Phase1Trainer(trainer_config, rl_config)
    print(f"Phase 1 Trainer initialized")
    print(f"  Device: {phase1.device}")
    print(f"  CQL-SAC config: batch_size={phase1.cql_config.batch_size}")

    # Test Phase 2 Trainer initialization
    phase2 = Phase2Trainer(trainer_config, rl_config)
    print(f"Phase 2 Trainer initialized")
    print(f"  PPO-CVaR config: horizon={phase2.ppo_config.horizon}")

    print("\nTraining module test passed!")
