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

# Theory validation modules (Paper Theorem 2: PAC Bound)
from evaluation.h_divergence import HdivergenceMonitor, compare_with_paper_claims

# Additional imports for comprehensive TensorBoard logging
from collections import deque


class PortfolioMetricsTracker:
    """
    Tracks portfolio performance metrics for TensorBoard logging.

    Metrics tracked:
    - Rolling Sharpe ratio
    - Maximum drawdown
    - Turnover
    - CVaR and violation rate
    """

    def __init__(self, window_size: int = 252, alpha_cvar: float = 0.95):
        """
        Initialize portfolio metrics tracker.

        Args:
            window_size: Rolling window for Sharpe calculation
            alpha_cvar: Confidence level for CVaR (default 95%)
        """
        self.window_size = window_size
        self.alpha_cvar = alpha_cvar

        # Rolling buffers
        self.returns_buffer = deque(maxlen=window_size)
        self.weights_history = deque(maxlen=2)  # For turnover
        self.cumulative_returns = []

        # CVaR tracking
        self.cvar_threshold = 0.05  # 5% threshold (paper: kappa)
        self.cvar_violations = 0
        self.cvar_total_checks = 0

    def update(self, ret: float, weights: Optional[np.ndarray] = None):
        """
        Update metrics with new return and weights.

        Args:
            ret: Single period return
            weights: Current portfolio weights (optional, for turnover)
        """
        self.returns_buffer.append(ret)

        if weights is not None:
            self.weights_history.append(weights.copy())

        # Track cumulative return for drawdown
        if len(self.cumulative_returns) == 0:
            self.cumulative_returns.append(1 + ret)
        else:
            self.cumulative_returns.append(self.cumulative_returns[-1] * (1 + ret))

    def get_rolling_sharpe(self, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized rolling Sharpe ratio."""
        if len(self.returns_buffer) < 20:
            return 0.0

        returns = np.array(self.returns_buffer)
        excess_returns = returns - risk_free_rate / 252  # Daily adjustment

        mean_ret = np.mean(excess_returns)
        std_ret = np.std(excess_returns, ddof=1)

        if std_ret < 1e-8:
            return 0.0

        # Annualize (assuming daily returns)
        sharpe = (mean_ret / std_ret) * np.sqrt(252)
        return float(sharpe)

    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        if len(self.cumulative_returns) < 2:
            return 0.0

        cum_returns = np.array(self.cumulative_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max

        return float(np.min(drawdowns))

    def get_turnover(self) -> float:
        """Calculate portfolio turnover (last rebalance)."""
        if len(self.weights_history) < 2:
            return 0.0

        prev_weights = self.weights_history[-2]
        curr_weights = self.weights_history[-1]

        # Turnover = sum of absolute weight changes / 2
        turnover = np.sum(np.abs(curr_weights - prev_weights)) / 2
        return float(turnover)

    def get_cvar(self) -> float:
        """Calculate empirical CVaR from returns buffer."""
        if len(self.returns_buffer) < 20:
            return 0.0

        returns = np.array(self.returns_buffer)
        losses = -returns  # Convert to losses

        sorted_losses = np.sort(losses)
        cutoff = int(len(sorted_losses) * (1 - self.alpha_cvar))
        cutoff = max(1, cutoff)

        cvar = sorted_losses[-cutoff:].mean()

        # Track violations
        self.cvar_total_checks += 1
        if cvar > self.cvar_threshold:
            self.cvar_violations += 1

        return float(cvar)

    def get_cvar_violation_rate(self) -> float:
        """Get rate of CVaR threshold violations."""
        if self.cvar_total_checks == 0:
            return 0.0
        return self.cvar_violations / self.cvar_total_checks

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all portfolio metrics for logging."""
        return {
            'sharpe_rolling': self.get_rolling_sharpe(),
            'max_drawdown': self.get_max_drawdown(),
            'turnover': self.get_turnover(),
            'cvar_95': self.get_cvar(),
            'cvar_violation_rate': self.get_cvar_violation_rate(),
        }


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
    save_interval: int = 100  # Save every 10000 steps for checkpoint resume

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

    def _get_gpu_suffix(self) -> str:
        """Get GPU ID suffix for run_name differentiation."""
        if torch.cuda.is_available():
            # Get current CUDA device index
            gpu_id = torch.cuda.current_device()
            return f"_gpu{gpu_id}"
        return "_cpu"

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
            state_dim=36,  # BUG FIX: Changed from 35 to match StateBuilder output
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

        # Theory validation: H-divergence monitor (Paper Theorem 2)
        self.h_div_monitor = HdivergenceMonitor(
            buffer_size=10000,
            compute_interval=10000,  # Compute every 10k steps
            alpha=0.95,  # CVaR confidence level
            kappa=0.05,  # CVaR threshold (5%)
        )

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
        timestamps = data.get('timestamps', None)  # (T,) ISO strings, optional for backward compat

        T = len(rewards)
        self.logger.info(f"Loaded {T} transitions")
        if timestamps is not None:
            self.logger.info(f"Loaded timestamps: {timestamps[0]} to {timestamps[-1]}")

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
            market_states, portfolio_states, actions, rewards, dones,
            timestamps=timestamps
        )

        return len(self.replay_buffer)

    def train(
        self,
        total_steps: int,
        run_name: Optional[str] = None,
        resume: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Run Phase 1 offline training.

        Args:
            total_steps: Total training steps
            run_name: Name for this training run
            resume: If True, resume from model.pt if it exists

        Returns:
            Dict of training metrics history
        """
        if self.replay_buffer is None or len(self.replay_buffer) == 0:
            raise RuntimeError("No data loaded. Call load_offline_data() first.")

        run_name = run_name or f"phase1_cql_sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}{self._get_gpu_suffix()}"
        self._init_tensorboard(run_name)

        # Check for existing checkpoint to resume (run_name 기반)
        start_step = 0
        if resume:
            checkpoint_path = self.find_checkpoint(run_name)
            if checkpoint_path:
                start_step = self.load_checkpoint(checkpoint_path)
                self.logger.info(f"Resuming training from step {start_step}")

        self.logger.info(f"Starting Phase 1 training: {total_steps} steps (starting from {start_step})")
        self.logger.info(f"Replay buffer size: {len(self.replay_buffer)}")

        # Training history
        history = {
            'critic_loss': [],
            'actor_loss': [],
            'cql_loss': [],
            'alpha': [],
            'd_H': [],  # H-divergence for Theorem 2 validation
        }

        start_time = time.time()

        for step in range(start_step + 1, total_steps + 1):
            # Sample batch and update
            batch = self.replay_buffer.sample(self.cql_config.batch_size)
            metrics = self.agent.update(batch)

            # Collect offline states for H-divergence (Paper Theorem 2)
            # Flatten market states: (B, N, D) -> (B, N*D)
            market_states = batch['market_states']
            if isinstance(market_states, torch.Tensor):
                market_states = market_states.cpu().numpy()
            flat_states = market_states.reshape(market_states.shape[0], -1)
            self.h_div_monitor.add_offline_states(flat_states)

            # Record metrics
            for key in history:
                if key in metrics:
                    history[key].append(metrics[key])

            # Log H-divergence metrics periodically
            if step % 10000 == 0:
                h_div_stats = self.h_div_monitor.get_statistics()
                if h_div_stats:
                    d_H = h_div_stats.get('d_H_latest', 0.0)
                    history['d_H'].append(d_H)
                    self._log_metrics({'d_H': d_H}, step, prefix='theory')

                    # Compare with paper claims
                    comparison = compare_with_paper_claims(d_H, method='CQL')
                    self.logger.info(
                        f"Step {step} H-divergence: d_H={d_H:.4f} "
                        f"(paper: {comparison['paper_claim_mean']:.2f}±{comparison['paper_claim_std']:.2f}, "
                        f"z={comparison['z_score']:.2f})"
                    )

            # Log
            self._log_metrics(metrics, step, prefix='train')

            # Save checkpoint every save_interval steps
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
        """Save training checkpoint with full state for resume."""
        suffix = 'final' if final else f'step_{step}'
        path = self.config.checkpoint_dir / f'{run_name}_{suffix}.pt'

        # Save comprehensive checkpoint for resume
        checkpoint = {
            'step': step,
            'run_name': run_name,
            'encoder': self.agent.encoder.state_dict(),
            'actor': self.agent.actor.state_dict(),
            'critic': self.agent.critic.state_dict(),
            'target_critic': self.agent.target_critic.state_dict(),
            'log_alpha': self.agent.log_alpha,
            'actor_optimizer': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent.critic_optimizer.state_dict(),
            'config': self.agent.config,
            'total_steps': self.agent.total_steps,
        }
        if self.agent.config.auto_entropy_tuning:
            checkpoint['alpha_optimizer'] = self.agent.alpha_optimizer.state_dict()

        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

        # Also save as {run_name}_model.pt for easy resume (run_name으로 구분)
        model_path = self.config.checkpoint_dir / f'{run_name}_model.pt'
        torch.save(checkpoint, model_path)
        self.logger.info(f"Saved latest model: {model_path}")

    def load_checkpoint(self, path: Path) -> int:
        """Load checkpoint and return the step to resume from."""
        self.logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.agent.encoder.load_state_dict(checkpoint['encoder'])
        self.agent.actor.load_state_dict(checkpoint['actor'])
        self.agent.critic.load_state_dict(checkpoint['critic'])
        self.agent.target_critic.load_state_dict(checkpoint['target_critic'])
        self.agent.log_alpha = checkpoint['log_alpha']
        self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.agent.total_steps = checkpoint['total_steps']

        if self.agent.config.auto_entropy_tuning and 'alpha_optimizer' in checkpoint:
            self.agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

        step = checkpoint.get('step', 0)
        self.logger.info(f"Resumed from step {step}")
        return step

    def find_checkpoint(self, run_name: str) -> Optional[Path]:
        """Find checkpoint file for given run_name."""
        # 1. run_name 전용 체크포인트
        model_path = self.config.checkpoint_dir / f'{run_name}_model.pt'
        if model_path.exists():
            return model_path

        # 2. 기존 model.pt (하위 호환)
        legacy_path = self.config.checkpoint_dir / 'model.pt'
        if legacy_path.exists():
            return legacy_path

        return None


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
            state_dim=36,  # BUG FIX: Changed from 35 to match StateBuilder output
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

        # Theory validation: H-divergence monitor (Paper Theorem 2)
        # Tracks distribution shift between offline (Phase 1) and online (Phase 2)
        self.h_div_monitor = HdivergenceMonitor(
            buffer_size=10000,
            compute_interval=5000,  # Compute every 5k steps (more frequent for online)
            alpha=0.95,
            kappa=0.05,
        )

        # Portfolio metrics tracker for TensorBoard
        self.portfolio_tracker = PortfolioMetricsTracker(
            window_size=252,
            alpha_cvar=0.95,
        )

    def load_pretrained(self, path: Path):
        """Load pre-trained CQL-SAC agent weights."""
        self.logger.info(f"Loading pre-trained weights from {path}")

        # Load CQL-SAC agent
        cql_agent = CQLSACAgent(device=str(self.device))
        cql_agent.load(str(path))

        # Transfer weights
        self.agent.load_from_cql_sac(cql_agent)

    def load_offline_states_for_h_divergence(self, data_path: Path, n_samples: int = 10000):
        """
        Load offline states from Phase1 data for H-divergence comparison.

        This enables Theorem 2 validation by comparing offline (Phase1) vs online (Phase2)
        state distributions.

        Args:
            data_path: Path to offline data (same as Phase1)
            n_samples: Number of samples to load for comparison
        """
        self.logger.info(f"Loading offline states for H-divergence from {data_path}")

        data = np.load(data_path, allow_pickle=True)
        market_states = data['market_states']  # (T, n_assets, state_dim)

        # Sample randomly if dataset is larger than n_samples
        T = len(market_states)
        if T > n_samples:
            indices = np.random.choice(T, n_samples, replace=False)
            market_states = market_states[indices]

        # Flatten and add to offline buffer
        flat_states = market_states.reshape(market_states.shape[0], -1)
        self.h_div_monitor.add_offline_states(flat_states)

        self.logger.info(f"Loaded {len(flat_states)} offline states for H-divergence baseline")

    def set_environment(self, env: TradingEnv):
        """Set the training environment."""
        self.env = env

    def train(
        self,
        total_steps: int,
        run_name: Optional[str] = None,
        resume: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Run Phase 2 online training.

        Args:
            total_steps: Total training steps
            run_name: Name for this training run
            resume: If True, resume from ppo_model.pt if it exists

        Returns:
            Dict of training metrics history
        """
        if self.env is None:
            raise RuntimeError("No environment set. Call set_environment() first.")

        run_name = run_name or f"phase2_ppo_cvar_{datetime.now().strftime('%Y%m%d_%H%M%S')}{self._get_gpu_suffix()}"
        self._init_tensorboard(run_name)

        # Check for existing checkpoint to resume (run_name 기반)
        start_step = 0
        if resume:
            checkpoint_path = self.find_checkpoint(run_name)
            if checkpoint_path:
                start_step = self.load_checkpoint(checkpoint_path)
                self.logger.info(f"Resuming training from step {start_step}")

        self.logger.info(f"Starting Phase 2 training: {total_steps} steps (starting from {start_step})")

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
            'd_H': [],  # H-divergence for Theorem 2 validation
        }

        # Reset environment
        obs, info = self.env.reset()
        market_state = obs['market']
        portfolio_state = obs['portfolio']

        episode_return = 0
        n_episodes = 0
        total_steps_done = start_step

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

                # Collect online states for H-divergence (Paper Theorem 2)
                # Flatten market state: (N, D) -> (N*D,)
                flat_state = market_state.flatten().reshape(1, -1)
                self.h_div_monitor.add_online_states(flat_state)

                episode_return += reward
                total_steps_done += 1

                # Update portfolio metrics tracker
                self.portfolio_tracker.update(reward, weights=action)

                if terminated or truncated:
                    # Log episode
                    history['episode_return'].append(episode_return)
                    self._log_metrics({'episode_return': episode_return}, total_steps_done, 'episode')

                    # Log portfolio metrics (theory/cvar and portfolio/)
                    portfolio_metrics = self.portfolio_tracker.get_all_metrics()
                    self._log_metrics({
                        'cvar_95': portfolio_metrics['cvar_95'],
                        'cvar_violation_rate': portfolio_metrics['cvar_violation_rate'],
                    }, total_steps_done, 'theory')
                    self._log_metrics({
                        'sharpe_rolling': portfolio_metrics['sharpe_rolling'],
                        'max_drawdown': portfolio_metrics['max_drawdown'],
                        'turnover': portfolio_metrics['turnover'],
                    }, total_steps_done, 'portfolio')

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

            # Separate train and theory metrics
            train_metrics = {k: v for k, v in metrics.items() if k not in ['gate_shapley_corr']}
            self._log_metrics(train_metrics, total_steps_done, 'train')

            # Log Shapley-Gate alignment under theory/ prefix (Paper Line 1375-1390)
            if 'gate_shapley_corr' in metrics:
                self._log_metrics({'gate_shapley_corr': metrics['gate_shapley_corr']}, total_steps_done, 'theory')

            # Log gate and TERC metrics periodically
            if total_steps_done % 1000 == 0:
                if hasattr(self.agent, 'actor_critic') and hasattr(self.agent.actor_critic, 'get_all_tensorboard_metrics'):
                    model_metrics = self.agent.actor_critic.get_all_tensorboard_metrics()
                    if model_metrics:
                        # Split metrics by prefix for proper TensorBoard organization
                        gate_metrics = {k.replace('gate/', ''): v for k, v in model_metrics.items() if k.startswith('gate/')}
                        terc_metrics = {k.replace('terc/', ''): v for k, v in model_metrics.items() if k.startswith('terc/')}

                        if gate_metrics:
                            self._log_metrics(gate_metrics, total_steps_done, 'gate')
                        if terc_metrics:
                            self._log_metrics(terc_metrics, total_steps_done, 'terc')

            # Log H-divergence metrics periodically (Paper Theorem 2)
            if total_steps_done % 5000 == 0:
                h_div_stats = self.h_div_monitor.get_statistics()
                if h_div_stats:
                    d_H = h_div_stats.get('d_H_latest', 0.0)
                    history['d_H'].append(d_H)
                    self._log_metrics({'d_H': d_H}, total_steps_done, prefix='theory')

                    # Compare with paper claims
                    comparison = compare_with_paper_claims(d_H, method='CQL')
                    self.logger.info(
                        f"Step {total_steps_done} H-divergence: d_H={d_H:.4f} "
                        f"(paper: {comparison['paper_claim_mean']:.2f}±{comparison['paper_claim_std']:.2f}, "
                        f"reduction={comparison['reduction_from_baseline_pct']:.1f}%)"
                    )

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
        """Save training checkpoint with full state for resume."""
        suffix = 'final' if final else f'step_{step}'
        path = self.config.checkpoint_dir / f'{run_name}_{suffix}.pt'

        # Save comprehensive checkpoint for resume
        checkpoint = {
            'step': step,
            'run_name': run_name,
            'actor_critic': self.agent.actor_critic.state_dict(),
            'optimizer': self.agent.optimizer.state_dict(),
            'lambda_cvar': self.agent.lambda_cvar,
            'config': self.agent.config,
            'total_steps': self.agent.total_steps,
        }
        if self.agent.scheduler is not None:
            checkpoint['scheduler'] = self.agent.scheduler.state_dict()

        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

        # Also save as {run_name}_ppo_model.pt for easy resume (run_name으로 구분)
        model_path = self.config.checkpoint_dir / f'{run_name}_ppo_model.pt'
        torch.save(checkpoint, model_path)
        self.logger.info(f"Saved latest model: {model_path}")

    def load_checkpoint(self, path: Path) -> int:
        """Load checkpoint and return the step to resume from."""
        self.logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.agent.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
        self.agent.lambda_cvar = checkpoint['lambda_cvar']
        self.agent.total_steps = checkpoint['total_steps']

        if self.agent.scheduler is not None and 'scheduler' in checkpoint:
            self.agent.scheduler.load_state_dict(checkpoint['scheduler'])

        step = checkpoint.get('step', 0)
        self.logger.info(f"Resumed from step {step}")
        return step

    def find_checkpoint(self, run_name: str) -> Optional[Path]:
        """Find checkpoint file for given run_name."""
        # 1. run_name 전용 체크포인트
        model_path = self.config.checkpoint_dir / f'{run_name}_ppo_model.pt'
        if model_path.exists():
            return model_path

        # 2. 기존 ppo_model.pt (하위 호환)
        legacy_path = self.config.checkpoint_dir / 'ppo_model.pt'
        if legacy_path.exists():
            return legacy_path

        return None


# ==============================================================================
# Hierarchical Trainer - For Attention-based Architecture
# ==============================================================================

class HierarchicalTrainer(Trainer):
    """
    Trainer for Hierarchical Actor-Critic with Attention.

    Supports the new architecture with:
    - HierarchicalFeatureEncoder (Cross-Alpha + Cross-Asset Attention)
    - HierarchicalActor (Meta Head + Portfolio Head)
    - HierarchicalCritic (CVaR)
    - HierarchicalTradingEnv (Conditional Execution)

    Key differences from Phase2Trainer:
    - Handles Dict state from HierarchicalStateBuilder
    - Hierarchical action (trade_prob, weights)
    - Separate log_prob computation for meta and portfolio heads
    """

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        rl_config: Optional[RLConfig] = None,
        model_config: Optional[Dict] = None,
    ):
        """
        Initialize hierarchical trainer.

        Args:
            config: Trainer configuration
            rl_config: RL training configuration
            model_config: Model architecture configuration
        """
        super().__init__(config, rl_config)

        # Model configuration (defaults)
        self.model_config = model_config or {
            'n_alphas': 292,
            'n_assets': 20,
            'd_alpha': 64,
            'd_text': 64,
            'd_temporal': 128,
            'd_global': 32,
            'd_portfolio': 16,
            'actor_hidden_dims': (128, 64),
            'critic_hidden_dims': (256, 128),
            'n_quantiles': 32,
            'dropout': 0.1,
        }

        # Training hyperparameters
        self.lr = rl_config.ppo_cvar.learning_rate if rl_config else 3e-4
        self.gamma = rl_config.ppo_cvar.gamma if rl_config else 0.99
        self.gae_lambda = rl_config.ppo_cvar.gae_lambda if rl_config else 0.95
        self.clip_epsilon = rl_config.ppo_cvar.clip_epsilon if rl_config else 0.2
        self.ppo_epochs = rl_config.ppo_cvar.ppo_epochs if rl_config else 10
        self.batch_size = rl_config.ppo_cvar.batch_size if rl_config else 256
        self.horizon = rl_config.ppo_cvar.horizon if rl_config else 2048
        self.alpha_cvar = rl_config.ppo_cvar.alpha_cvar if rl_config else 0.05
        self.kappa = rl_config.ppo_cvar.kappa if rl_config else 1.0
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        self.max_grad_norm = 0.5

        # Model and optimizer (lazy initialization)
        self.model = None
        self.optimizer = None

        # Environment
        self.env = None

        # Rollout storage
        self.rollout_storage = None

    def _init_model(self):
        """Initialize hierarchical model."""
        from agents.networks import HierarchicalActorCritic

        self.model = HierarchicalActorCritic(**self.model_config).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-5,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10000,
            T_mult=2,
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Initialized HierarchicalActorCritic: {n_params:,} parameters")

    def set_environment(self, env):
        """Set the training environment."""
        self.env = env
        self.logger.info(f"Environment set: {type(env).__name__}")

    def train(
        self,
        total_steps: int,
        run_name: Optional[str] = None,
        resume: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Run hierarchical PPO training.

        Args:
            total_steps: Total training steps
            run_name: Name for this training run
            resume: If True, resume from hierarchical_model.pt if it exists

        Returns:
            Dict of training metrics history
        """
        if self.env is None:
            raise RuntimeError("No environment set. Call set_environment() first.")

        run_name = run_name or f"hierarchical_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}{self._get_gpu_suffix()}"

        # Check for existing checkpoint to resume (run_name 기반)
        start_step = 0
        if resume:
            checkpoint_path = self.find_checkpoint(run_name)
            if checkpoint_path:
                start_step = self.load_checkpoint(checkpoint_path)
                self.logger.info(f"Resuming training from step {start_step}")

        if self.model is None:
            self._init_model()

        self._init_tensorboard(run_name)

        self.logger.info(f"Starting Hierarchical training: {total_steps} steps (starting from {start_step})")
        self.logger.info(f"Model config: {self.model_config}")

        # Training history
        history = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'meta_loss': [],
            'portfolio_loss': [],
            'cvar_loss': [],
            'episode_return': [],
            'trade_frequency': [],
        }

        # Reset environment and get initial observation
        obs, info = self.env.reset()

        # Convert obs to tensor format if needed
        state_dict = self._obs_to_state_dict(obs)

        episode_return = 0
        n_episodes = 0
        total_steps_done = start_step

        # Rollout storage (Lazy Agent - no trade_probs)
        rollout_states = []
        rollout_weights = []
        rollout_rewards = []
        rollout_values = []
        rollout_log_probs = []
        rollout_dones = []

        start_time = time.time()

        while total_steps_done < total_steps:
            # Clear rollout storage
            rollout_states.clear()
            rollout_weights.clear()
            rollout_rewards.clear()
            rollout_values.clear()
            rollout_log_probs.clear()
            rollout_dones.clear()

            # Collect rollout
            for _ in range(self.horizon):
                with torch.no_grad():
                    # Get action from model (Lazy Agent - no trade_prob)
                    weights, log_prob, value = self.model.get_action_and_value(
                        state_dict, deterministic=False
                    )

                # Store in rollout
                rollout_states.append({k: v.clone() for k, v in state_dict.items()})
                rollout_weights.append(weights.cpu())
                rollout_values.append(value.cpu())
                rollout_log_probs.append(log_prob.cpu())

                # Execute action in environment (weights only)
                action = weights.cpu().numpy().flatten()
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                rollout_rewards.append(reward)
                rollout_dones.append(terminated)

                episode_return += reward
                total_steps_done += 1

                if terminated or truncated:
                    # Log episode stats
                    stats = self.env.get_episode_stats()
                    history['episode_return'].append(episode_return)
                    history['trade_frequency'].append(stats.get('trade_frequency', 0))

                    self._log_metrics({
                        'episode_return': episode_return,
                        'trade_frequency': stats.get('trade_frequency', 0),
                        'sharpe_ratio': stats.get('sharpe_ratio', 0),
                        'max_drawdown': stats.get('max_drawdown', 0),
                    }, total_steps_done, 'episode')

                    # Reset
                    obs, info = self.env.reset()
                    episode_return = 0
                    n_episodes += 1

                state_dict = self._obs_to_state_dict(next_obs if not (terminated or truncated) else obs)

            # Compute advantages using GAE
            with torch.no_grad():
                last_value = self.model.get_value(state_dict)

            advantages, returns = self._compute_gae(
                rewards=rollout_rewards,
                values=[v.squeeze() for v in rollout_values],
                dones=rollout_dones,
                last_value=last_value.cpu().squeeze(),
            )

            # PPO update (Lazy Agent - no trade_probs)
            metrics = self._ppo_update(
                states=rollout_states,
                weights=rollout_weights,
                log_probs=rollout_log_probs,
                advantages=advantages,
                returns=returns,
            )

            # Record metrics
            for key in ['policy_loss', 'value_loss', 'entropy', 'meta_loss', 'portfolio_loss', 'cvar_loss']:
                if key in metrics:
                    history[key].append(metrics[key])

            self._log_metrics(metrics, total_steps_done, 'train')

            # Update learning rate
            self.scheduler.step()

            # Save checkpoint
            if total_steps_done % self.config.save_interval == 0:
                self._save_checkpoint(total_steps_done, run_name)

        # Final save
        self._save_checkpoint(total_steps_done, run_name, final=True)

        elapsed = time.time() - start_time
        self.logger.info(f"Hierarchical training completed in {elapsed/3600:.2f} hours")
        self.logger.info(f"Episodes completed: {n_episodes}")

        if self.writer:
            self.writer.close()

        return history

    def _obs_to_state_dict(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """Convert environment observation to model state dict."""
        # For now, create dummy state dict from legacy obs format
        # In production, HierarchicalStateBuilder should provide proper format

        B = 1
        T = 24  # lookback
        N = self.model_config['n_assets']

        # If obs already has the right format (from HierarchicalStateBuilder)
        if 'alphas' in obs:
            state_dict = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
                    if tensor.dim() < 4 and key not in ['portfolio_state']:
                        tensor = tensor.unsqueeze(0)  # Add batch dim
                    state_dict[key] = tensor
                else:
                    state_dict[key] = value
            return state_dict

        # Legacy format: convert market/portfolio to hierarchical format
        market = obs.get('market', np.zeros((N, 36)))
        portfolio = obs.get('portfolio', np.zeros(22))

        # Create placeholder state dict
        state_dict = {
            'alphas': torch.zeros(B, T, N, self.model_config['n_alphas'], device=self.device),
            'news_embedding': torch.zeros(B, T, N, 768, device=self.device),
            'social_embedding': torch.zeros(B, T, N, 768, device=self.device),
            'has_social_signal': torch.ones(B, T, N, 1, device=self.device),
            'global_features': torch.zeros(B, T, 6, device=self.device),
            'portfolio_state': torch.tensor(portfolio, dtype=torch.float32, device=self.device).unsqueeze(0),
        }

        # Fill in from market state (first 20 features as mock alphas)
        if market.shape[1] >= 20:
            for t in range(T):
                state_dict['alphas'][0, t, :, :20] = torch.tensor(
                    market[:, :20], dtype=torch.float32, device=self.device
                )

        return state_dict

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        dones: List[bool],
        last_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        T = len(rewards)
        advantages = torch.zeros(T)
        returns = torch.zeros(T)

        gae = 0
        next_value = last_value.item()

        for t in reversed(range(T)):
            if dones[t]:
                next_value = 0
                gae = 0

            delta = rewards[t] + self.gamma * next_value - values[t].item()
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            returns[t] = gae + values[t].item()

            next_value = values[t].item()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def _ppo_update(
        self,
        states: List[Dict],
        weights: List[torch.Tensor],
        log_probs: List[torch.Tensor],
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Run PPO update epochs (Lazy Agent - no trade_probs)."""
        T = len(states)

        # Stack tensors
        old_weights = torch.stack(weights)
        old_log_probs = torch.stack(log_probs)

        # Convert to device
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.ppo_epochs):
            # Generate random indices for mini-batches
            indices = np.random.permutation(T)

            for start in range(0, T, self.batch_size):
                end = min(start + self.batch_size, T)
                batch_indices = indices[start:end]

                # Get batch
                batch_states = self._stack_state_dicts([states[i] for i in batch_indices])
                batch_old_log_probs = old_log_probs[batch_indices].to(self.device)
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass (Lazy Agent - no trade_prob)
                new_weights, value, quantiles = self.model(batch_states)

                # Compute new log probs (Gaussian approximation for weights only)
                new_log_probs = self._compute_log_prob(
                    new_weights,
                    old_weights[batch_indices].to(self.device),
                )

                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((value.squeeze() - batch_returns) ** 2).mean()

                # Entropy bonus (encourage exploration)
                entropy = -new_log_probs.mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / max(1, n_updates),
            'value_loss': total_value_loss / max(1, n_updates),
            'entropy': total_entropy / max(1, n_updates),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }

    def _stack_state_dicts(self, state_dicts: List[Dict]) -> Dict[str, torch.Tensor]:
        """Stack list of state dicts into batched state dict."""
        result = {}
        keys = state_dicts[0].keys()

        for key in keys:
            tensors = [s[key] for s in state_dicts]
            # Remove existing batch dim and re-stack
            tensors = [t.squeeze(0) if t.dim() > 0 and t.shape[0] == 1 else t for t in tensors]
            result[key] = torch.stack(tensors).to(self.device)

        return result

    def _compute_log_prob(
        self,
        weights: torch.Tensor,
        old_weights: torch.Tensor,
        std: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute log probability for portfolio weights (Lazy Agent).

        Uses Gaussian approximation for continuous action space.
        """
        # Gaussian approximation for weights
        diff = (weights - old_weights) / std
        log_prob = -0.5 * (diff ** 2).sum(dim=-1)

        return log_prob

    def _save_checkpoint(self, step: int, run_name: str, final: bool = False):
        """Save training checkpoint with full state for resume."""
        suffix = 'final' if final else f'step_{step}'
        path = self.config.checkpoint_dir / f'{run_name}_{suffix}.pt'

        checkpoint = {
            'step': step,
            'run_name': run_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_config': self.model_config,
        }

        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

        # Also save as {run_name}_hierarchical_model.pt for easy resume (run_name으로 구분)
        model_path = self.config.checkpoint_dir / f'{run_name}_hierarchical_model.pt'
        torch.save(checkpoint, model_path)
        self.logger.info(f"Saved latest model: {model_path}")

    def load_checkpoint(self, path: Path) -> int:
        """Load checkpoint and return the step to resume from."""
        self.logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model_config = checkpoint['model_config']
        self._init_model()

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        step = checkpoint.get('step', 0)
        self.logger.info(f"Resumed from step {step}")
        return step

    def find_checkpoint(self, run_name: str) -> Optional[Path]:
        """Find checkpoint file for given run_name."""
        # 1. run_name 전용 체크포인트
        model_path = self.config.checkpoint_dir / f'{run_name}_hierarchical_model.pt'
        if model_path.exists():
            return model_path

        # 2. 기존 hierarchical_model.pt (하위 호환)
        legacy_path = self.config.checkpoint_dir / 'hierarchical_model.pt'
        if legacy_path.exists():
            return legacy_path

        return None


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
