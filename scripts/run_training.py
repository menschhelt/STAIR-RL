#!/usr/bin/env python3
"""
RL Training Script.

Trains the STAIR-RL agent (Factor-only RL without LLM):
- Phase 1: CQL-SAC offline pre-training on historical data
- Phase 2: PPO-CVaR online fine-tuning

Usage:
    python scripts/run_training.py --phase 1 --steps 500000
    python scripts/run_training.py --phase 2 --steps 100000
    python scripts/run_training.py --all

GPU:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_training.py --phase 1
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import sys
import os

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config, DATA_DIR, BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_device(gpu_id: int = 0):
    """Setup CUDA device."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        logger.info(f"Using CUDA device {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.warning("CUDA not available, using CPU")
    return device


def run_phase1_cql_sac(config: Config, device: torch.device, steps: int, checkpoint_dir: Path):
    """
    Phase 1: CQL-SAC Offline Pre-training.

    Uses historical data (2021.01 - 2023.06) to pre-train agent
    with Conservative Q-Learning to avoid overestimation.
    """
    from agents.cql_sac import CQLSACAgent
    from training.replay_buffer import ReplayBuffer
    from environments.trading_env import TradingEnv, EnvConfig
    from backtesting.data_loader import BacktestDataLoader

    logger.info("=" * 60)
    logger.info("Phase 1: CQL-SAC Offline Pre-training")
    logger.info(f"Training steps: {steps:,}")
    logger.info("=" * 60)

    # Load training data
    logger.info("Loading training data...")
    data_loader = BacktestDataLoader(
        data_dir=DATA_DIR,
        feature_dir=DATA_DIR / 'features',
    )

    train_data = data_loader.load_period(
        start_date=config.backtest.train_start,
        end_date=config.backtest.train_end,
    )
    logger.info(f"Training data loaded: {len(train_data)} rows")

    # Create environment
    env_config = EnvConfig(
        n_assets=config.universe.top_n,
        target_leverage=config.rl.target_leverage,
        transaction_cost_rate=config.backtest.taker_fee + config.backtest.slippage,
    )

    env = TradingEnv(
        data=train_data,
        config=env_config,
    )

    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Initialize agent
    agent = CQLSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        learning_rate_actor=config.rl.cql_sac.learning_rate_actor,
        learning_rate_critic=config.rl.cql_sac.learning_rate_critic,
        lambda_cql=config.rl.cql_sac.lambda_cql,
        lambda_gp=config.rl.cql_sac.lambda_gp,
        alpha=config.rl.cql_sac.alpha,
        tau=config.rl.cql_sac.tau,
        gamma=config.rl.cql_sac.gamma,
    )

    # Fill replay buffer with historical data
    logger.info("Building replay buffer from historical data...")
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=config.rl.cql_sac.replay_buffer_size,
        device=device,
    )

    # Collect transitions from environment
    state, _ = env.reset()
    episode_transitions = []

    while len(replay_buffer) < min(config.rl.cql_sac.replay_buffer_size, len(train_data) * 0.9):
        # Use behavior policy (can be random or heuristic)
        action = env.action_space.sample()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay_buffer.add(state, action, reward, next_state, done)

        if done:
            state, _ = env.reset()
        else:
            state = next_state

        if len(replay_buffer) % 10000 == 0:
            logger.info(f"Replay buffer: {len(replay_buffer):,} transitions")

    logger.info(f"Replay buffer filled: {len(replay_buffer):,} transitions")

    # Training loop
    logger.info("Starting CQL-SAC training...")
    batch_size = config.rl.cql_sac.batch_size

    for step in range(1, steps + 1):
        # Sample batch
        batch = replay_buffer.sample(batch_size)

        # Update agent
        metrics = agent.update(batch)

        # Logging
        if step % 1000 == 0:
            logger.info(
                f"Step {step:,}/{steps:,} | "
                f"Critic Loss: {metrics.get('critic_loss', 0):.4f} | "
                f"Actor Loss: {metrics.get('actor_loss', 0):.4f} | "
                f"CQL Loss: {metrics.get('cql_loss', 0):.4f}"
            )

        # Checkpoint
        if step % 50000 == 0:
            checkpoint_path = checkpoint_dir / f'cql_sac_step_{step}.pt'
            agent.save(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_path = checkpoint_dir / 'cql_sac_final.pt'
    agent.save(final_path)
    logger.info(f"Phase 1 completed. Model saved: {final_path}")

    return agent


def run_phase2_ppo_cvar(
    config: Config,
    device: torch.device,
    steps: int,
    checkpoint_dir: Path,
    pretrained_path: Path = None,
):
    """
    Phase 2: PPO-CVaR Online Fine-tuning.

    Fine-tunes the pre-trained agent with PPO and CVaR constraint
    on validation data (2023.07 - 2023.12).
    """
    from agents.ppo_cvar import PPOCVaRAgent
    from training.rollout_buffer import RolloutBuffer
    from environments.trading_env import TradingEnv, EnvConfig
    from backtesting.data_loader import BacktestDataLoader

    logger.info("=" * 60)
    logger.info("Phase 2: PPO-CVaR Online Fine-tuning")
    logger.info(f"Training steps: {steps:,}")
    logger.info("=" * 60)

    # Load validation data
    logger.info("Loading validation data...")
    data_loader = BacktestDataLoader(
        data_dir=DATA_DIR,
        feature_dir=DATA_DIR / 'features',
    )

    val_data = data_loader.load_period(
        start_date=config.backtest.val_start,
        end_date=config.backtest.val_end,
    )
    logger.info(f"Validation data loaded: {len(val_data)} rows")

    # Create environment
    env_config = EnvConfig(
        n_assets=config.universe.top_n,
        target_leverage=config.rl.target_leverage,
        transaction_cost_rate=config.backtest.taker_fee + config.backtest.slippage,
    )

    env = TradingEnv(
        data=val_data,
        config=env_config,
    )

    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize agent
    agent = PPOCVaRAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        learning_rate=config.rl.ppo_cvar.learning_rate,
        clip_epsilon=config.rl.ppo_cvar.clip_epsilon,
        ppo_epochs=config.rl.ppo_cvar.ppo_epochs,
        gae_lambda=config.rl.ppo_cvar.gae_lambda,
        gamma=config.rl.ppo_cvar.gamma,
        alpha_cvar=config.rl.ppo_cvar.alpha_cvar,
        kappa=config.rl.ppo_cvar.kappa,
    )

    # Load pretrained weights if available
    if pretrained_path and pretrained_path.exists():
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        agent.load_pretrained(pretrained_path)

    # Rollout buffer
    rollout_buffer = RolloutBuffer(
        horizon=config.rl.ppo_cvar.horizon,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )

    # Training loop
    logger.info("Starting PPO-CVaR training...")
    total_steps = 0
    episode = 0

    while total_steps < steps:
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0

        # Collect rollout
        while not rollout_buffer.is_full():
            # Get action from policy
            action, log_prob, value = agent.get_action(state)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            rollout_buffer.add(state, action, reward, value, log_prob, done)

            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if done:
                break

            state = next_state

        # Update policy
        if rollout_buffer.is_full() or done:
            # Compute returns and advantages
            last_value = agent.get_value(state) if not done else 0
            rollout_buffer.compute_returns(last_value, agent.gamma, agent.gae_lambda)

            # PPO update
            metrics = agent.update(rollout_buffer)
            rollout_buffer.reset()

            # Logging
            if episode % 10 == 0:
                logger.info(
                    f"Episode {episode} | Steps: {total_steps:,}/{steps:,} | "
                    f"Reward: {episode_reward:.4f} | "
                    f"CVaR: {metrics.get('cvar', 0):.4f} | "
                    f"Lambda: {metrics.get('lambda_cvar', 0):.4f}"
                )

        episode += 1

        # Checkpoint
        if total_steps % 20000 == 0:
            checkpoint_path = checkpoint_dir / f'ppo_cvar_step_{total_steps}.pt'
            agent.save(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_path = checkpoint_dir / 'ppo_cvar_final.pt'
    agent.save(final_path)
    logger.info(f"Phase 2 completed. Model saved: {final_path}")

    return agent


def run_full_training(config: Config, device: torch.device, checkpoint_dir: Path):
    """Run full 2-phase training pipeline."""
    logger.info("=" * 60)
    logger.info("STAIR-RL Full Training Pipeline")
    logger.info("=" * 60)

    # Phase 1: CQL-SAC
    logger.info("\n" + "=" * 60)
    logger.info("Starting Phase 1: CQL-SAC Offline Pre-training")
    logger.info("=" * 60)

    run_phase1_cql_sac(
        config=config,
        device=device,
        steps=config.rl.cql_sac.training_steps,
        checkpoint_dir=checkpoint_dir / 'phase1',
    )

    # Phase 2: PPO-CVaR
    logger.info("\n" + "=" * 60)
    logger.info("Starting Phase 2: PPO-CVaR Online Fine-tuning")
    logger.info("=" * 60)

    pretrained_path = checkpoint_dir / 'phase1' / 'cql_sac_final.pt'

    run_phase2_ppo_cvar(
        config=config,
        device=device,
        steps=config.rl.ppo_cvar.training_steps,
        checkpoint_dir=checkpoint_dir / 'phase2',
        pretrained_path=pretrained_path,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Full training pipeline completed")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Train STAIR-RL agent'
    )
    parser.add_argument(
        '--phase', type=int, choices=[1, 2], default=None,
        help='Training phase (1: CQL-SAC, 2: PPO-CVaR)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run full training pipeline (Phase 1 + Phase 2)'
    )
    parser.add_argument(
        '--steps', type=int, default=None,
        help='Number of training steps (overrides config)'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU ID to use'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default=None,
        help='Directory for saving checkpoints'
    )
    parser.add_argument(
        '--pretrained', type=str, default=None,
        help='Path to pretrained model (for Phase 2)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load config
    if args.config:
        config = Config.from_yaml(Path(args.config))
    else:
        config = Config()

    # Setup device
    device = setup_device(args.gpu)

    # Setup checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_dir = BASE_DIR / 'checkpoints' / f'run_{timestamp}'

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / 'phase1').mkdir(exist_ok=True)
    (checkpoint_dir / 'phase2').mkdir(exist_ok=True)

    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Run training
    if args.all:
        run_full_training(config, device, checkpoint_dir)
    elif args.phase == 1:
        steps = args.steps or config.rl.cql_sac.training_steps
        run_phase1_cql_sac(config, device, steps, checkpoint_dir / 'phase1')
    elif args.phase == 2:
        steps = args.steps or config.rl.ppo_cvar.training_steps
        pretrained_path = Path(args.pretrained) if args.pretrained else None
        run_phase2_ppo_cvar(config, device, steps, checkpoint_dir / 'phase2', pretrained_path)
    else:
        logger.error("Please specify --phase 1, --phase 2, or --all")
        sys.exit(1)


if __name__ == '__main__':
    main()
