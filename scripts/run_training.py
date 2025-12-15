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
import subprocess
import socket

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config, DATA_DIR, BASE_DIR
from training.vectorized_env import make_vec_env

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_free_port(start_port=6006, max_tries=10):
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    return None


def start_tensorboard(logdir: Path, port: int = None):
    """Start TensorBoard server in background and return the process."""
    if port is None:
        port = find_free_port()
        if port is None:
            logger.warning("Could not find free port for TensorBoard (6006-6016 all in use)")
            return None, None

    try:
        # Start TensorBoard in background
        process = subprocess.Popen(
            ['tensorboard', '--logdir', str(logdir), '--port', str(port), '--bind_all'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 TENSORBOARD STARTED")
        logger.info("=" * 80)
        logger.info(f"  📁 Log directory: {logdir}")
        logger.info(f"  🌐 URL: http://localhost:{port}")
        logger.info(f"  🖥️  Process ID: {process.pid}")
        logger.info("")
        logger.info("  💡 Tip: TensorBoard will continue running after training completes")
        logger.info(f"  💡 To stop: kill {process.pid}")
        logger.info("=" * 80)
        logger.info("")

        return process, port
    except FileNotFoundError:
        logger.warning("TensorBoard not found. Install with: pip install tensorboard")
        return None, None
    except Exception as e:
        logger.warning(f"Failed to start TensorBoard: {e}")
        return None, None


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


def run_phase1_cql_sac(
    config: Config,
    device: torch.device,
    steps: int,
    checkpoint_dir: Path,
    embedding_dir: Path = None,
    n_envs: int = 1,
):
    """
    Phase 1: CQL-SAC Offline Pre-training.

    Uses historical data (2021.01 - 2023.06) to pre-train agent
    with Conservative Q-Learning to avoid overestimation.
    """
    from agents.cql_sac import CQLSACAgent, CQLSACConfig, ReplayBuffer
    from environments.trading_env import TradingEnv, EnvConfig
    from training.data_loader import TrainingDataLoader

    logger.info("=" * 60)
    logger.info("Phase 1: CQL-SAC Offline Pre-training")
    logger.info(f"Training steps: {steps:,}")
    logger.info("=" * 60)

    # Load training data
    logger.info("Loading training data...")
    data_loader = TrainingDataLoader(
        data_dir=DATA_DIR,
        n_assets=config.universe.top_n,
    )

    train_data = data_loader.load_period(
        start_date=config.backtest.train_start,
        end_date=config.backtest.train_end,
    )
    logger.info(f"Training data loaded: {train_data['states'].shape[0]} timesteps")

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

    # Create agent config
    agent_config = CQLSACConfig(
        n_assets=action_dim,
        state_dim=state_dim,
        lr_actor=config.rl.cql_sac.learning_rate_actor,
        lr_critic=config.rl.cql_sac.learning_rate_critic,
        lambda_cql=config.rl.cql_sac.lambda_cql,
        lambda_gp=config.rl.cql_sac.lambda_gp,
        tau=config.rl.cql_sac.tau,
        gamma=config.rl.cql_sac.gamma,
        batch_size=config.rl.cql_sac.batch_size,
    )

    # Set embedding paths if provided
    if embedding_dir:
        agent_config.gdelt_embeddings_path = str(embedding_dir / 'gdelt_embeddings.h5')
        agent_config.nostr_embeddings_path = str(embedding_dir / 'nostr_embeddings.h5')
        logger.info(f"Using embeddings from: {embedding_dir}")
        logger.info(f"  GDELT: {agent_config.gdelt_embeddings_path}")
        logger.info(f"  Nostr: {agent_config.nostr_embeddings_path}")

    # Initialize agent
    agent = CQLSACAgent(
        config=agent_config,
        device=str(device),
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

    # Setup TensorBoard
    tb_log_dir = checkpoint_dir / 'tensorboard'
    tb_log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    # Auto-start TensorBoard server
    tb_process, tb_port = start_tensorboard(tb_log_dir)

    # Training loop
    logger.info("Starting CQL-SAC training...")
    batch_size = config.rl.cql_sac.batch_size

    for step in range(1, steps + 1):
        # Sample batch
        batch = replay_buffer.sample(batch_size)

        # Update agent
        metrics = agent.update(batch)

        # TensorBoard logging (every step)
        writer.add_scalar('Loss/Critic', metrics.get('critic_loss', 0), step)
        writer.add_scalar('Loss/Actor', metrics.get('actor_loss', 0), step)
        writer.add_scalar('Loss/CQL', metrics.get('cql_loss', 0), step)
        writer.add_scalar('Loss/Total', metrics.get('total_loss', 0), step)

        # SAC-specific metrics
        if 'alpha' in metrics:
            writer.add_scalar('SAC/Alpha', metrics['alpha'], step)
        if 'q1_value' in metrics:
            writer.add_scalar('Q-Value/Q1_Mean', metrics['q1_value'], step)
        if 'q2_value' in metrics:
            writer.add_scalar('Q-Value/Q2_Mean', metrics['q2_value'], step)

        # Gradient norms
        if 'actor_grad_norm' in metrics:
            writer.add_scalar('GradNorm/Actor', metrics['actor_grad_norm'], step)
        if 'critic_grad_norm' in metrics:
            writer.add_scalar('GradNorm/Critic', metrics['critic_grad_norm'], step)

        # Console logging (every 1000 steps)
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

    # Close TensorBoard writer
    writer.close()

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
    embedding_dir: Path = None,
    n_envs: int = 1,
):
    """
    Phase 2: PPO-CVaR Online Fine-tuning.

    Fine-tunes the pre-trained agent with PPO and CVaR constraint
    on validation data (2023.07 - 2023.12).
    """
    from agents.ppo_cvar import PPOCVaRAgent, PPOCVaRConfig, RolloutBuffer
    from environments.trading_env import TradingEnv, EnvConfig
    from training.data_loader import TrainingDataLoader

    logger.info("=" * 60)
    logger.info("Phase 2: PPO-CVaR Online Fine-tuning")
    logger.info(f"Training steps: {steps:,}")
    logger.info("=" * 60)

    # Load validation data
    logger.info("Loading validation data...")
    data_loader = TrainingDataLoader(
        data_dir=DATA_DIR,
        n_assets=config.universe.top_n,
    )

    val_data = data_loader.load_period(
        start_date=config.backtest.val_start,
        end_date=config.backtest.val_end,
    )
    logger.info(f"Validation data loaded: {val_data['states'].shape[0]} timesteps")

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

    # Create agent config
    agent_config = PPOCVaRConfig(
        n_assets=action_dim,
        state_dim=state_dim,
        lr=config.rl.ppo_cvar.learning_rate,
        clip_epsilon=config.rl.ppo_cvar.clip_epsilon,
        ppo_epochs=config.rl.ppo_cvar.ppo_epochs,
        gae_lambda=config.rl.ppo_cvar.gae_lambda,
        gamma=config.rl.ppo_cvar.gamma,
        alpha_cvar=config.rl.ppo_cvar.alpha_cvar,
        kappa=config.rl.ppo_cvar.kappa,
        horizon=config.rl.ppo_cvar.horizon,
        batch_size=config.rl.ppo_cvar.batch_size,
    )

    # Set embedding paths if provided
    if embedding_dir:
        agent_config.gdelt_embeddings_path = str(embedding_dir / 'gdelt_embeddings.h5')
        agent_config.nostr_embeddings_path = str(embedding_dir / 'nostr_embeddings.h5')
        logger.info(f"Using embeddings from: {embedding_dir}")
        logger.info(f"  GDELT: {agent_config.gdelt_embeddings_path}")
        logger.info(f"  Nostr: {agent_config.nostr_embeddings_path}")

    # Initialize agent
    agent = PPOCVaRAgent(
        config=agent_config,
        device=str(device),
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

    # Setup TensorBoard
    tb_log_dir = checkpoint_dir / 'tensorboard'
    tb_log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    # Auto-start TensorBoard server
    tb_process, tb_port = start_tensorboard(tb_log_dir)

    # Training loop
    logger.info("Starting PPO-CVaR training...")
    total_steps = 0
    episode = 0

    while total_steps < steps:
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_costs = 0
        episode_turnover = 0

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

            # Accumulate costs and turnover from env info
            if 'transaction_cost' in info:
                episode_costs += info['transaction_cost']
            if 'turnover' in info:
                episode_turnover += info['turnover']

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

            # TensorBoard logging (per episode)
            writer.add_scalar('Episode/Reward', episode_reward, episode)
            writer.add_scalar('Episode/Steps', episode_steps, episode)
            writer.add_scalar('Episode/TransactionCost', episode_costs, episode)
            writer.add_scalar('Episode/Turnover', episode_turnover, episode)

            # PPO metrics
            writer.add_scalar('Loss/Policy', metrics.get('policy_loss', 0), total_steps)
            writer.add_scalar('Loss/Value', metrics.get('value_loss', 0), total_steps)
            writer.add_scalar('Loss/Entropy', metrics.get('entropy_loss', 0), total_steps)
            writer.add_scalar('Loss/Total', metrics.get('total_loss', 0), total_steps)

            # CVaR metrics
            if 'cvar' in metrics:
                writer.add_scalar('CVaR/Value', metrics['cvar'], total_steps)
            if 'lambda_cvar' in metrics:
                writer.add_scalar('CVaR/Lambda', metrics['lambda_cvar'], total_steps)
            if 'cvar_violation' in metrics:
                writer.add_scalar('CVaR/Violation', metrics['cvar_violation'], total_steps)

            # Policy metrics
            if 'policy_entropy' in metrics:
                writer.add_scalar('Policy/Entropy', metrics['policy_entropy'], total_steps)
            if 'kl_divergence' in metrics:
                writer.add_scalar('Policy/KL_Divergence', metrics['kl_divergence'], total_steps)
            if 'clip_fraction' in metrics:
                writer.add_scalar('Policy/ClipFraction', metrics['clip_fraction'], total_steps)

            # Console logging (every 10 episodes)
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

    # Close TensorBoard writer
    writer.close()

    # Save final model
    final_path = checkpoint_dir / 'ppo_cvar_final.pt'
    agent.save(final_path)
    logger.info(f"Phase 2 completed. Model saved: {final_path}")

    return agent


def run_full_training(config: Config, device: torch.device, checkpoint_dir: Path, embedding_dir: Path = None, n_envs: int = 1):
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
        embedding_dir=embedding_dir,
        n_envs=n_envs,
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
        embedding_dir=embedding_dir,
        n_envs=n_envs,
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
    parser.add_argument(
        '--embedding-dir', type=str, default='/home/work/data/stair-local/embeddings',
        help='Directory containing GDELT and Nostr embedding HDF5 files'
    )
    parser.add_argument(
        '--n-envs', type=int, default=1,
        help='Number of parallel environments (vectorized training)'
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

    # Setup embedding directory
    embedding_dir = Path(args.embedding_dir) if args.embedding_dir else None
    if embedding_dir and not embedding_dir.exists():
        logger.warning(f"Embedding directory does not exist: {embedding_dir}")
        logger.warning("Hierarchical mode will use zero embeddings (fallback)")
        embedding_dir = None

    # Run training
    if args.all:
        run_full_training(config, device, checkpoint_dir, embedding_dir, args.n_envs)
    elif args.phase == 1:
        steps = args.steps or config.rl.cql_sac.training_steps
        run_phase1_cql_sac(config, device, steps, checkpoint_dir / 'phase1', embedding_dir, args.n_envs)
    elif args.phase == 2:
        steps = args.steps or config.rl.ppo_cvar.training_steps
        pretrained_path = Path(args.pretrained) if args.pretrained else None
        run_phase2_ppo_cvar(config, device, steps, checkpoint_dir / 'phase2', pretrained_path, embedding_dir, args.n_envs)
    else:
        logger.error("Please specify --phase 1, --phase 2, or --all")
        sys.exit(1)


if __name__ == '__main__':
    main()
