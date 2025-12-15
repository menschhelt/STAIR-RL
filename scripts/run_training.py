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
import pandas as pd
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


def preload_data_for_fast_training(
    agent,
    symbols: list,
    timestamps: np.ndarray,
    start_date: str,
    end_date: str,
    slot_symbols: np.ndarray = None,
):
    """
    Preload alpha and macro data into memory for O(1) lookup during training.

    This dramatically speeds up training by avoiding repeated pandas operations.
    Without preload: ~2,020 pandas ops per encode (101 alphas × 20 symbols)
    With preload: O(1) numpy indexing

    For dynamic universe support, pass slot_symbols to enable proper mapping
    from (timestamp, slot) → symbol → preload_index.

    Args:
        agent: CQLSACAgent or PPOCVaRAgent with hierarchical adapter
        symbols: List of ALL trading symbols ever in universe (e.g., 100 symbols)
        timestamps: Array of ISO timestamp strings from training data
        start_date: Start date for macro data preload (e.g., '2021-01-01')
        end_date: End date for macro data preload (e.g., '2022-06-30')
        slot_symbols: (T, N_slots) array of symbol names at each timestamp.
                     Enables dynamic universe: model gets only 20 slot symbols at each time.
    """
    logger.info("=" * 60)
    logger.info("Preloading data for fast training...")
    logger.info("=" * 60)

    # Check if agent has hierarchical adapter
    if not hasattr(agent, 'adapter') or agent.adapter is None:
        logger.warning("Agent does not have hierarchical adapter, skipping preload")
        return

    state_builder = agent.adapter.state_builder

    # 1. Set symbols for alpha loading
    if state_builder._current_symbols is None:
        state_builder.set_symbols(symbols)
        logger.info(f"Set symbols: {len(symbols)} symbols")

    # 2. Preload Alpha data as numpy array for O(1) lookup
    if state_builder.alpha_loader is not None:
        # Convert timestamps to DatetimeIndex
        ts_datetime = pd.to_datetime(timestamps)
        ts_datetime = pd.DatetimeIndex(ts_datetime)

        logger.info(f"Preloading alphas: {len(ts_datetime)} timestamps, {len(symbols)} symbols")
        state_builder.alpha_loader.preload_as_numpy(
            symbols=symbols,
            timestamps=ts_datetime,
        )
        logger.info(f"Alpha preload complete: {state_builder.alpha_loader.is_preloaded()}")

        # 2b. Set slot_symbols mapping for dynamic universe support
        if slot_symbols is not None:
            logger.info(f"Setting slot_symbols mapping: {slot_symbols.shape}")
            state_builder.alpha_loader.set_slot_symbols(slot_symbols)
            logger.info(f"Slot mapping ready: {state_builder.alpha_loader.has_slot_mapping()}")
        else:
            logger.warning("No slot_symbols provided - will use simple slicing (may cause issues)")
    else:
        logger.warning("Alpha loader not available, skipping alpha preload")

    # 3. Preload Macro data for O(1) lookup
    if state_builder.macro_loader is not None:
        logger.info(f"Preloading macro data: {start_date} to {end_date}")
        state_builder.macro_loader.preload_all(
            start_date=start_date,
            end_date=end_date,
            freq='5min',
        )
        logger.info(f"Macro preload complete: {state_builder.macro_loader.is_preloaded()}")
    else:
        logger.warning("Macro loader not available, skipping macro preload")

    logger.info("=" * 60)
    logger.info("Data preload complete - ready for fast training!")
    logger.info("=" * 60)


def precompute_features_for_buffer(
    agent,
    timestamps: np.ndarray,
    slot_symbols: np.ndarray,
    n_slots: int = 20,
    embedding_dim: int = 768,
    temporal_window: int = 24,
) -> dict:
    """
    Precompute ALL features into numpy arrays for O(1) buffer lookup.

    This is the KEY optimization that eliminates build_state_dict() overhead.
    Features are precomputed ONCE and indexed directly during training.

    Args:
        agent: CQLSACAgent with hierarchical adapter (already preloaded)
        timestamps: Array of ISO timestamp strings (T_total,)
        slot_symbols: (T_total, N_slots) - which symbols are in each slot per timestamp
        n_slots: Number of asset slots (default 20)
        embedding_dim: Embedding dimension (default 768)
        temporal_window: Temporal window for lookback (default 24)

    Returns:
        dict with precomputed arrays and timestamp mapping:
        - 'alphas': (T_total, N_slots, 101) - Alpha factors
        - 'gdelt': (T_total, 768) - GDELT embeddings
        - 'nostr': (T_total, 768) - Nostr embeddings
        - 'macro': (T_total, n_macro) - Macro features
        - 'ts_to_idx': timestamp -> index mapping
        - 'timestamp_indices': (T_total,) integer indices
    """
    import time
    from tqdm import tqdm

    logger.info("=" * 60)
    logger.info("Precomputing features for ReplayBuffer...")
    logger.info("=" * 60)

    T_total = len(timestamps)
    state_builder = agent.adapter.state_builder

    # Create timestamp -> index mapping
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    timestamp_indices = np.arange(T_total, dtype=np.int32)

    logger.info(f"Total timestamps: {T_total:,}")
    logger.info(f"Slots per timestamp: {n_slots}")
    logger.info(f"Temporal window: {temporal_window}")

    # 1. Extract alphas from preloaded alpha_loader
    logger.info("[1/4] Extracting alpha factors...")
    start_time = time.time()

    alpha_loader = state_builder.alpha_loader
    if alpha_loader is not None and alpha_loader.has_slot_mapping():
        # Use get_alphas_for_slots for all timestamps
        # This uses pre-computed slot_symbol_indices mapping
        all_indices = np.arange(T_total)
        preloaded_alphas = alpha_loader.get_alphas_for_slots(all_indices)
        logger.info(f"  Alpha shape: {preloaded_alphas.shape}")  # (T_total, N_slots, 101)
    else:
        logger.warning("  Alpha loader not ready, using zeros")
        preloaded_alphas = np.zeros((T_total, n_slots, 101), dtype=np.float32)

    elapsed = time.time() - start_time
    logger.info(f"  Done in {elapsed:.1f}s")

    # 2. Extract macro features from preloaded macro_loader
    logger.info("[2/4] Extracting macro features...")
    start_time = time.time()

    macro_loader = state_builder.macro_loader
    if macro_loader is not None and macro_loader.is_preloaded():
        # Get macro features for all timestamps
        # macro_loader has preloaded_data indexed by timestamp
        preloaded_macro = macro_loader.get_features_batch(timestamps)
        logger.info(f"  Macro shape: {preloaded_macro.shape}")  # (T_total, n_macro)
    else:
        logger.warning("  Macro loader not ready, using zeros")
        n_macro = 32  # Default
        preloaded_macro = np.zeros((T_total, n_macro), dtype=np.float32)

    elapsed = time.time() - start_time
    logger.info(f"  Done in {elapsed:.1f}s")

    # 3. Extract GDELT embeddings
    logger.info("[3/4] Extracting GDELT embeddings...")
    start_time = time.time()

    embedding_loader = state_builder.embedding_loader
    if embedding_loader is not None and hasattr(embedding_loader, 'get_gdelt_embeddings_marketwide_fast'):
        try:
            # Use the fast batch method that returns (T, 768) tensor
            gdelt_tensor = embedding_loader.get_gdelt_embeddings_marketwide_fast(list(timestamps))
            preloaded_gdelt = gdelt_tensor.cpu().numpy()
            logger.info(f"  GDELT shape: {preloaded_gdelt.shape}")
        except Exception as e:
            logger.warning(f"  GDELT extraction failed: {e}, using zeros")
            preloaded_gdelt = np.zeros((T_total, embedding_dim), dtype=np.float32)
    else:
        logger.warning("  GDELT loader not available, using zeros")
        preloaded_gdelt = np.zeros((T_total, embedding_dim), dtype=np.float32)

    elapsed = time.time() - start_time
    logger.info(f"  Done in {elapsed:.1f}s")

    # 4. Extract Nostr embeddings
    logger.info("[4/4] Extracting Nostr embeddings...")
    start_time = time.time()

    if embedding_loader is not None and hasattr(embedding_loader, 'get_nostr_embeddings_marketwide_fast'):
        try:
            # Use the fast batch method that returns (T, 768) tensor
            nostr_tensor = embedding_loader.get_nostr_embeddings_marketwide_fast(list(timestamps))
            preloaded_nostr = nostr_tensor.cpu().numpy()
            logger.info(f"  Nostr shape: {preloaded_nostr.shape}")
        except Exception as e:
            logger.warning(f"  Nostr extraction failed: {e}, using zeros")
            preloaded_nostr = np.zeros((T_total, embedding_dim), dtype=np.float32)
    else:
        logger.warning("  Nostr loader not available, using zeros")
        preloaded_nostr = np.zeros((T_total, embedding_dim), dtype=np.float32)

    elapsed = time.time() - start_time
    logger.info(f"  Done in {elapsed:.1f}s")

    # Memory usage
    total_memory = (
        preloaded_alphas.nbytes +
        preloaded_macro.nbytes +
        preloaded_gdelt.nbytes +
        preloaded_nostr.nbytes
    ) / 1e9
    logger.info(f"Total precomputed memory: {total_memory:.2f} GB")

    logger.info("=" * 60)
    logger.info("Feature precomputation complete!")
    logger.info("=" * 60)

    return {
        'alphas': preloaded_alphas,
        'gdelt': preloaded_gdelt,
        'nostr': preloaded_nostr,
        'macro': preloaded_macro,
        'ts_to_idx': ts_to_idx,
        'timestamp_indices': timestamp_indices,
    }


def run_phase1_cql_sac(
    config: Config,
    device: torch.device,
    steps: int,
    checkpoint_dir: Path,
    embedding_dir: Path = None,
    n_envs: int = 1,
    lr_actor: float = None,
    lr_critic: float = None,
    batch_size: int = None,
    resume_path: Path = None,
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

    # Load training data with DYNAMIC universe rebalancing
    logger.info("Loading training data with DYNAMIC universe...")
    data_loader = TrainingDataLoader(
        data_dir=DATA_DIR,
        n_assets=config.universe.top_n,
    )

    # Use dynamic loading for realistic universe rotation
    train_data = data_loader.load_period_dynamic(
        start_date=config.backtest.train_start,
        end_date=config.backtest.train_end,
    )
    logger.info(f"Training data loaded: {train_data['states'].shape[0]} timesteps")

    # Log dynamic universe stats
    if 'slot_changes' in train_data:
        rebalances = train_data['rebalance_mask'].sum()
        total_changes = train_data['slot_changes'].sum()
        logger.info(f"Dynamic universe: {rebalances} rebalance events, {total_changes} total slot changes")
        logger.info(f"All symbols in period: {len(train_data.get('all_symbols', []))}")

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
    # observation_space is Dict: {'market': (N, state_dim), 'portfolio': (N+2,)}
    market_space = env.observation_space['market']
    state_dim = market_space.shape[1]  # state_dim per asset
    action_dim = env.action_space.shape[0]  # N assets
    logger.info(f"State dim: {state_dim}, Action dim (n_assets): {action_dim}")

    # Create agent config (use CLI args if provided, otherwise use config defaults)
    final_lr_actor = lr_actor if lr_actor is not None else config.rl.cql_sac.learning_rate_actor
    final_lr_critic = lr_critic if lr_critic is not None else config.rl.cql_sac.learning_rate_critic
    final_batch_size = batch_size if batch_size is not None else config.rl.cql_sac.batch_size

    logger.info(f"Learning rates - Actor: {final_lr_actor}, Critic: {final_lr_critic}")
    logger.info(f"Batch size: {final_batch_size}")

    agent_config = CQLSACConfig(
        n_assets=action_dim,
        state_dim=state_dim,
        lr_actor=final_lr_actor,
        lr_critic=final_lr_critic,
        lambda_cql=config.rl.cql_sac.lambda_cql,
        lambda_gp=config.rl.cql_sac.lambda_gp,
        tau=config.rl.cql_sac.tau,
        gamma=config.rl.cql_sac.gamma,
        batch_size=final_batch_size,
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

    # Resume from checkpoint if provided
    start_step = 1
    if resume_path and resume_path.exists():
        logger.info(f"Resuming from checkpoint: {resume_path}")
        agent.load(str(resume_path))
        # Extract step number from checkpoint filename (e.g., cql_sac_step_50000.pt)
        try:
            step_str = resume_path.stem.split('_')[-1]
            start_step = int(step_str) + 1
            logger.info(f"Resuming from step {start_step}")
        except (ValueError, IndexError):
            logger.warning("Could not parse step from checkpoint name, starting from step 1")
            start_step = 1

    # Preload alpha and macro data for fast training
    # This converts ~2,020 pandas ops per encode to O(1) numpy lookup
    # For dynamic universe, use all_symbols (all symbols ever in universe)
    symbols = train_data.get('all_symbols', train_data.get('symbols', [f'ASSET_{i}' for i in range(config.universe.top_n)]))
    preload_data_for_fast_training(
        agent=agent,
        symbols=symbols,
        timestamps=train_data['timestamps'],
        start_date=config.backtest.train_start,
        end_date=config.backtest.train_end,
        slot_symbols=train_data.get('slot_symbols'),  # (T, 20) - which symbols in each slot
    )

    # Fill replay buffer with historical data
    logger.info("Building replay buffer from historical data...")
    portfolio_dim = action_dim + 2  # weights + leverage_ratio + cash_ratio
    replay_buffer = ReplayBuffer(
        capacity=config.rl.cql_sac.replay_buffer_size,
        n_assets=action_dim,
        state_dim=state_dim,
        portfolio_dim=portfolio_dim,
        device=str(device),
    )

    # Collect transitions from environment
    state_dict, info = env.reset()
    current_timestamp = info.get('timestamp', None)

    while len(replay_buffer) < min(config.rl.cql_sac.replay_buffer_size, len(train_data['states']) * 0.9):
        # Use behavior policy (can be random or heuristic)
        action = env.action_space.sample()

        next_state_dict, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_timestamp = info.get('timestamp', None)

        # Extract market and portfolio states from dict
        replay_buffer.add(
            market_state=state_dict['market'],
            portfolio_state=state_dict['portfolio'],
            action=action,
            reward=reward,
            next_market_state=next_state_dict['market'],
            next_portfolio_state=next_state_dict['portfolio'],
            done=done,
            timestamp=current_timestamp,
            next_timestamp=next_timestamp,
        )

        if done:
            state_dict, info = env.reset()
            current_timestamp = info.get('timestamp', None)
        else:
            state_dict = next_state_dict
            current_timestamp = next_timestamp

        if len(replay_buffer) % 10000 == 0:
            logger.info(f"Replay buffer: {len(replay_buffer):,} transitions")

    logger.info(f"Replay buffer filled: {len(replay_buffer):,} transitions")

    # ========== FAST PATH SETUP: Precompute features for O(1) lookup ==========
    # This converts ~300ms/step build_state_dict() overhead to ~5ms numpy indexing
    slot_symbols = train_data.get('slot_symbols')
    if slot_symbols is not None and hasattr(agent, 'adapter') and agent.adapter is not None:
        logger.info("Setting up FAST training path...")

        # 1. Create timestamp -> index mapping
        timestamps = train_data['timestamps']
        ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

        # 2. Convert buffer's string timestamps to integer indices
        logger.info("Converting buffer timestamps to indices...")
        for i in range(replay_buffer.size):
            ts = replay_buffer.timestamps[i]
            next_ts = replay_buffer.next_timestamps[i]

            # Map to index (fallback to 0 if not found)
            replay_buffer.timestamp_indices[i] = ts_to_idx.get(ts, 0)
            replay_buffer.next_timestamp_indices[i] = ts_to_idx.get(next_ts, 0)

        # 3. Precompute ALL features into numpy arrays
        precomputed = precompute_features_for_buffer(
            agent=agent,
            timestamps=timestamps,
            slot_symbols=slot_symbols,
            n_slots=config.universe.top_n,
            embedding_dim=768,
            temporal_window=replay_buffer.temporal_window,
        )

        # 4. Set preloaded features in buffer (SHARED references, no copy)
        replay_buffer.set_preloaded_features(
            alphas=precomputed['alphas'],
            gdelt=precomputed['gdelt'],
            nostr=precomputed['nostr'],
            macro=precomputed['macro'],
        )

        logger.info("FAST training path ready!")
        logger.info(f"  Expected speedup: ~300ms/step -> ~25ms/step")
    else:
        logger.warning("FAST path not available - using SLOW build_state_dict() per step")
    # ========== END FAST PATH SETUP ==========

    # Setup TensorBoard
    tb_log_dir = checkpoint_dir / 'tensorboard'
    tb_log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    # Auto-start TensorBoard server
    tb_process, tb_port = start_tensorboard(tb_log_dir)

    # Training loop
    logger.info("Starting CQL-SAC training...")
    if start_step > 1:
        logger.info(f"Resuming from step {start_step}, target: {steps}")
    batch_size = config.rl.cql_sac.batch_size

    for step in range(start_step, steps + 1):
        # Sample batch
        batch = replay_buffer.sample(batch_size)

        # Update agent
        metrics = agent.update(batch)

        # TensorBoard logging (every step)
        writer.add_scalar('Loss/Critic', metrics.get('critic_loss', 0), step)
        writer.add_scalar('Loss/Actor', metrics.get('actor_loss', 0), step)
        writer.add_scalar('Loss/CQL', metrics.get('cql_loss', 0), step)
        writer.add_scalar('Loss/Total', metrics.get('total_loss', 0), step)

        # Batch reward statistics (offline data)
        batch_rewards = batch['rewards']
        if hasattr(batch_rewards, 'cpu'):
            batch_rewards = batch_rewards.cpu().numpy()
        elif hasattr(batch_rewards, 'numpy'):
            batch_rewards = batch_rewards.numpy()
        writer.add_scalar('Batch/Reward_Mean', float(batch_rewards.mean()), step)
        writer.add_scalar('Batch/Reward_Std', float(batch_rewards.std()), step)
        writer.add_scalar('Batch/Reward_Min', float(batch_rewards.min()), step)
        writer.add_scalar('Batch/Reward_Max', float(batch_rewards.max()), step)

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

        # Checkpoint per 10000
        if step % 5000 == 0:
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
    resume_path: Path = None,
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

    # Load validation data with DYNAMIC universe rebalancing
    logger.info("Loading validation data with DYNAMIC universe...")
    data_loader = TrainingDataLoader(
        data_dir=DATA_DIR,
        n_assets=config.universe.top_n,
    )

    # Use dynamic loading for realistic universe rotation
    val_data = data_loader.load_period_dynamic(
        start_date=config.backtest.val_start,
        end_date=config.backtest.val_end,
    )
    logger.info(f"Validation data loaded: {val_data['states'].shape[0]} timesteps")

    # Log dynamic universe stats
    if 'slot_changes' in val_data:
        rebalances = val_data['rebalance_mask'].sum()
        total_changes = val_data['slot_changes'].sum()
        logger.info(f"Dynamic universe: {rebalances} rebalance events, {total_changes} total slot changes")
        logger.info(f"All symbols in period: {len(val_data.get('all_symbols', []))}")

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
    # observation_space is Dict: {'market': (N, state_dim), 'portfolio': (N+2,)}
    market_space = env.observation_space['market']
    state_dim = market_space.shape[1]  # state_dim per asset
    action_dim = env.action_space.shape[0]  # N assets

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

    # Load pretrained weights if available (for fresh Phase 2 start)
    if pretrained_path and pretrained_path.exists() and not resume_path:
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        agent.load_pretrained(pretrained_path)

    # Resume from Phase 2 checkpoint if provided
    start_step = 0
    start_episode = 0
    if resume_path and resume_path.exists():
        logger.info(f"Resuming Phase 2 from checkpoint: {resume_path}")
        agent.load(str(resume_path))
        # Extract step number from checkpoint filename (e.g., ppo_cvar_step_20000.pt)
        try:
            step_str = resume_path.stem.split('_')[-1]
            start_step = int(step_str)
            # Estimate episode from steps (rough approximation)
            start_episode = start_step // config.rl.ppo_cvar.horizon
            logger.info(f"Resuming from step {start_step}, episode ~{start_episode}")
        except (ValueError, IndexError):
            logger.warning("Could not parse step from checkpoint filename, starting from 0")

    # Preload alpha and macro data for fast training
    # This converts ~2,020 pandas ops per encode to O(1) numpy lookup
    # For dynamic universe, use all_symbols (all symbols ever in universe)
    symbols = val_data.get('all_symbols', val_data.get('symbols', [f'ASSET_{i}' for i in range(config.universe.top_n)]))
    preload_data_for_fast_training(
        agent=agent,
        symbols=symbols,
        timestamps=val_data['timestamps'],
        start_date=config.backtest.val_start,
        end_date=config.backtest.val_end,
        slot_symbols=val_data.get('slot_symbols'),  # (T, 20) - which symbols in each slot
    )

    # Rollout buffer
    portfolio_dim = action_dim + 2  # weights + leverage_ratio + cash_ratio
    rollout_buffer = RolloutBuffer(
        horizon=config.rl.ppo_cvar.horizon,
        n_assets=action_dim,
        state_dim=state_dim,
        portfolio_dim=portfolio_dim,
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
    total_steps = start_step
    episode = start_episode

    while total_steps < steps:
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_costs = 0
        episode_turnover = 0

        # Collect rollout
        while not rollout_buffer.is_full():
            # Get action from policy
            # state is dict with 'market' and 'portfolio' keys
            market_state = state['market']
            portfolio_state = state['portfolio']
            action, log_prob, value = agent.select_action(market_state, portfolio_state)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition (RolloutBuffer expects market and portfolio separately)
            rollout_buffer.add(market_state, portfolio_state, action, reward, value, log_prob, done)

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
            market_state = state['market']
            portfolio_state = state['portfolio']
            last_value = agent.get_value(market_state, portfolio_state) if not done else 0
            rollout_buffer.compute_gae(last_value, agent.config.gamma, agent.config.gae_lambda)

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

        # Checkpoint (every 5000 steps, same as Phase 1)
        if total_steps % 5000 == 0 and total_steps > 0:
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
    parser.add_argument(
        '--lr-actor', type=float, default=None,
        help='Actor learning rate (Phase 1 CQL-SAC, default: 3e-4)'
    )
    parser.add_argument(
        '--lr-critic', type=float, default=None,
        help='Critic learning rate (Phase 1 CQL-SAC, default: 1e-3)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size (Phase 1 CQL-SAC, default: 384)'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume training from (Phase 1: cql_sac_step_*.pt, Phase 2: ppo_cvar_step_*.pt)'
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
        resume_path = Path(args.resume) if args.resume else None
        run_phase1_cql_sac(
            config, device, steps, checkpoint_dir / 'phase1', embedding_dir, args.n_envs,
            lr_actor=args.lr_actor, lr_critic=args.lr_critic, batch_size=args.batch_size,
            resume_path=resume_path
        )
    elif args.phase == 2:
        steps = args.steps or config.rl.ppo_cvar.training_steps
        pretrained_path = Path(args.pretrained) if args.pretrained else None
        resume_path = Path(args.resume) if args.resume else None
        run_phase2_ppo_cvar(
            config, device, steps, checkpoint_dir / 'phase2', pretrained_path,
            embedding_dir, args.n_envs, resume_path=resume_path
        )
    else:
        logger.error("Please specify --phase 1, --phase 2, or --all")
        sys.exit(1)


if __name__ == '__main__':
    main()
