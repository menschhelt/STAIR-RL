"""
Vectorized Environment Wrapper for Parallel Training

Enables parallel environment execution using SubprocVecEnv to maximize
GPU utilization by parallelizing CPU-bound env.step() operations.

Usage:
    envs = make_vec_env(
        env_config=env_config,
        n_envs=64,
        device='cuda:0',
    )

    # Standard vectorized env interface
    states = envs.reset()  # (n_envs, N, state_dim)
    actions = agent.select_action(states)  # (n_envs, N)
    next_states, rewards, dones, infos = envs.step(actions)
"""

import multiprocessing as mp
from typing import Callable, List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environments.trading_env import TradingEnv


def make_env(
    env_id: int,
    env_config: dict,
    seed: Optional[int] = None,
) -> Callable:
    """
    Create a single environment instance.

    Args:
        env_id: Environment ID for seeding
        env_config: Environment configuration dict
        seed: Random seed (if None, use env_id)

    Returns:
        Callable that creates the environment
    """
    def _init():
        env = TradingEnv(**env_config)

        # Set seed for reproducibility
        if seed is not None:
            env.seed(seed + env_id)
        else:
            env.seed(env_id)

        return env

    return _init


class SubprocVecEnv:
    """
    Vectorized environment using subprocess for parallel execution.

    Each environment runs in a separate process, enabling true parallelism
    for CPU-bound operations (data loading, state building, reward calculation).

    Interface compatible with stable-baselines3 VecEnv.
    """

    def __init__(
        self,
        env_fns: List[Callable],
        start_method: Optional[str] = None,
    ):
        """
        Initialize vectorized environment.

        Args:
            env_fns: List of environment creation functions
            start_method: Multiprocessing start method ('fork', 'spawn', 'forkserver')
        """
        self.n_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        # Set multiprocessing start method
        if start_method is None:
            start_method = 'forkserver' if hasattr(mp, 'get_context') else 'spawn'

        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])

        # Start worker processes
        ctx = mp.get_context(start_method)
        self.ps = [
            ctx.Process(
                target=worker,
                args=(work_remote, remote, env_fn),
                daemon=True,
            )
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]

        for p in self.ps:
            p.start()

        for remote in self.work_remotes:
            remote.close()

        # Get observation and action spaces
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all environments.

        Returns:
            observations: (n_envs, N, state_dim) array of initial states
        """
        for remote in self.remotes:
            remote.send(('reset', None))

        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def step(self, actions):
        """
        Step all environments.

        Args:
            actions: (n_envs, N) array of actions

        Returns:
            observations: (n_envs, N, state_dim)
            rewards: (n_envs,)
            dones: (n_envs,)
            infos: List of dicts (length n_envs)
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        obs, rewards, dones, infos = zip(*results)

        return np.stack(obs), np.array(rewards), np.array(dones), list(infos)

    def close(self):
        """Close all environments and terminate worker processes."""
        if self.closed:
            return

        if self.waiting:
            for remote in self.remotes:
                remote.recv()

        for remote in self.remotes:
            remote.send(('close', None))

        for p in self.ps:
            p.join()

        self.closed = True

    def get_attr(self, attr_name: str, indices=None):
        """Get attribute from environments."""
        if indices is None:
            indices = range(self.n_envs)

        for i in indices:
            self.remotes[i].send(('get_attr', attr_name))

        return [self.remotes[i].recv() for i in indices]

    def set_attr(self, attr_name: str, value, indices=None):
        """Set attribute in environments."""
        if indices is None:
            indices = range(self.n_envs)

        for i in indices:
            self.remotes[i].send(('set_attr', (attr_name, value)))

        for i in indices:
            self.remotes[i].recv()

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        """Call method on environments."""
        if indices is None:
            indices = range(self.n_envs)

        for i in indices:
            self.remotes[i].send(('env_method', (method_name, args, kwargs)))

        return [self.remotes[i].recv() for i in indices]


def worker(remote, parent_remote, env_fn):
    """
    Worker process that runs a single environment.

    Args:
        remote: Communication pipe (worker side)
        parent_remote: Communication pipe (parent side) - closed immediately
        env_fn: Function to create environment
    """
    parent_remote.close()
    env = env_fn()

    try:
        while True:
            cmd, data = remote.recv()

            if cmd == 'step':
                obs, reward, done, info = env.step(data)

                # Auto-reset on done
                if done:
                    obs = env.reset()

                remote.send((obs, reward, done, info))

            elif cmd == 'reset':
                obs = env.reset()
                remote.send(obs)

            elif cmd == 'close':
                env.close()
                remote.close()
                break

            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))

            elif cmd == 'get_attr':
                remote.send(getattr(env, data))

            elif cmd == 'set_attr':
                attr_name, value = data
                setattr(env, attr_name, value)
                remote.send(None)

            elif cmd == 'env_method':
                method_name, args, kwargs = data
                method = getattr(env, method_name)
                result = method(*args, **kwargs)
                remote.send(result)

            else:
                raise NotImplementedError(f"Unknown command: {cmd}")

    except KeyboardInterrupt:
        print(f"Worker {env_fn}: Caught KeyboardInterrupt, closing...")
    finally:
        env.close()


def make_vec_env(
    env_config: dict,
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_method: Optional[str] = None,
) -> SubprocVecEnv:
    """
    Create vectorized environment.

    Args:
        env_config: Environment configuration dict
        n_envs: Number of parallel environments
        seed: Random seed
        start_method: Multiprocessing start method

    Returns:
        SubprocVecEnv instance

    Example:
        >>> env_config = {
        ...     'data_dir': '/path/to/data',
        ...     'n_assets': 20,
        ...     'initial_capital': 100_000.0,
        ... }
        >>> envs = make_vec_env(env_config, n_envs=64, seed=42)
        >>> obs = envs.reset()  # (64, 20, 36)
        >>> actions = agent.select_action(obs)
        >>> obs, rewards, dones, infos = envs.step(actions)
    """
    if n_envs == 1:
        # Single environment - no need for multiprocessing
        return DummyVecEnv([make_env(0, env_config, seed)])

    # Create environment functions
    env_fns = [make_env(i, env_config, seed) for i in range(n_envs)]

    return SubprocVecEnv(env_fns, start_method=start_method)


class DummyVecEnv:
    """
    Vectorized environment wrapper for single environment (no multiprocessing).

    Provides same interface as SubprocVecEnv but runs in main process.
    Useful for debugging.
    """

    def __init__(self, env_fns: List[Callable]):
        """Initialize dummy vectorized environment."""
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        """Reset all environments."""
        obs = [env.reset() for env in self.envs]
        return np.stack(obs)

    def step(self, actions):
        """Step all environments."""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        obs, rewards, dones, infos = zip(*results)

        # Auto-reset on done
        for i, done in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()

        return np.stack(obs), np.array(rewards), np.array(dones), list(infos)

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

    def get_attr(self, attr_name: str, indices=None):
        """Get attribute from environments."""
        if indices is None:
            indices = range(self.n_envs)
        return [getattr(self.envs[i], attr_name) for i in indices]

    def set_attr(self, attr_name: str, value, indices=None):
        """Set attribute in environments."""
        if indices is None:
            indices = range(self.n_envs)
        for i in indices:
            setattr(self.envs[i], attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        """Call method on environments."""
        if indices is None:
            indices = range(self.n_envs)
        return [getattr(self.envs[i], method_name)(*args, **kwargs) for i in indices]
