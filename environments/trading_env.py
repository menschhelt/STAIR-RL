"""
Trading Environment - Gym-style RL environment for portfolio management.

Implements a multi-asset trading environment with:
- Continuous action space: Tanh outputs [-1, +1] per asset
- Long-short portfolio support
- NAV-based position sizing
- Transaction cost modeling
- Risk-adjusted rewards (CVaR, volatility penalties)

State design: (N_assets, 36 features) + (22-dim portfolio state)
Action space: Box(-1, 1, shape=(N_assets,))
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, field
import logging

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from environments.position_sizer import PositionSizer, PositionInfo
from backtesting.pnl_calculator import PnLCalculator


@dataclass
class EnvConfig:
    """Trading environment configuration."""
    n_assets: int = 20
    state_dim: int = 36
    target_leverage: float = 2.0

    # Transaction costs
    transaction_cost_rate: float = 0.0025  # 0.25% round trip
    slippage_rate: float = 0.0005  # 0.05% slippage

    # Funding rate (applied every 8 hours for futures)
    funding_rate_interval: int = 8 * 12  # 8 hours in 5-min bars

    # Reward parameters (aligned with paper specifications)
    reward_scale: float = 1.0
    lambda_vol: float = 0.5       # Volatility penalty weight
    lambda_dd: float = 2.0        # Drawdown penalty weight (paper: 2.0)
    lambda_tc: float = 1.0        # Transaction cost penalty weight (paper: 1.0)
    lambda_to: float = 0.1        # Turnover penalty weight (paper: 0.1)

    # Risk parameters
    max_drawdown_threshold: float = 0.2  # 20% max drawdown before penalty
    volatility_lookback: int = 100       # Rolling window for volatility (legacy)
    volatility_decay_rho: float = 0.94   # Exponential decay factor (paper: ρ=0.94)
    volatility_window_K: int = 20        # Exponential volatility window (paper: K=20)

    # Episode parameters
    initial_nav: float = 100_000.0
    episode_length: Optional[int] = None  # None = use full data
    nav_threshold: float = 0.3  # Terminate if NAV drops below this ratio (30% = 70% loss)


@dataclass
class PortfolioState:
    """Current portfolio state for RL agent."""
    weights: np.ndarray           # Current portfolio weights (N_assets,)
    leverage_ratio: float         # Current leverage / target
    cash_ratio: float             # Available cash / NAV
    nav: float                    # Current NAV
    margin_used: float            # Currently used margin
    unrealized_pnl: float         # Unrealized P&L
    realized_pnl: float           # Realized P&L (cumulative)
    peak_nav: float               # Peak NAV for drawdown calculation
    current_drawdown: float       # Current drawdown from peak


class TradingEnv(gym.Env):
    """
    Multi-asset trading environment for reinforcement learning.

    Observation space:
    - market_state: (N_assets, 36) - market features per asset
    - portfolio_state: (22,) - [weights(20), leverage_ratio, cash_ratio]

    Action space:
    - Box(-1, 1, shape=(N_assets,)) - target weights from Tanh activation

    Reward:
    - Risk-adjusted log returns with penalties for volatility,
      drawdown, and transaction costs
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        data: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Initialize trading environment.

        Args:
            config: Environment configuration
            data: Pre-loaded data dict containing:
                - 'states': (T, N_assets, D_features) market state tensor
                - 'returns': (T, N_assets) asset returns
                - 'prices': (T, N_assets) asset prices
                - 'timestamps': (T,) timestamps
                - 'funding_rates': (T, N_assets) funding rates (optional)
        """
        super().__init__()

        self.config = config or EnvConfig()
        self.data = data

        # Initialize position sizer
        self.position_sizer = PositionSizer(
            target_leverage=self.config.target_leverage
        )

        # Initialize PnL calculator (unified with BacktestEngine)
        self.pnl_calculator = PnLCalculator(
            transaction_cost_rate=self.config.transaction_cost_rate,
            slippage_rate=self.config.slippage_rate,
        )

        # Define spaces
        self._define_spaces()

        # Episode state
        self._portfolio: Optional[PortfolioState] = None
        self._step_idx: int = 0
        self._episode_returns: List[float] = []
        self._previous_drawdown: float = 0.0  # For incremental drawdown penalty

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)

    def _define_spaces(self):
        """Define observation and action spaces."""
        n_assets = self.config.n_assets
        state_dim = self.config.state_dim

        # Observation space: dict with market and portfolio states
        self.observation_space = spaces.Dict({
            'market': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(n_assets, state_dim),
                dtype=np.float32
            ),
            'portfolio': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(n_assets + 2,),  # weights + leverage_ratio + cash_ratio
                dtype=np.float32
            )
        })

        # Action space: continuous weights [-1, 1] per asset
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_assets,),
            dtype=np.float32
        )

    def set_data(self, data: Dict[str, np.ndarray]):
        """
        Set or update environment data.

        Args:
            data: Data dict with states, returns, prices, etc.
        """
        self.data = data
        self._validate_data()

    def _validate_data(self):
        """Validate loaded data."""
        if self.data is None:
            return

        required_keys = ['states', 'returns', 'prices']
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Missing required data key: {key}")

        T, N, D = self.data['states'].shape
        assert N == self.config.n_assets, f"Asset count mismatch: {N} vs {self.config.n_assets}"
        assert D == self.config.state_dim, f"State dim mismatch: {D} vs {self.config.state_dim}"

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options (e.g., 'start_idx' for specific start)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if self.data is None:
            raise RuntimeError("No data loaded. Call set_data() first.")

        # Determine start index
        if options and 'start_idx' in options:
            self._step_idx = options['start_idx']
        else:
            self._step_idx = 0

        # Initialize portfolio
        self._portfolio = PortfolioState(
            weights=np.zeros(self.config.n_assets, dtype=np.float32),
            leverage_ratio=0.0,
            cash_ratio=1.0,
            nav=self.config.initial_nav,
            margin_used=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            peak_nav=self.config.initial_nav,
            current_drawdown=0.0,
        )

        # Reset episode tracking
        self._episode_returns = []
        self._previous_drawdown = 0.0  # Reset for incremental DD penalty

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute one environment step.

        Args:
            action: Target portfolio weights [-1, 1]^N

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self._portfolio is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Get current and next prices/returns
        current_prices = self.data['prices'][self._step_idx]
        next_returns = self._get_next_returns()
        funding_rates = self._get_funding_rates()

        # Store previous weights for transaction cost and deadband
        prev_weights = self._portfolio.weights.copy()

        # Compute new positions (with deadband filter)
        position_info = self.position_sizer.compute_positions(
            nav=self._portfolio.nav,
            weights_raw=action,
            prices=current_prices,
            margin_used=self._portfolio.margin_used,
            current_weights=prev_weights,  # Pass current weights for deadband
        )

        # Update portfolio weights
        self._portfolio.weights = position_info.final_weights
        self._portfolio.leverage_ratio = position_info.leverage_ratio
        self._portfolio.cash_ratio = position_info.cash_ratio

        # Check for slot changes (dynamic universe rebalancing)
        slot_changes = self._get_slot_changes()
        rotation_cost = self._calculate_rotation_cost(slot_changes, prev_weights)

        # Compute portfolio return using PnLCalculator (unified with BacktestEngine)
        pnl_result = self.pnl_calculator.calculate_portfolio_return(
            weights=position_info.final_weights,
            asset_returns=next_returns,
            prev_weights=prev_weights,
            funding_rates=funding_rates,
            include_costs=True,
        )

        # Extract results
        port_return = pnl_result['net_return']
        turnover = pnl_result['turnover']
        transaction_cost = pnl_result['transaction_cost']

        # Calculate slippage separately (PnLCalculator uses transaction_cost only)
        slippage_cost = turnover * self.config.slippage_rate

        # Add rotation cost to total cost
        total_cost = transaction_cost + slippage_cost + rotation_cost

        # Deduct rotation cost from return
        port_return -= rotation_cost

        # Update NAV using PnLCalculator
        old_nav = self._portfolio.nav
        self._portfolio.nav = self.pnl_calculator.update_nav(
            prev_nav=old_nav,
            portfolio_return=port_return,
        )

        # Update peak and drawdown
        if self._portfolio.nav > self._portfolio.peak_nav:
            self._portfolio.peak_nav = self._portfolio.nav
        self._portfolio.current_drawdown = (
            (self._portfolio.peak_nav - self._portfolio.nav) / self._portfolio.peak_nav
        )

        # Track episode returns
        self._episode_returns.append(port_return)

        # Compute reward (using total cost = transaction + slippage)
        reward = self._compute_reward(
            port_return=port_return,
            transaction_cost=total_cost,  # Use total cost for reward penalty
            turnover=turnover,
        )

        # Advance step
        self._step_idx += 1

        # Check termination
        terminated = False
        truncated = False

        # Terminate if NAV drops below threshold
        if self._portfolio.nav < self.config.initial_nav * self.config.nav_threshold:
            terminated = True
            self.logger.warning(f"Episode terminated: NAV dropped to {self._portfolio.nav:.0f} (threshold: {self.config.nav_threshold:.0%})")

        # Truncate if end of data
        max_idx = len(self.data['returns']) - 1
        if self._step_idx >= max_idx:
            truncated = True

        # Episode length limit
        if self.config.episode_length and len(self._episode_returns) >= self.config.episode_length:
            truncated = True

        obs = self._get_observation()
        info = self._get_info()
        info['port_return'] = port_return
        info['transaction_cost'] = transaction_cost
        info['slippage_cost'] = slippage_cost
        info['rotation_cost'] = rotation_cost
        info['total_cost'] = total_cost
        info['turnover'] = turnover
        info['is_rebalance'] = slot_changes.any()
        info['slots_changed'] = slot_changes.sum()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        # Market state
        market_state = self.data['states'][self._step_idx].astype(np.float32)

        # Portfolio state: [weights, leverage_ratio, cash_ratio]
        portfolio_state = np.concatenate([
            self._portfolio.weights,
            [self._portfolio.leverage_ratio, self._portfolio.cash_ratio]
        ]).astype(np.float32)

        return {
            'market': market_state,
            'portfolio': portfolio_state,
        }

    def _get_info(self) -> Dict[str, Any]:
        """Get current info dict."""
        info = {
            'step': self._step_idx,
            'nav': self._portfolio.nav,
            'peak_nav': self._portfolio.peak_nav,
            'drawdown': self._portfolio.current_drawdown,
            'leverage_ratio': self._portfolio.leverage_ratio,
            'cash_ratio': self._portfolio.cash_ratio,
            'gross_exposure': np.abs(self._portfolio.weights).sum(),
            'net_exposure': self._portfolio.weights.sum(),
            'weights': self._portfolio.weights.copy(),  # For backtest logging
        }

        # Add timestamp if available in data
        if self.data is not None and 'timestamps' in self.data:
            timestamp = self.data['timestamps'][self._step_idx]
            # Convert to string if not already
            if isinstance(timestamp, str):
                info['timestamp'] = timestamp
            else:
                info['timestamp'] = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)

        return info

    def _get_next_returns(self) -> np.ndarray:
        """Get returns for next timestep."""
        next_idx = min(self._step_idx + 1, len(self.data['returns']) - 1)
        return self.data['returns'][next_idx]

    def _get_funding_rates(self) -> np.ndarray:
        """Get funding rates if available."""
        if 'funding_rates' not in self.data:
            return np.zeros(self.config.n_assets)

        # Funding rates are applied every 8 hours
        if self._step_idx % self.config.funding_rate_interval == 0:
            return self.data['funding_rates'][self._step_idx]
        return np.zeros(self.config.n_assets)

    def _get_slot_changes(self) -> np.ndarray:
        """
        Get slot changes for current timestep (dynamic universe rebalancing).

        Returns:
            Boolean array (N,) indicating which slots changed symbol
        """
        if 'slot_changes' not in self.data:
            return np.zeros(self.config.n_assets, dtype=bool)
        return self.data['slot_changes'][self._step_idx]

    def _calculate_rotation_cost(
        self,
        slot_changes: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Calculate rotation cost when universe rebalances.

        When a slot's symbol changes, we effectively:
        1. Close position in old symbol (transaction cost)
        2. Open position in new symbol (transaction cost)

        Args:
            slot_changes: Boolean array (N,) of which slots changed
            weights: Current portfolio weights (N,)

        Returns:
            Rotation cost as fraction of NAV
        """
        if not slot_changes.any():
            return 0.0

        # For each slot that changed, we pay 2x transaction cost (close + open)
        # on the absolute weight in that slot
        changed_weights = np.abs(weights[slot_changes])
        rotation_turnover = 2 * changed_weights.sum()

        # Apply transaction cost rate
        rotation_cost = rotation_turnover * (
            self.config.transaction_cost_rate + self.config.slippage_rate
        )

        return rotation_cost

    # NOTE: _compute_portfolio_return method removed - now using PnLCalculator
    # This eliminates code duplication with BacktestEngine

    def _compute_reward(
        self,
        port_return: float,
        transaction_cost: float,
        turnover: float,
    ) -> float:
        """
        Compute risk-adjusted reward (aligned with paper specifications).

        r_t = log(1 + R^port) - λ_vol × σ̂_t - λ_dd × ΔDD_t - λ_tc × TC_t - λ_to × TO_t

        Where:
        - σ̂_t = sqrt(Σ_{k=0}^{K-1} ρ^k (R^port_{t-k})^2)  (ρ=0.94, K=20)
        - ΔDD_t = DD_t - DD_{t-1}  (incremental drawdown)
        - TO_t = turnover

        Args:
            port_return: Portfolio return
            transaction_cost: Transaction cost
            turnover: Portfolio turnover

        Returns:
            Reward value
        """
        # Log return (more numerically stable)
        log_return = np.log1p(port_return)

        # Exponential decay volatility (paper: ρ=0.94, K=20)
        K = self.config.volatility_window_K
        rho = self.config.volatility_decay_rho

        if len(self._episode_returns) >= K:
            # Compute σ̂_t = sqrt(Σ_{k=0}^{K-1} ρ^k (R^port_{t-k})^2)
            recent_returns = np.array(self._episode_returns[-K:])
            weights = np.array([rho ** k for k in range(K)])
            weighted_sq_returns = weights * (recent_returns ** 2)
            volatility = np.sqrt(weighted_sq_returns.sum())
        else:
            # Not enough history, use simple std
            volatility = np.std(self._episode_returns) if len(self._episode_returns) > 1 else 0.0

        # Incremental drawdown penalty (paper: ΔDD_t = DD_t - DD_{t-1})
        current_dd = self._portfolio.current_drawdown
        delta_dd = max(0.0, current_dd - self._previous_drawdown)  # Only penalize increases

        # Update previous drawdown for next step
        self._previous_drawdown = current_dd

        # Turnover penalty (paper: λ_to = 0.1)
        turnover_penalty = turnover

        # Compute reward (aligned with paper)
        reward = (
            log_return
            - self.config.lambda_vol * volatility
            - self.config.lambda_dd * delta_dd
            - self.config.lambda_tc * transaction_cost
            - self.config.lambda_to * turnover_penalty
        )

        return reward * self.config.reward_scale

    def get_episode_stats(self) -> Dict[str, float]:
        """
        Get statistics for completed episode.

        Returns:
            Dict with performance metrics
        """
        if not self._episode_returns:
            return {}

        returns = np.array(self._episode_returns)

        # Basic stats
        total_return = (self._portfolio.nav / self.config.initial_nav) - 1
        mean_return = returns.mean()
        std_return = returns.std()

        # Sharpe ratio (annualized, assuming 5-min bars)
        # 252 days * 24 hours * 12 bars/hour = 72,576 bars/year
        bars_per_year = 252 * 24 * 12
        sharpe = (mean_return / std_return * np.sqrt(bars_per_year)
                  if std_return > 0 else 0.0)

        # Max drawdown
        nav_series = self.config.initial_nav * np.cumprod(1 + returns)
        peak_series = np.maximum.accumulate(nav_series)
        drawdowns = (peak_series - nav_series) / peak_series
        max_drawdown = drawdowns.max()

        # Win rate
        win_rate = (returns > 0).mean()

        return {
            'total_return': total_return,
            'mean_return': mean_return,
            'std_return': std_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_nav': self._portfolio.nav,
            'n_steps': len(returns),
        }

    def render(self, mode: str = 'human'):
        """Render environment state."""
        if mode != 'human':
            return

        info = self._get_info()
        print(f"Step {info['step']:,d} | "
              f"NAV: ${info['nav']:,.0f} | "
              f"DD: {info['drawdown']:.1%} | "
              f"Leverage: {info['leverage_ratio']:.2f}")

    def close(self):
        """Clean up resources."""
        pass


# ========== Vectorized Environment for Parallel Training ==========

class VectorizedTradingEnv:
    """
    Vectorized trading environment for parallel training.

    Runs multiple environment instances in parallel for faster
    sample collection during RL training.
    """

    def __init__(
        self,
        n_envs: int,
        config: Optional[EnvConfig] = None,
        data: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Initialize vectorized environment.

        Args:
            n_envs: Number of parallel environments
            config: Shared configuration
            data: Shared data
        """
        self.n_envs = n_envs
        self.envs = [TradingEnv(config=config, data=data) for _ in range(n_envs)]

    def reset(self) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
        """Reset all environments."""
        obs_list = []
        info_list = []

        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)

        # Stack observations
        stacked_obs = {
            'market': np.stack([o['market'] for o in obs_list]),
            'portfolio': np.stack([o['portfolio'] for o in obs_list]),
        }

        return stacked_obs, info_list

    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments.

        Args:
            actions: (n_envs, n_assets) actions

        Returns:
            Tuple of (obs, rewards, terminateds, truncateds, infos)
        """
        obs_list = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            obs_list.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

            # Auto-reset on termination
            if terminated or truncated:
                obs, _ = env.reset()
                obs_list[-1] = obs

        stacked_obs = {
            'market': np.stack([o['market'] for o in obs_list]),
            'portfolio': np.stack([o['portfolio'] for o in obs_list]),
        }

        return (
            stacked_obs,
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos,
        )


# ==============================================================================
# Hierarchical Trading Environment - Supports Meta-Controller + Portfolio Head
# ==============================================================================

@dataclass
class HierarchicalEnvConfig(EnvConfig):
    """Configuration for hierarchical trading environment."""
    # Frequency penalty for trading
    lambda_freq: float = 0.001   # Penalty for each trade decision

    # Conditional execution threshold
    trade_threshold: float = 0.5  # trade_prob > threshold triggers trade

    # Trading statistics
    track_trade_frequency: bool = True


class HierarchicalTradingEnv(TradingEnv):
    """
    Hierarchical Trading Environment with Meta-Controller.

    Extends TradingEnv to support hierarchical actions:
    - trade_prob: Probability of executing trade [0, 1]
    - weights: Target portfolio weights [-1, 1]^N

    When trade_prob > threshold:
        Execute rebalancing with transaction costs
    Otherwise:
        Hold current position (no costs)

    This enables the agent to learn WHEN to trade (timing)
    in addition to WHAT to trade (weights).
    """

    def __init__(
        self,
        config: Optional[HierarchicalEnvConfig] = None,
        data: Optional[Dict[str, np.ndarray]] = None,
        state_builder=None,  # HierarchicalStateBuilder instance
    ):
        """
        Initialize hierarchical trading environment.

        Args:
            config: Hierarchical environment config
            data: Pre-loaded data (for legacy mode)
            state_builder: HierarchicalStateBuilder for new architecture
        """
        self.hierarchical_config = config or HierarchicalEnvConfig()
        super().__init__(config=self.hierarchical_config, data=data)

        self.state_builder = state_builder

        # Track trading statistics
        self._trade_count: int = 0
        self._hold_count: int = 0
        self._total_steps: int = 0

        # Redefine action space for hierarchical actions
        self._define_hierarchical_spaces()

    def _define_hierarchical_spaces(self):
        """Define hierarchical action space."""
        n_assets = self.config.n_assets

        # Action space: Dict with trade_prob and weights
        self.action_space = spaces.Dict({
            'trade_prob': spaces.Box(
                low=0.0, high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            'weights': spaces.Box(
                low=-1.0, high=1.0,
                shape=(n_assets,),
                dtype=np.float32
            )
        })

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset with hierarchical state."""
        obs, info = super().reset(seed=seed, options=options)

        # Reset trading counters
        self._trade_count = 0
        self._hold_count = 0
        self._total_steps = 0

        return obs, info

    def step(
        self,
        action: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute hierarchical step with conditional trading.

        Args:
            action: Dict with:
                - 'trade_prob': (1,) or scalar - probability of trading
                - 'weights': (N,) - target portfolio weights

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self._portfolio is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Parse hierarchical action
        if isinstance(action, dict):
            trade_prob = np.atleast_1d(action['trade_prob'])[0]
            target_weights = np.clip(action['weights'], -1.0, 1.0)
        elif isinstance(action, tuple):
            trade_prob = float(action[0])
            target_weights = np.clip(action[1], -1.0, 1.0)
        else:
            # Fallback: treat as just weights (legacy mode)
            trade_prob = 1.0
            target_weights = np.clip(action, -1.0, 1.0)

        # Decide whether to trade
        trade_decision = trade_prob > self.hierarchical_config.trade_threshold

        # Update counters
        self._total_steps += 1
        if trade_decision:
            self._trade_count += 1
        else:
            self._hold_count += 1

        # Get market data
        current_prices = self.data['prices'][self._step_idx]
        next_returns = self._get_next_returns()
        funding_rates = self._get_funding_rates()
        prev_weights = self._portfolio.weights.copy()

        if trade_decision:
            # Execute trade with transaction costs
            position_info = self.position_sizer.compute_positions(
                nav=self._portfolio.nav,
                weights_raw=target_weights,
                prices=current_prices,
                margin_used=self._portfolio.margin_used,
                current_weights=prev_weights,
            )

            new_weights = position_info.final_weights
            turnover = np.abs(new_weights - prev_weights).sum()
            transaction_cost = turnover * self.config.transaction_cost_rate
            slippage_cost = turnover * self.config.slippage_rate
            total_cost = transaction_cost + slippage_cost

            # Update portfolio
            self._portfolio.weights = new_weights
            self._portfolio.leverage_ratio = position_info.leverage_ratio
            self._portfolio.cash_ratio = position_info.cash_ratio

        else:
            # Hold - no trading, no costs
            new_weights = prev_weights
            turnover = 0.0
            transaction_cost = 0.0
            slippage_cost = 0.0
            total_cost = 0.0

        # Compute portfolio return
        port_return = self._compute_portfolio_return(
            weights=new_weights,
            returns=next_returns,
            funding_rates=funding_rates,
            transaction_cost=total_cost,
        )

        # Update NAV
        old_nav = self._portfolio.nav
        self._portfolio.nav *= (1 + port_return)

        # Update peak and drawdown
        if self._portfolio.nav > self._portfolio.peak_nav:
            self._portfolio.peak_nav = self._portfolio.nav
        self._portfolio.current_drawdown = (
            (self._portfolio.peak_nav - self._portfolio.nav) / self._portfolio.peak_nav
        )

        # Track returns
        self._episode_returns.append(port_return)

        # Compute hierarchical reward (includes frequency penalty)
        reward = self._compute_hierarchical_reward(
            port_return=port_return,
            transaction_cost=total_cost,
            turnover=turnover,
            trade_prob=trade_prob,
            trade_decision=trade_decision,
        )

        # Advance step
        self._step_idx += 1

        # Check termination
        terminated = False
        truncated = False

        if self._portfolio.nav < self.config.initial_nav * 0.5:
            terminated = True

        max_idx = len(self.data['returns']) - 1
        if self._step_idx >= max_idx:
            truncated = True

        if self.config.episode_length and len(self._episode_returns) >= self.config.episode_length:
            truncated = True

        obs = self._get_observation()
        info = self._get_info()
        info.update({
            'port_return': port_return,
            'transaction_cost': transaction_cost,
            'slippage_cost': slippage_cost,
            'total_cost': total_cost,
            'turnover': turnover,
            'trade_decision': trade_decision,
            'trade_prob': trade_prob,
            'trade_count': self._trade_count,
            'hold_count': self._hold_count,
            'trade_frequency': self._trade_count / max(1, self._total_steps),
        })

        return obs, reward, terminated, truncated, info

    def _compute_hierarchical_reward(
        self,
        port_return: float,
        transaction_cost: float,
        turnover: float,
        trade_prob: float,
        trade_decision: bool,
    ) -> float:
        """
        Compute hierarchical reward with frequency penalty.

        r_t = log(1 + R^port) - λ_vol × σ - λ_dd × DD - λ_tc × TC - λ_freq × trade_prob

        The frequency penalty discourages unnecessary trading,
        encouraging the agent to learn optimal trade timing.
        """
        # Base reward from parent
        base_reward = self._compute_reward(
            port_return=port_return,
            transaction_cost=transaction_cost,
            turnover=turnover,
        )

        # Frequency penalty: penalize high trade probability
        # This encourages the agent to be selective about when to trade
        freq_penalty = self.hierarchical_config.lambda_freq * trade_prob

        return base_reward - freq_penalty

    def get_episode_stats(self) -> Dict[str, float]:
        """Get episode stats including trading frequency."""
        stats = super().get_episode_stats()

        # Add hierarchical-specific stats
        if self._total_steps > 0:
            stats['trade_count'] = self._trade_count
            stats['hold_count'] = self._hold_count
            stats['trade_frequency'] = self._trade_count / self._total_steps
            stats['avg_trades_per_day'] = (
                self._trade_count / max(1, self._total_steps) * 288  # 288 5-min bars per day
            )

        return stats


class VectorizedHierarchicalEnv:
    """
    Vectorized version of HierarchicalTradingEnv.

    For parallel sample collection during training.
    """

    def __init__(
        self,
        n_envs: int,
        config: Optional[HierarchicalEnvConfig] = None,
        data: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.n_envs = n_envs
        self.envs = [
            HierarchicalTradingEnv(config=config, data=data)
            for _ in range(n_envs)
        ]

    def reset(self) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
        """Reset all environments."""
        obs_list = []
        info_list = []

        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)

        stacked_obs = {
            'market': np.stack([o['market'] for o in obs_list]),
            'portfolio': np.stack([o['portfolio'] for o in obs_list]),
        }

        return stacked_obs, info_list

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments with hierarchical actions.

        Args:
            actions: Dict with:
                - 'trade_prob': (n_envs, 1)
                - 'weights': (n_envs, n_assets)
        """
        obs_list = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for i, env in enumerate(self.envs):
            action = {
                'trade_prob': actions['trade_prob'][i],
                'weights': actions['weights'][i],
            }
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

            if terminated or truncated:
                obs, _ = env.reset()
                obs_list[-1] = obs

        stacked_obs = {
            'market': np.stack([o['market'] for o in obs_list]),
            'portfolio': np.stack([o['portfolio'] for o in obs_list]),
        }

        return (
            stacked_obs,
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos,
        )

    def get_aggregate_stats(self) -> Dict[str, float]:
        """Get aggregate stats across all environments."""
        all_stats = [env.get_episode_stats() for env in self.envs]
        if not all_stats or not all_stats[0]:
            return {}

        # Average stats
        keys = all_stats[0].keys()
        avg_stats = {}
        for key in keys:
            values = [s.get(key, 0) for s in all_stats if key in s]
            if values:
                avg_stats[f'avg_{key}'] = np.mean(values)

        return avg_stats


# ========== Standalone Testing ==========

if __name__ == '__main__':
    # Create dummy data for testing
    T = 1000  # timesteps
    N = 20    # assets
    D = 36    # features (including has_signal)

    np.random.seed(42)

    dummy_data = {
        'states': np.random.randn(T, N, D).astype(np.float32),
        'returns': np.random.randn(T, N).astype(np.float32) * 0.01,  # 1% daily vol
        'prices': np.abs(np.random.randn(T, N) * 1000 + 5000).astype(np.float32),
        'funding_rates': np.random.randn(T, N).astype(np.float32) * 0.0001,
    }

    # Test environment
    env = TradingEnv()
    env.set_data(dummy_data)

    obs, info = env.reset()
    print(f"Initial observation shapes:")
    print(f"  market: {obs['market'].shape}")
    print(f"  portfolio: {obs['portfolio'].shape}")
    print(f"Initial info: {info}")

    # Run a few steps
    total_reward = 0
    for step in range(100):
        action = np.random.uniform(-0.5, 0.5, N)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 20 == 0:
            env.render()

        if terminated or truncated:
            break

    print(f"\nEpisode stats:")
    stats = env.get_episode_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
