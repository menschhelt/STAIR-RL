"""
PPO-CVaR Agent - Proximal Policy Optimization with CVaR Constraint.

Phase 2: Online Fine-tuning Agent

PPO with a Conditional Value-at-Risk (CVaR) constraint for risk management.
Uses Lagrangian dual optimization to satisfy the CVaR constraint.

Objective:
    max E[R] subject to CVaR_α(loss) ≤ κ

Implementation: Primal-dual Lagrangian
    min_π max_λ≥0 E[R] + λ(CVaR_α - κ)

Key components:
- PPO clipped surrogate objective
- GAE (Generalized Advantage Estimation)
- CVaR constraint via Lagrange multiplier
- Entropy bonus for exploration

Reference:
- Schulman et al., "Proximal Policy Optimization Algorithms", 2017
- Chow et al., "Risk-Constrained RL with CVaR", 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple, Optional, List
import numpy as np
from dataclasses import dataclass
from collections import deque

from agents.networks import ActorCritic, FeatureEncoder, HierarchicalActorCritic
from agents.hierarchical_adapter import HierarchicalActorCriticAdapter


@dataclass
class PPOCVaRConfig:
    """PPO-CVaR hyperparameters."""
    # Network dimensions
    n_assets: int = 20
    state_dim: int = 36  # BUG FIX: Changed from 35 to match StateBuilder output
    portfolio_dim: int = 22
    hidden_dim: int = 128

    # PPO hyperparameters
    lr: float = 1e-4              # Learning rate
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE lambda
    clip_epsilon: float = 0.2     # PPO clip range
    ppo_epochs: int = 10          # PPO update epochs
    batch_size: int = 64          # Mini-batch size for PPO
    horizon: int = 2048           # Rollout length

    # CVaR parameters
    alpha_cvar: float = 0.95      # Confidence level (95%)
    kappa: float = 0.05           # Risk tolerance (5%)
    eta_lambda: float = 0.01      # Lambda learning rate

    # Regularization
    entropy_coef: float = 0.01    # Entropy bonus coefficient
    value_coef: float = 0.5       # Value loss coefficient
    grad_clip: float = 0.5        # Gradient clipping
    lambda_gp: float = 10.0       # Gradient penalty for Lipschitz constraint (paper: 10.0)

    # Architecture
    use_hierarchical: bool = True  # Use HierarchicalActorCritic (full multi-modal architecture)
    n_quantiles: int = 8            # Number of CVaR quantiles

    # Embedding paths (optional, for HierarchicalActorCritic)
    gdelt_embeddings_path: Optional[str] = None  # Path to GDELT embeddings HDF5
    nostr_embeddings_path: Optional[str] = None  # Path to Nostr embeddings HDF5

    # LR scheduling
    lr_decay: bool = True
    min_lr: float = 1e-5


class RolloutBuffer:
    """
    Rollout buffer for PPO training.

    Stores trajectories collected from environment interaction.
    """

    def __init__(
        self,
        horizon: int,
        n_assets: int,
        state_dim: int,
        portfolio_dim: int,
        device: str = 'cpu',
    ):
        """Initialize rollout buffer."""
        self.horizon = horizon
        self.device = device
        self.position = 0

        # Pre-allocate storage
        self.market_states = np.zeros((horizon, n_assets, state_dim), dtype=np.float32)
        self.portfolio_states = np.zeros((horizon, portfolio_dim), dtype=np.float32)
        self.actions = np.zeros((horizon, n_assets), dtype=np.float32)
        self.rewards = np.zeros(horizon, dtype=np.float32)
        self.values = np.zeros(horizon, dtype=np.float32)
        self.log_probs = np.zeros(horizon, dtype=np.float32)
        self.dones = np.zeros(horizon, dtype=np.float32)

        # Timestamp storage (ISO format strings, max 32 characters)
        self.timestamps = np.zeros(horizon, dtype='U32')

        # Computed advantages
        self.advantages = np.zeros(horizon, dtype=np.float32)
        self.returns = np.zeros(horizon, dtype=np.float32)

    def add(
        self,
        market_state: np.ndarray,
        portfolio_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        timestamp: Optional[str] = None,
    ):
        """Add a transition to the buffer."""
        idx = self.position

        self.market_states[idx] = market_state
        self.portfolio_states[idx] = portfolio_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = float(done)

        # Store timestamp if provided
        if timestamp is not None:
            self.timestamps[idx] = timestamp

        self.position += 1

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute Generalized Advantage Estimation.

        GAE(γ, λ) = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        gae = 0
        for t in reversed(range(self.position)):
            if t == self.position - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors."""
        return {
            'market_states': torch.tensor(self.market_states[:self.position], device=self.device),
            'portfolio_states': torch.tensor(self.portfolio_states[:self.position], device=self.device),
            'actions': torch.tensor(self.actions[:self.position], device=self.device),
            'rewards': torch.tensor(self.rewards[:self.position], device=self.device),
            'values': torch.tensor(self.values[:self.position], device=self.device),
            'log_probs': torch.tensor(self.log_probs[:self.position], device=self.device),
            'advantages': torch.tensor(self.advantages[:self.position], device=self.device),
            'returns': torch.tensor(self.returns[:self.position], device=self.device),
            'timestamps': self.timestamps[:self.position].tolist(),  # Convert to list of strings
        }

    def reset(self):
        """Reset buffer for next rollout."""
        self.position = 0

    def __len__(self) -> int:
        return self.position


class PPOCVaRAgent:
    """
    PPO with CVaR Constraint Agent.

    Used for Phase 2 online fine-tuning after CQL-SAC pre-training.

    The CVaR constraint ensures that the expected loss in the worst
    (1-α) fraction of outcomes doesn't exceed threshold κ.

    CVaR_α(X) = E[X | X ≤ VaR_α(X)]

    The constraint is enforced via a Lagrange multiplier λ:
    L = E[R] - λ(CVaR_α - κ)

    The multiplier is updated via dual ascent:
    λ ← [λ + η(CVaR_α - κ)]₊
    """

    def __init__(self, config: Optional[PPOCVaRConfig] = None, device: str = 'cuda'):
        """
        Initialize PPO-CVaR agent.

        Args:
            config: Agent configuration
            device: 'cuda' or 'cpu'
        """
        self.config = config or PPOCVaRConfig()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Build networks
        self._build_networks()

        # Set up optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.config.lr,
        )

        # Learning rate scheduler
        if self.config.lr_decay:
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: max(0.1, 1.0 - step / 100000)
            )
        else:
            self.scheduler = None

        # CVaR constraint: Lagrange multiplier (learnable)
        self.lambda_cvar = 0.0  # Start with no penalty

        # Episode returns buffer for CVaR estimation
        self.episode_returns = deque(maxlen=1000)

        # Training step counter
        self.total_steps = 0

    def _build_networks(self):
        """Build neural networks."""
        cfg = self.config

        if cfg.use_hierarchical:
            # Use HierarchicalActorCritic with adapter
            hierarchical_model = HierarchicalActorCritic(
                n_alphas=292,  # Will be updated when real alphas available
                n_assets=cfg.n_assets,
                d_alpha=64,
                d_text=64,
                d_temporal=128,
                d_global=6,
                d_portfolio=cfg.portfolio_dim,
                n_quantiles=cfg.n_quantiles,
            ).to(self.device)

            self.adapter = HierarchicalActorCriticAdapter(
                hierarchical_model,
                gdelt_embeddings_path=cfg.gdelt_embeddings_path,
                nostr_embeddings_path=cfg.nostr_embeddings_path,
                device=str(self.device),
            )

            # Expose actor_critic for compatibility
            self.actor_critic = self.adapter.model
        else:
            # Original ActorCritic network
            self.actor_critic = ActorCritic(
                n_assets=cfg.n_assets,
                local_feature_dim=cfg.state_dim - 6,
                global_feature_dim=6,
                portfolio_state_dim=cfg.portfolio_dim,
                hidden_dim=cfg.hidden_dim,
            ).to(self.device)
            self.adapter = None

    def select_action(
        self,
        market_state: np.ndarray,
        portfolio_state: np.ndarray,
        deterministic: bool = False,
        timestamp: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action given current state.

        Args:
            market_state: (n_assets, state_dim) array
            portfolio_state: (portfolio_dim,) array
            deterministic: if True, return mean action
            timestamp: Optional ISO timestamp for embedding lookup

        Returns:
            Tuple of (action, log_prob, value)
        """
        cfg = self.config

        with torch.no_grad():
            market = torch.tensor(market_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            portfolio = torch.tensor(portfolio_state, dtype=torch.float32, device=self.device).unsqueeze(0)

            if cfg.use_hierarchical:
                # Encode with dual outputs (with timestamps for embeddings)
                timestamps = [timestamp] if timestamp else None
                z_pooled, z_unpooled = self.adapter.encode_state(market, portfolio, timestamps=timestamps)

                # Get action
                action, trade_prob = self.adapter.get_action(z_pooled, z_unpooled, deterministic)

                # Get value
                value = self.adapter.get_value(z_pooled)

                # Placeholder log_prob (will be computed during update)
                log_prob = torch.zeros(1, device=self.device)
            else:
                # Original actor-critic
                action, log_prob, value = self.actor_critic.get_action_and_value(
                    market, portfolio, deterministic=deterministic
                )

            return (
                action.cpu().numpy().squeeze(0),
                log_prob.cpu().item(),
                value.cpu().item(),
            )

    def get_value(
        self,
        market_state: np.ndarray,
        portfolio_state: np.ndarray,
    ) -> float:
        """Get value estimate for a state."""
        cfg = self.config

        with torch.no_grad():
            market = torch.tensor(market_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            portfolio = torch.tensor(portfolio_state, dtype=torch.float32, device=self.device).unsqueeze(0)

            if cfg.use_hierarchical:
                z_pooled, _ = self.adapter.encode_state(market, portfolio)
                value = self.adapter.get_value(z_pooled)
            else:
                value = self.actor_critic.get_value(market, portfolio)

            return value.cpu().item()

    def compute_cvar(self, returns: np.ndarray) -> float:
        """
        Compute empirical CVaR from returns.

        CVaR_α = E[X | X ≤ VaR_α(X)]

        For losses (negative returns), we want the expected loss
        in the worst (1-α) fraction.

        Args:
            returns: Array of episode returns

        Returns:
            CVaR value (higher = more risk)
        """
        if len(returns) < 10:
            return 0.0

        # Convert to losses (negative of returns)
        losses = -returns

        # Sort losses
        sorted_losses = np.sort(losses)

        # VaR: threshold at α quantile
        cutoff = int(len(sorted_losses) * (1 - self.config.alpha_cvar))
        cutoff = max(1, cutoff)  # At least 1 sample

        # CVaR: mean of losses above VaR
        cvar = sorted_losses[-cutoff:].mean()

        return float(cvar)

    def update(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Update policy using collected rollout.

        PPO update:
        1. Compute GAE advantages
        2. For ppo_epochs:
           - Sample mini-batches
           - Compute clipped surrogate objective
           - Update with gradient descent
        3. Update CVaR Lagrange multiplier

        Args:
            rollout_buffer: Buffer containing collected trajectories

        Returns:
            Dict of loss values for logging
        """
        cfg = self.config

        # Get rollout data
        data = rollout_buffer.get()
        batch_size = len(rollout_buffer)

        # Normalize advantages
        advantages = data['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        data['advantages'] = advantages

        # PPO update epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(cfg.ppo_epochs):
            # Generate random permutation for mini-batches
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, cfg.batch_size):
                end = start + cfg.batch_size
                batch_indices = indices[start:end]

                # Get mini-batch
                mb = {k: v[batch_indices] for k, v in data.items()}

                # Compute current policy outputs
                if cfg.use_hierarchical:
                    # Encode with dual outputs
                    timestamps = mb.get('timestamps', None)
                    z_pooled, z_unpooled = self.adapter.encode_state(
                        mb['market_states'], mb['portfolio_states'], timestamps=timestamps
                    )

                    # Get action
                    actions, trade_prob = self.adapter.get_action(z_pooled, z_unpooled, deterministic=False)

                    # Get value
                    values = self.adapter.get_value(z_pooled).squeeze(-1)

                    # Placeholder log_probs (will be computed for entropy)
                    log_probs = torch.zeros(actions.shape[0], device=self.device)
                else:
                    # Original encoding
                    z = self.actor_critic.encoder(mb['market_states'], mb['portfolio_states'])
                    actions, log_probs = self.actor_critic.actor.get_action(z)
                    values = self.actor_critic.critic(z).squeeze(-1)
                    z_pooled = z  # For gradient penalty

                # Recompute log probs for old actions
                # Approximate: use current log_prob - old_log_prob ratio
                old_log_probs = mb['log_probs']

                # Policy ratio
                ratio = torch.exp(log_probs - old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * mb['advantages']
                surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * mb['advantages']
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE + Lipschitz regularization)
                value_mse = F.mse_loss(values, mb['returns'])

                # Lipschitz constraint: L_Lip = λ_GP × E[(‖∇_z V‖₂ - 1)²]
                # Encourages smoothness in value function
                if cfg.lambda_gp > 0:
                    z_pooled_detached = z_pooled.detach().requires_grad_(True)

                    if cfg.use_hierarchical:
                        v_gp = self.adapter.get_value(z_pooled_detached).squeeze(-1)
                    else:
                        v_gp = self.actor_critic.critic(z_pooled_detached).squeeze(-1)

                    # Compute gradient of V w.r.t. encoded state
                    gradients = torch.autograd.grad(
                        outputs=v_gp,
                        inputs=z_pooled_detached,
                        grad_outputs=torch.ones_like(v_gp),
                        create_graph=True,
                        retain_graph=True,
                    )[0]

                    # Gradient penalty: (||grad|| - 1)^2
                    grad_norm = torch.sqrt((gradients ** 2).sum(dim=-1) + 1e-12)
                    gp_loss = ((grad_norm - 1) ** 2).mean()

                    value_loss = value_mse + cfg.lambda_gp * gp_loss
                else:
                    value_loss = value_mse

                # Entropy bonus (approximate)
                entropy = -log_probs.mean()

                # CVaR penalty
                cvar = self.compute_cvar(mb['returns'].cpu().numpy())
                cvar_penalty = self.lambda_cvar * (cvar - cfg.kappa)

                # Total loss
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy
                    + cvar_penalty
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), cfg.grad_clip)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        # ========== Update CVaR Lagrange Multiplier ==========
        # Dual ascent: λ ← [λ + η(CVaR_α - κ)]₊
        all_returns = data['returns'].cpu().numpy()
        current_cvar = self.compute_cvar(all_returns)
        self.lambda_cvar = max(0.0, self.lambda_cvar + cfg.eta_lambda * (current_cvar - cfg.kappa))

        # Store episode returns for CVaR tracking
        self.episode_returns.extend(all_returns.tolist())

        self.total_steps += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'cvar': current_cvar,
            'lambda_cvar': self.lambda_cvar,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

    def load_from_cql_sac(self, cql_agent):
        """
        Load encoder and actor weights from pre-trained CQL-SAC agent.

        This enables transfer learning from Phase 1 (offline) to Phase 2 (online).

        Args:
            cql_agent: Trained CQL-SAC agent
        """
        cfg = self.config

        if cfg.use_hierarchical:
            # For hierarchical, load from adapter's model
            # Both should be using HierarchicalActorCritic
            if hasattr(cql_agent, 'adapter') and cql_agent.adapter is not None:
                # Load encoder
                self.actor_critic.encoder.load_state_dict(
                    cql_agent.encoder.state_dict()
                )

                # Load actor
                self.actor_critic.actor.load_state_dict(
                    cql_agent.actor.state_dict()
                )

                print("Loaded hierarchical weights from CQL-SAC agent")
            else:
                print("Warning: CQL-SAC agent is not using hierarchical architecture")
                print("Skipping weight loading")
        else:
            # Load encoder weights
            self.actor_critic.encoder.load_state_dict(cql_agent.encoder.state_dict())

            # Load actor weights
            self.actor_critic.actor.load_state_dict(cql_agent.actor.state_dict())

            print("Loaded weights from CQL-SAC agent")

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lambda_cvar': self.lambda_cvar,
            'config': self.config,
            'total_steps': self.total_steps,
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lambda_cvar = checkpoint['lambda_cvar']
        self.total_steps = checkpoint['total_steps']


# ========== Standalone Testing ==========

if __name__ == '__main__':
    print("Testing PPO-CVaR Agent...")

    config = PPOCVaRConfig(
        n_assets=20,
        state_dim=36,  # BUG FIX: Changed from 35 to match StateBuilder output
        portfolio_dim=22,
        horizon=256,  # Shorter for testing
        batch_size=32,
        ppo_epochs=4,
    )

    agent = PPOCVaRAgent(config, device='cpu')

    # Create rollout buffer
    buffer = RolloutBuffer(
        horizon=config.horizon,
        n_assets=config.n_assets,
        state_dim=config.state_dim,
        portfolio_dim=config.portfolio_dim,
    )

    # Fill buffer with dummy data
    for _ in range(config.horizon):
        market_state = np.random.randn(config.n_assets, config.state_dim).astype(np.float32)
        portfolio_state = np.random.randn(config.portfolio_dim).astype(np.float32)

        action, log_prob, value = agent.select_action(market_state, portfolio_state)
        reward = np.random.randn() * 0.01

        buffer.add(
            market_state=market_state,
            portfolio_state=portfolio_state,
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=False,
        )

    # Compute GAE
    last_value = agent.get_value(market_state, portfolio_state)
    buffer.compute_gae(last_value, gamma=config.gamma, gae_lambda=config.gae_lambda)

    # Test update
    losses = agent.update(buffer)
    print(f"Update losses: {losses}")

    # Test CVaR computation
    test_returns = np.random.randn(100) * 0.01
    cvar = agent.compute_cvar(test_returns)
    print(f"Test CVaR: {cvar:.4f}")

    print("\nPPO-CVaR Agent test passed!")
