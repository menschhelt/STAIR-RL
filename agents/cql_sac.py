"""
CQL-SAC Agent - Conservative Q-Learning + Soft Actor-Critic.

Phase 1: Offline Pre-training Agent

CQL addresses the distribution shift problem in offline RL by adding
a regularizer that penalizes Q-values on out-of-distribution actions.

Key components:
- CQL regularization: penalize OOD actions
- SAC: entropy-regularized policy optimization
- Lipschitz constraint: smooth value function (gradient penalty)

Loss functions:
- L_critic = L_TD + λ_CQL × L_CQL + λ_GP × L_Lip
- L_actor = -E[min(Q1, Q2)(s, π(s)) - α × log π(a|s)]

Reference:
- Kumar et al., "Conservative Q-Learning for Offline RL", NeurIPS 2020
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple, Optional, List
import numpy as np
from dataclasses import dataclass

from agents.networks import FeatureEncoder, Actor, TwinCritic, soft_update


@dataclass
class CQLSACConfig:
    """CQL-SAC hyperparameters."""
    # Network dimensions
    n_assets: int = 20
    state_dim: int = 35
    portfolio_dim: int = 22
    hidden_dim: int = 128

    # Learning rates
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_alpha: float = 3e-4  # SAC temperature

    # SAC hyperparameters
    gamma: float = 0.99           # Discount factor
    tau: float = 0.005            # Target network soft update
    alpha_init: float = 0.2       # Initial SAC temperature
    auto_entropy_tuning: bool = True
    target_entropy: Optional[float] = None  # If None, use -action_dim

    # CQL hyperparameters
    lambda_cql: float = 1.0       # CQL regularization weight
    cql_n_actions: int = 10       # Number of actions to sample for CQL
    cql_temp: float = 1.0         # Temperature for logsumexp

    # Gradient penalty (Lipschitz constraint)
    lambda_gp: float = 10.0       # Gradient penalty weight

    # Training
    batch_size: int = 256
    grad_clip: float = 1.0        # Gradient clipping


class CQLSACAgent:
    """
    Conservative Q-Learning + Soft Actor-Critic Agent.

    Used for Phase 1 offline pre-training on historical data.

    The CQL objective adds a penalty that pushes down Q-values on
    actions sampled from the policy while pushing up Q-values on
    actions from the dataset.

    L_CQL = α × E[log Σ_a exp(Q(s,a)) - E_{a~β}[Q(s,a)]]

    This prevents overestimation on out-of-distribution actions.
    """

    def __init__(self, config: Optional[CQLSACConfig] = None, device: str = 'cuda'):
        """
        Initialize CQL-SAC agent.

        Args:
            config: Agent configuration
            device: 'cuda' or 'cpu'
        """
        self.config = config or CQLSACConfig()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Build networks
        self._build_networks()

        # Set up optimizers
        self._setup_optimizers()

        # SAC temperature (log for stability)
        self.log_alpha = torch.tensor(
            np.log(self.config.alpha_init),
            dtype=torch.float32,
            device=self.device,
            requires_grad=self.config.auto_entropy_tuning,
        )
        if self.config.auto_entropy_tuning:
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr_alpha)

        # Target entropy
        self.target_entropy = (
            self.config.target_entropy if self.config.target_entropy is not None
            else -self.config.n_assets  # Default: -action_dim
        )

        # Training step counter
        self.total_steps = 0

    def _build_networks(self):
        """Build neural networks."""
        cfg = self.config

        # Feature encoder
        self.encoder = FeatureEncoder(
            n_assets=cfg.n_assets,
            local_feature_dim=cfg.state_dim - 6,  # 29 local + 6 global = 35
            global_feature_dim=6,
            portfolio_state_dim=cfg.portfolio_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)

        # Actor (policy)
        self.actor = Actor(
            input_dim=self.encoder.output_dim,
            n_assets=cfg.n_assets,
        ).to(self.device)

        # Twin Q-networks
        self.critic = TwinCritic(
            state_dim=self.encoder.output_dim,
            action_dim=cfg.n_assets,
        ).to(self.device)

        # Target networks
        self.target_critic = copy.deepcopy(self.critic)

        # Freeze target networks
        for param in self.target_critic.parameters():
            param.requires_grad = False

    def _setup_optimizers(self):
        """Set up optimizers."""
        cfg = self.config

        # Combined encoder + actor optimizer
        self.actor_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()),
            lr=cfg.lr_actor,
        )

        # Critic optimizer
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=cfg.lr_critic,
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Current SAC temperature."""
        return self.log_alpha.exp()

    def encode_state(
        self,
        market_state: torch.Tensor,
        portfolio_state: torch.Tensor,
    ) -> torch.Tensor:
        """Encode state using feature encoder."""
        return self.encoder(market_state, portfolio_state)

    def select_action(
        self,
        market_state: np.ndarray,
        portfolio_state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select action given current state.

        Args:
            market_state: (n_assets, state_dim) array
            portfolio_state: (portfolio_dim,) array
            deterministic: if True, return mean action

        Returns:
            action: (n_assets,) portfolio weights
        """
        with torch.no_grad():
            # Add batch dimension and convert to tensor
            market = torch.tensor(market_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            portfolio = torch.tensor(portfolio_state, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Encode state
            z = self.encode_state(market, portfolio)

            # Get action
            if deterministic:
                action = self.actor(z)
            else:
                action, _ = self.actor.get_action(z)

            return action.cpu().numpy().squeeze(0)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update networks with a batch from replay buffer.

        Args:
            batch: Dict containing:
                - market_states: (batch, n_assets, state_dim)
                - portfolio_states: (batch, portfolio_dim)
                - actions: (batch, n_assets)
                - rewards: (batch,)
                - next_market_states: (batch, n_assets, state_dim)
                - next_portfolio_states: (batch, portfolio_dim)
                - dones: (batch,)

        Returns:
            Dict of loss values for logging
        """
        cfg = self.config

        # Unpack batch
        market_states = batch['market_states'].to(self.device)
        portfolio_states = batch['portfolio_states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_market_states = batch['next_market_states'].to(self.device)
        next_portfolio_states = batch['next_portfolio_states'].to(self.device)
        dones = batch['dones'].to(self.device)

        batch_size = market_states.shape[0]

        # Encode current and next states
        z = self.encode_state(market_states, portfolio_states)
        with torch.no_grad():
            z_next = self.encode_state(next_market_states, next_portfolio_states)

        # ========== Update Critic ==========
        critic_loss, cql_loss, td_loss = self._update_critic(
            z, actions, rewards, z_next, dones
        )

        # ========== Update Actor ==========
        actor_loss, alpha_loss = self._update_actor(z)

        # ========== Update Target Networks ==========
        soft_update(self.target_critic, self.critic, cfg.tau)

        self.total_steps += 1

        return {
            'critic_loss': critic_loss,
            'cql_loss': cql_loss,
            'td_loss': td_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha.item(),
        }

    def _update_critic(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        z_next: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """Update critic networks."""
        cfg = self.config
        batch_size = z.shape[0]

        # ========== TD Target ==========
        with torch.no_grad():
            # Sample next actions from policy
            next_actions, next_log_probs = self.actor.get_action(z_next)

            # Compute target Q-value
            target_q1, target_q2 = self.target_critic(z_next, next_actions)
            target_q = torch.min(target_q1, target_q2).squeeze(-1)

            # Entropy-regularized target
            target_q = target_q - self.alpha * next_log_probs

            # TD target
            td_target = rewards + cfg.gamma * (1 - dones) * target_q

        # Current Q-values
        current_q1, current_q2 = self.critic(z, actions)
        current_q1 = current_q1.squeeze(-1)
        current_q2 = current_q2.squeeze(-1)

        # TD loss
        td_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)

        # ========== CQL Regularization ==========
        # Sample random actions for logsumexp
        random_actions = torch.rand(
            cfg.cql_n_actions, batch_size, cfg.n_assets,
            device=self.device
        ) * 2 - 1  # [-1, 1]

        # Sample actions from current policy
        with torch.no_grad():
            policy_actions = torch.stack([
                self.actor.get_action(z)[0] for _ in range(cfg.cql_n_actions)
            ])  # (n_actions, batch, n_assets)

        # Compute Q-values for random and policy actions
        cql_q_random = []
        cql_q_policy = []

        for i in range(cfg.cql_n_actions):
            q1_r, q2_r = self.critic(z, random_actions[i])
            cql_q_random.append(torch.min(q1_r, q2_r).squeeze(-1))

            q1_p, q2_p = self.critic(z, policy_actions[i])
            cql_q_policy.append(torch.min(q1_p, q2_p).squeeze(-1))

        cql_q_random = torch.stack(cql_q_random, dim=0)  # (n_actions, batch)
        cql_q_policy = torch.stack(cql_q_policy, dim=0)  # (n_actions, batch)

        # CQL loss: logsumexp over random/policy actions - dataset Q-values
        all_q = torch.cat([cql_q_random, cql_q_policy], dim=0)  # (2*n_actions, batch)
        logsumexp_q = torch.logsumexp(all_q / cfg.cql_temp, dim=0) * cfg.cql_temp

        # Dataset Q-values (already computed as current_q)
        dataset_q = (current_q1 + current_q2) / 2

        cql_loss = (logsumexp_q - dataset_q).mean()

        # ========== Gradient Penalty (Lipschitz) ==========
        if cfg.lambda_gp > 0:
            gp_loss = self._gradient_penalty(z, actions)
        else:
            gp_loss = 0.0

        # Total critic loss
        total_loss = td_loss + cfg.lambda_cql * cql_loss
        if isinstance(gp_loss, torch.Tensor):
            total_loss = total_loss + cfg.lambda_gp * gp_loss

        # Optimize
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.critic_optimizer.step()

        return total_loss.item(), cql_loss.item(), td_loss.item()

    def _gradient_penalty(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient penalty for Lipschitz constraint."""
        z.requires_grad_(True)
        actions.requires_grad_(True)

        q1, q2 = self.critic(z, actions)
        q = (q1 + q2) / 2

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=q,
            inputs=[z, actions],
            grad_outputs=torch.ones_like(q),
            create_graph=True,
            retain_graph=True,
        )

        # Compute gradient penalty
        grad_z = gradients[0]
        grad_a = gradients[1]
        grad_norm = torch.sqrt(
            (grad_z ** 2).sum(dim=-1) + (grad_a ** 2).sum(dim=-1) + 1e-12
        )

        gp = ((grad_norm - 1) ** 2).mean()

        return gp

    def _update_actor(self, z: torch.Tensor) -> Tuple[float, float]:
        """Update actor network."""
        # Sample actions from current policy
        actions, log_probs = self.actor.get_action(z)

        # Q-values for sampled actions
        q1, q2 = self.critic(z, actions)
        q_value = torch.min(q1, q2).squeeze(-1)

        # Actor loss: maximize Q - α * entropy
        actor_loss = (self.alpha.detach() * log_probs - q_value).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.actor.parameters()),
                self.config.grad_clip
            )
        self.actor_optimizer.step()

        # ========== Update Temperature (Auto Entropy Tuning) ==========
        alpha_loss = 0.0
        if self.config.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss = alpha_loss.item()

        return actor_loss.item(), alpha_loss

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'log_alpha': self.log_alpha,
            'config': self.config,
            'total_steps': self.total_steps,
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.log_alpha = checkpoint['log_alpha']
        self.total_steps = checkpoint['total_steps']


# ========== Replay Buffer ==========

class ReplayBuffer:
    """
    Experience replay buffer for offline RL.

    Stores transitions and samples random batches for training.
    """

    def __init__(
        self,
        capacity: int,
        n_assets: int,
        state_dim: int,
        portfolio_dim: int,
        device: str = 'cpu',
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions
            n_assets: Number of assets
            state_dim: State feature dimension per asset
            portfolio_dim: Portfolio state dimension
            device: Storage device
        """
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        # Pre-allocate storage
        self.market_states = np.zeros((capacity, n_assets, state_dim), dtype=np.float32)
        self.portfolio_states = np.zeros((capacity, portfolio_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_assets), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_market_states = np.zeros((capacity, n_assets, state_dim), dtype=np.float32)
        self.next_portfolio_states = np.zeros((capacity, portfolio_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        market_state: np.ndarray,
        portfolio_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_market_state: np.ndarray,
        next_portfolio_state: np.ndarray,
        done: bool,
    ):
        """Add a transition to the buffer."""
        idx = self.position

        self.market_states[idx] = market_state
        self.portfolio_states[idx] = portfolio_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_market_states[idx] = next_market_state
        self.next_portfolio_states[idx] = next_portfolio_state
        self.dones[idx] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            'market_states': torch.tensor(self.market_states[indices], device=self.device),
            'portfolio_states': torch.tensor(self.portfolio_states[indices], device=self.device),
            'actions': torch.tensor(self.actions[indices], device=self.device),
            'rewards': torch.tensor(self.rewards[indices], device=self.device),
            'next_market_states': torch.tensor(self.next_market_states[indices], device=self.device),
            'next_portfolio_states': torch.tensor(self.next_portfolio_states[indices], device=self.device),
            'dones': torch.tensor(self.dones[indices], device=self.device),
        }

    def load_from_data(
        self,
        market_states: np.ndarray,
        portfolio_states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ):
        """
        Load pre-collected offline data into buffer.

        Args:
            market_states: (T, n_assets, state_dim)
            portfolio_states: (T, portfolio_dim)
            actions: (T, n_assets)
            rewards: (T,)
            dones: (T,)
        """
        T = len(rewards)
        assert T <= self.capacity, f"Data size {T} exceeds capacity {self.capacity}"

        for t in range(T - 1):  # -1 because we need next state
            self.add(
                market_state=market_states[t],
                portfolio_state=portfolio_states[t],
                action=actions[t],
                reward=rewards[t],
                next_market_state=market_states[t + 1],
                next_portfolio_state=portfolio_states[t + 1],
                done=dones[t],
            )

    def __len__(self) -> int:
        return self.size


# ========== Standalone Testing ==========

if __name__ == '__main__':
    # Test CQL-SAC Agent
    print("Testing CQL-SAC Agent...")

    config = CQLSACConfig(
        n_assets=20,
        state_dim=35,
        portfolio_dim=22,
    )

    agent = CQLSACAgent(config, device='cpu')

    # Create dummy batch
    batch_size = 32
    dummy_batch = {
        'market_states': torch.randn(batch_size, 20, 35),
        'portfolio_states': torch.randn(batch_size, 22),
        'actions': torch.randn(batch_size, 20).clamp(-1, 1),
        'rewards': torch.randn(batch_size),
        'next_market_states': torch.randn(batch_size, 20, 35),
        'next_portfolio_states': torch.randn(batch_size, 22),
        'dones': torch.zeros(batch_size),
    }

    # Test update
    losses = agent.update(dummy_batch)
    print(f"Update losses: {losses}")

    # Test action selection
    market_state = np.random.randn(20, 35).astype(np.float32)
    portfolio_state = np.random.randn(22).astype(np.float32)
    action = agent.select_action(market_state, portfolio_state)
    print(f"Action shape: {action.shape}")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")

    print("\nCQL-SAC Agent test passed!")
