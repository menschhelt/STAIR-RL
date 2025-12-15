#!/usr/bin/env python3
"""
Training profiler - Identify bottlenecks in STAIR-RL training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import numpy as np
from contextlib import contextmanager
from collections import defaultdict

# Profiling utilities
class Profiler:
    def __init__(self):
        self.times = defaultdict(list)
        self.counts = defaultdict(int)

    @contextmanager
    def measure(self, name):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        yield
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
        self.times[name].append(elapsed)
        self.counts[name] += 1

    def report(self):
        print("\n" + "=" * 70)
        print("PROFILING REPORT")
        print("=" * 70)

        total = sum(sum(t) for t in self.times.values())

        # Sort by total time
        sorted_items = sorted(
            self.times.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )

        print(f"{'Component':<35} {'Total(s)':<10} {'Avg(ms)':<10} {'%':<8} {'Count'}")
        print("-" * 70)

        for name, times in sorted_items:
            total_time = sum(times)
            avg_time = np.mean(times) * 1000  # ms
            pct = (total_time / total * 100) if total > 0 else 0
            count = self.counts[name]
            print(f"{name:<35} {total_time:<10.3f} {avg_time:<10.2f} {pct:<8.1f} {count}")

        print("-" * 70)
        print(f"{'TOTAL':<35} {total:<10.3f}")
        print("=" * 70)


def profile_cql_sac_step(profiler, agent, batch, device):
    """Profile a single CQL-SAC training step."""

    with profiler.measure("total_step"):
        # Unpack batch
        with profiler.measure("batch_to_device"):
            states = torch.FloatTensor(batch['states']).to(device)
            actions = torch.FloatTensor(batch['actions']).to(device)
            rewards = torch.FloatTensor(batch['rewards']).to(device)
            next_states = torch.FloatTensor(batch['next_states']).to(device)
            dones = torch.FloatTensor(batch['dones']).to(device)

        # Encoder forward
        with profiler.measure("encoder_forward"):
            if hasattr(agent, 'adapter') and agent.adapter is not None:
                # Hierarchical encoder
                market = states[:, :agent.config.n_assets * agent.config.state_dim].reshape(
                    -1, agent.config.n_assets, agent.config.state_dim
                )
                portfolio = states[:, agent.config.n_assets * agent.config.state_dim:]
                z, _ = agent.adapter.encode_state(market, portfolio)
            else:
                z = agent.encoder(states)

        # Critic forward (Q values)
        with profiler.measure("critic_forward"):
            q1, q2 = agent.critic(z, actions)

        # Actor forward
        with profiler.measure("actor_forward"):
            new_actions, log_probs = agent.actor.get_action(z)

        # Target Q computation
        with profiler.measure("target_q"):
            with torch.no_grad():
                if hasattr(agent, 'adapter') and agent.adapter is not None:
                    next_market = next_states[:, :agent.config.n_assets * agent.config.state_dim].reshape(
                        -1, agent.config.n_assets, agent.config.state_dim
                    )
                    next_portfolio = next_states[:, agent.config.n_assets * agent.config.state_dim:]
                    next_z, _ = agent.adapter.encode_state(next_market, next_portfolio)
                else:
                    next_z = agent.encoder(next_states)

                next_actions, next_log_probs = agent.actor.get_action(next_z)
                target_q1, target_q2 = agent.target_critic(next_z, next_actions)

        # Loss computation
        with profiler.measure("loss_computation"):
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * agent.config.gamma * target_q

            critic_loss = torch.nn.functional.mse_loss(q1, target_q) + \
                         torch.nn.functional.mse_loss(q2, target_q)

        # Backward pass
        with profiler.measure("backward_critic"):
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

        with profiler.measure("backward_actor"):
            q1_new, q2_new = agent.critic(z.detach(), new_actions)
            actor_loss = (agent.alpha * log_probs - torch.min(q1_new, q2_new)).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()


def profile_encoder_only(profiler, model, device, n_iters=100):
    """Profile just the encoder (HierarchicalFeatureEncoder)."""

    # Create dummy inputs
    B, T, N = 256, 24, 20

    state_dict = {
        'alphas': torch.randn(B, T, N, 101, device=device),
        'news_embedding': torch.randn(B, T, 768, device=device),
        'social_embedding': torch.randn(B, T, 768, device=device),
        'global_features': torch.randn(B, T, 28, device=device),  # (B, T, 28) not (B, 6)
        'portfolio_state': torch.randn(B, 22, device=device),
    }

    print(f"\nProfiling encoder with B={B}, T={T}, N={N}")

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model.encoder(state_dict)

    # Profile
    for i in range(n_iters):
        with profiler.measure("encoder_total"):
            with torch.no_grad():
                z_pooled, z_unpooled = model.encoder(state_dict)


def profile_full_forward(profiler, model, device, n_iters=50):
    """Profile full forward pass including actor."""

    B, T, N = 256, 24, 20

    state_dict = {
        'alphas': torch.randn(B, T, N, 101, device=device),
        'news_embedding': torch.randn(B, T, 768, device=device),
        'social_embedding': torch.randn(B, T, 768, device=device),
        'global_features': torch.randn(B, T, 28, device=device),  # (B, T, 28) not (B, 6)
        'portfolio_state': torch.randn(B, 22, device=device),
    }

    print(f"\nProfiling full forward with B={B}, T={T}, N={N}")

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model.get_action_and_value(state_dict)

    # Profile components
    for i in range(n_iters):
        with profiler.measure("full_forward"):
            with torch.no_grad():
                # Encoder
                with profiler.measure("  encoder"):
                    z_pooled, z_unpooled = model.encoder(state_dict)

                # Actor
                with profiler.measure("  actor"):
                    weights, log_prob = model.actor.get_action(z_pooled, z_unpooled)

                # Critic
                with profiler.measure("  critic"):
                    value, _ = model.critic(z_pooled)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-iters', type=int, default=50)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
        print(f"CUDA Version: {torch.version.cuda}")

    profiler = Profiler()

    # Import model
    from agents.networks import HierarchicalActorCritic

    print("\nInitializing HierarchicalActorCritic...")
    model = HierarchicalActorCritic(
        n_alphas=101,
        n_assets=20,
        d_alpha=64,
        d_text=64,
        d_temporal=128,
        d_global=32,
        d_portfolio=16,
        n_quantiles=8,
        use_terc=True,  # Test with TERC enabled
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Profile encoder
    print("\n" + "=" * 50)
    print("1. ENCODER ONLY PROFILING")
    print("=" * 50)
    profile_encoder_only(profiler, model, device, n_iters=args.n_iters)

    # Profile full forward
    print("\n" + "=" * 50)
    print("2. FULL FORWARD PROFILING")
    print("=" * 50)
    profile_full_forward(profiler, model, device, n_iters=args.n_iters)

    # Report
    profiler.report()

    # Memory usage
    if torch.cuda.is_available():
        print(f"\nGPU Memory: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")


if __name__ == '__main__':
    main()
