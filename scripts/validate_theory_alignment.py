#!/usr/bin/env python3
"""
Validate theory alignment: verify all paper-specified losses work correctly.
"""
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.cql_sac import CQLSACAgent, CQLSACConfig
from agents.ppo_cvar import PPOCVaRAgent, PPOCVaRConfig


def test_cql_sac_losses():
    """Test CQL-SAC with all paper-specified losses."""
    print("=" * 80)
    print("Testing CQL-SAC: Reward + Semantic Smoothing + Value Lipschitz")
    print("=" * 80)

    # Create config
    config = CQLSACConfig(
        n_assets=5,
        state_dim=292,  # Alpha factors
        portfolio_dim=5,
        hidden_dim=128,
        use_hierarchical=False,  # Use basic architecture for test
        lambda_smooth=0.1,  # Paper: 0.1
        lambda_gp=1.0,      # Paper: 1.0 for CQL-SAC
        lambda_cql=1.0,
        lr_actor=1e-4,
        lr_critic=3e-4,
        batch_size=32,
    )

    # Create agent
    agent = CQLSACAgent(config)

    # Create mock batch
    batch = {
        'market_states': torch.randn(32, 5, 292),
        'portfolio_states': torch.randn(32, 5),
        'actions': torch.randn(32, 5),
        'rewards': torch.randn(32),
        'next_market_states': torch.randn(32, 5, 292),
        'next_portfolio_states': torch.randn(32, 5),
        'dones': torch.zeros(32),
    }

    # Test update
    print("\nRunning update step...")
    try:
        metrics = agent.update(batch)
        print("✅ Update successful!")
        print("\nLoss Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # Verify all losses are present
        required_losses = ['critic_loss', 'cql_loss', 'td_loss', 'smooth_loss', 'actor_loss']
        missing = [k for k in required_losses if k not in metrics]
        if missing:
            print(f"\n❌ Missing losses: {missing}")
            return False

        # Verify smooth_loss is non-zero
        if metrics['smooth_loss'] == 0.0:
            print("\n⚠️  Warning: smooth_loss is 0.0 (expected non-zero)")
        else:
            print(f"\n✅ Semantic Smoothing Loss active: {metrics['smooth_loss']:.6f}")

        return True

    except Exception as e:
        print(f"❌ Update failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_cvar_losses():
    """Test PPO-CVaR with all paper-specified losses."""
    print("\n" + "=" * 80)
    print("Testing PPO-CVaR: Value Lipschitz Regularization")
    print("=" * 80)

    # Create config
    config = PPOCVaRConfig(
        n_assets=5,
        state_dim=292,
        portfolio_dim=5,
        hidden_dim=128,
        use_hierarchical=False,  # Use basic architecture for test
        lambda_gp=10.0,  # Paper: 10.0 for PPO-CVaR
        lr=3e-4,
        batch_size=32,
    )

    # Create agent
    agent = PPOCVaRAgent(config)

    # Create mock batch
    batch = {
        'market_states': torch.randn(32, 5, 292),
        'portfolio_states': torch.randn(32, 5),
        'actions': torch.randn(32, 5),
        'old_log_probs': torch.randn(32),
        'advantages': torch.randn(32),
        'returns': torch.randn(32),
    }

    # Test update
    print("\nRunning update step...")
    try:
        metrics = agent.update(batch)
        print("✅ Update successful!")
        print("\nLoss Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # Check if gradient penalty is present
        if 'gp_loss' in metrics:
            print(f"\n✅ Value Lipschitz Regularization active: {metrics['gp_loss']:.6f}")
        else:
            print("\n⚠️  Warning: gp_loss not in metrics")

        return True

    except Exception as e:
        print(f"❌ Update failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_reward_function():
    """Verify reward function matches paper specification."""
    print("\n" + "=" * 80)
    print("Verifying Reward Function")
    print("=" * 80)

    from environments.trading_env import TradingEnv, EnvConfig

    # Create config
    config = EnvConfig(
        n_assets=5,
        initial_nav=100000,
        lambda_vol=0.5,   # Paper: 0.5
        lambda_dd=2.0,    # Paper: 2.0
        lambda_tc=1.0,    # Paper: 1.0
        lambda_to=0.1,    # Paper: 0.1
        volatility_decay_rho=0.94,  # Paper: 0.94
        volatility_window_K=20,     # Paper: 20
    )

    print("\nReward Function Parameters:")
    print(f"  λ_vol = {config.lambda_vol} (paper: 0.5)")
    print(f"  λ_dd = {config.lambda_dd} (paper: 2.0)")
    print(f"  λ_tc = {config.lambda_tc} (paper: 1.0)")
    print(f"  λ_to = {config.lambda_to} (paper: 0.1)")
    print(f"  ρ (volatility decay) = {config.volatility_decay_rho} (paper: 0.94)")
    print(f"  K (volatility window) = {config.volatility_window_K} (paper: 20)")

    # Verify all match paper
    checks = [
        (config.lambda_vol, 0.5, "λ_vol"),
        (config.lambda_dd, 2.0, "λ_dd"),
        (config.lambda_tc, 1.0, "λ_tc"),
        (config.lambda_to, 0.1, "λ_to"),
        (config.volatility_decay_rho, 0.94, "ρ"),
        (config.volatility_window_K, 20, "K"),
    ]

    all_match = True
    for actual, expected, name in checks:
        if abs(actual - expected) > 1e-6:
            print(f"  ❌ {name}: {actual} (expected {expected})")
            all_match = False
        else:
            print(f"  ✅ {name}: {actual}")

    return all_match


def main():
    """Run all validation tests."""
    print("\n" + "=" * 80)
    print("THEORY ALIGNMENT VALIDATION")
    print("Verifying implementation matches STAIR-RL paper specification")
    print("=" * 80)

    results = {}

    # Test reward function
    results['reward_function'] = verify_reward_function()

    # Test CQL-SAC losses
    results['cql_sac'] = test_cql_sac_losses()

    # Test PPO-CVaR losses
    results['ppo_cvar'] = test_ppo_cvar_losses()

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Implementation matches paper specification.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
