#!/usr/bin/env python3
"""
Test Embedding Integration

Verifies that GDELT and Nostr embeddings are correctly loaded and used by
HierarchicalStateBuilder.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from agents.embedding_loader import EmbeddingLoader
from agents.hierarchical_state_builder import HierarchicalStateBuilder
from agents.networks import HierarchicalActorCritic
from agents.hierarchical_adapter import HierarchicalActorCriticAdapter


def test_embedding_loader():
    """Test EmbeddingLoader can load GDELT and Nostr embeddings."""
    print("\n" + "="*60)
    print("TEST 1: EmbeddingLoader")
    print("="*60)

    gdelt_path = '/home/work/data/stair-local/embeddings/gdelt_embeddings.h5'
    nostr_path = '/home/work/data/stair-local/embeddings/nostr_embeddings.h5'

    loader = EmbeddingLoader(
        gdelt_path=gdelt_path,
        nostr_path=nostr_path,
    )

    # Test GDELT loading
    timestamps = ['2021-01-01T00:00:00+00:00', '2021-01-01T00:05:00+00:00']
    assets = [0, 1]  # BTC, ETH

    gdelt_emb = loader.get_gdelt_embeddings(timestamps, assets)
    print(f"✓ GDELT embeddings shape: {gdelt_emb.shape}")
    assert gdelt_emb.shape == (2, 2, 768), f"Expected (2, 2, 768), got {gdelt_emb.shape}"

    # Test Nostr loading
    nostr_emb = loader.get_nostr_embeddings(timestamps, assets)
    print(f"✓ Nostr embeddings shape: {nostr_emb.shape}")
    assert nostr_emb.shape == (2, 2, 768), f"Expected (2, 2, 768), got {nostr_emb.shape}"

    # Test social signal mask
    mask = loader.get_social_signal_mask(timestamps, assets)
    print(f"✓ Social signal mask shape: {mask.shape}")
    assert mask.shape == (2, 2, 1), f"Expected (2, 2, 1), got {mask.shape}"

    # Check that GDELT embeddings are not all zeros (should have some data)
    gdelt_sum = gdelt_emb.abs().sum().item()
    print(f"✓ GDELT embedding sum (should be > 0): {gdelt_sum:.2f}")
    assert gdelt_sum > 0, "GDELT embeddings are all zeros!"

    # Nostr might be sparse, so we don't assert it's non-zero
    nostr_sum = nostr_emb.abs().sum().item()
    print(f"✓ Nostr embedding sum: {nostr_sum:.2f} (may be sparse)")

    loader.close()
    print("✓ TEST 1 PASSED\n")


def test_state_builder_with_embeddings():
    """Test StateBuilder loads embeddings correctly."""
    print("="*60)
    print("TEST 2: StateBuilder with Embeddings")
    print("="*60)

    gdelt_path = '/home/work/data/stair-local/embeddings/gdelt_embeddings.h5'
    nostr_path = '/home/work/data/stair-local/embeddings/nostr_embeddings.h5'

    builder = HierarchicalStateBuilder(
        n_assets=10,
        n_alphas=292,
        temporal_window=20,
        gdelt_embeddings_path=gdelt_path,
        nostr_embeddings_path=nostr_path,
    )

    market_state = torch.randn(4, 10, 36)  # (B=4, N=10, state_dim=36)
    portfolio_state = torch.randn(4, 22)   # (B=4, portfolio_dim=22)
    timestamps = ['2021-01-01T12:00:00+00:00']

    state_dict = builder.build_state_dict(
        market_state, portfolio_state, timestamps=timestamps
    )

    # Verify shapes
    print(f"✓ Alphas shape: {state_dict['alphas'].shape}")
    assert state_dict['alphas'].shape == (4, 20, 10, 292)

    print(f"✓ News embedding shape: {state_dict['news_embedding'].shape}")
    assert state_dict['news_embedding'].shape == (4, 20, 10, 768)

    print(f"✓ Social embedding shape: {state_dict['social_embedding'].shape}")
    assert state_dict['social_embedding'].shape == (4, 20, 10, 768)

    print(f"✓ Has social signal shape: {state_dict['has_social_signal'].shape}")
    assert state_dict['has_social_signal'].shape == (4, 20, 10, 1)

    print(f"✓ Global features shape: {state_dict['global_features'].shape}")
    assert state_dict['global_features'].shape == (4, 20, 6)

    print(f"✓ Portfolio state shape: {state_dict['portfolio_state'].shape}")
    assert state_dict['portfolio_state'].shape == (4, 22)

    # Verify embeddings are loaded (not all zeros)
    news_sum = state_dict['news_embedding'].abs().sum().item()
    print(f"✓ News embedding sum (should be > 0): {news_sum:.2f}")
    assert news_sum > 0, "News embeddings are all zeros!"

    print("✓ TEST 2 PASSED\n")


def test_hierarchical_adapter_with_embeddings():
    """Test HierarchicalActorCriticAdapter with embeddings."""
    print("="*60)
    print("TEST 3: HierarchicalActorCriticAdapter with Embeddings")
    print("="*60)

    gdelt_path = '/home/work/data/stair-local/embeddings/gdelt_embeddings.h5'
    nostr_path = '/home/work/data/stair-local/embeddings/nostr_embeddings.h5'

    # Create HierarchicalActorCritic
    model = HierarchicalActorCritic(
        n_alphas=292,
        n_assets=10,
        d_alpha=64,
        d_text=64,
        d_temporal=128,
        d_global=6,
        d_portfolio=22,
        n_quantiles=10,
    )

    # Create adapter with embedding paths
    adapter = HierarchicalActorCriticAdapter(
        model,
        gdelt_embeddings_path=gdelt_path,
        nostr_embeddings_path=nostr_path,
    )

    # Test encoding with timestamps
    market_state = torch.randn(4, 10, 36)  # (B=4, N=10, state_dim=36)
    portfolio_state = torch.randn(4, 22)   # (B=4, portfolio_dim=22)
    timestamps = ['2021-01-01T12:00:00+00:00']

    z_pooled, z_unpooled = adapter.encode_state(
        market_state, portfolio_state, timestamps=timestamps
    )

    # Expected dim: d_temporal (128) + d_global (6) + d_portfolio (22) = 156
    print(f"✓ z_pooled shape: {z_pooled.shape}")
    assert z_pooled.shape == (4, 156)

    print(f"✓ z_unpooled shape: {z_unpooled.shape}")
    assert z_unpooled.shape == (4, 10, 156)

    # Test action selection
    weights, trade_prob = adapter.get_action(z_pooled, z_unpooled)
    print(f"✓ Weights shape: {weights.shape}")
    assert weights.shape == (4, 10)

    print(f"✓ Trade prob shape: {trade_prob.shape}")
    assert trade_prob.shape == (4, 1)

    # Test value estimation
    value = adapter.get_value(z_pooled)
    print(f"✓ Value shape: {value.shape}")
    assert value.shape == (4, 1)

    print("✓ TEST 3 PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("EMBEDDING INTEGRATION TESTS")
    print("="*60)

    try:
        test_embedding_loader()
        test_state_builder_with_embeddings()
        test_hierarchical_adapter_with_embeddings()

        print("="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nEmbedding integration is working correctly!")
        print("- GDELT news embeddings: Loaded ✓")
        print("- Nostr social embeddings: Loaded ✓")
        print("- HierarchicalStateBuilder: Using real embeddings ✓")
        print("- HierarchicalActorCritic: Receiving multi-modal input ✓")
        return 0

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED!")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
