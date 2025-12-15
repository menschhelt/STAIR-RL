#!/usr/bin/env python3
"""
Test script to verify HierarchicalStateBuilder + EmbeddingLoader integration.

Tests:
1. Embedding files exist
2. EmbeddingLoader can load embeddings
3. HierarchicalStateBuilder can build state_dict with embeddings
4. Agents can initialize with embedding paths
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.embedding_loader import EmbeddingLoader
from agents.hierarchical_state_builder import HierarchicalStateBuilder
from agents.cql_sac import CQLSACAgent, CQLSACConfig
from config.settings import DATA_DIR

def test_embedding_files_exist():
    """Test that embedding HDF5 files exist."""
    print("\n" + "=" * 60)
    print("Test 1: Embedding files exist")
    print("=" * 60)

    embedding_dir = Path('/home/work/data/stair-local/embeddings')
    gdelt_path = embedding_dir / 'gdelt_embeddings.h5'
    nostr_path = embedding_dir / 'nostr_embeddings.h5'

    assert embedding_dir.exists(), f"Embedding directory not found: {embedding_dir}"
    assert gdelt_path.exists(), f"GDELT embeddings not found: {gdelt_path}"
    assert nostr_path.exists(), f"Nostr embeddings not found: {nostr_path}"

    # Check file sizes
    gdelt_size = gdelt_path.stat().st_size / 1e6  # MB
    nostr_size = nostr_path.stat().st_size / 1e6  # MB

    print(f"✓ Embedding directory: {embedding_dir}")
    print(f"✓ GDELT embeddings: {gdelt_path} ({gdelt_size:.1f} MB)")
    print(f"✓ Nostr embeddings: {nostr_path} ({nostr_size:.1f} MB)")

    return str(gdelt_path), str(nostr_path)


def test_embedding_loader(gdelt_path, nostr_path):
    """Test that EmbeddingLoader can load embeddings."""
    print("\n" + "=" * 60)
    print("Test 2: EmbeddingLoader initialization")
    print("=" * 60)

    loader = EmbeddingLoader(
        gdelt_path=gdelt_path,
        nostr_path=nostr_path,
        device='cpu',
    )

    print(f"✓ EmbeddingLoader initialized")
    print(f"  GDELT index size: {len(loader.gdelt_index):,}")
    print(f"  Nostr index size: {len(loader.nostr_index):,}")

    # Test loading embeddings
    timestamps = ['2024-01-01T12:00:00+00:00', '2024-01-01T12:05:00+00:00']
    asset_indices = [0, 1, 2]

    print("\n  Testing market-wide embedding methods...")
    gdelt_emb = loader.get_gdelt_embeddings_marketwide(timestamps, asset_indices)
    nostr_emb = loader.get_nostr_embeddings_marketwide(timestamps, asset_indices)
    social_mask = loader.get_social_signal_mask_marketwide(timestamps, asset_indices)

    print(f"  ✓ GDELT market-wide embeddings: {gdelt_emb.shape} (expected: (2, 768))")
    print(f"  ✓ Nostr market-wide embeddings: {nostr_emb.shape} (expected: (2, 768))")
    print(f"  ✓ Social signal mask: {social_mask.shape} (expected: (2, 1))")

    assert gdelt_emb.shape == (2, 768), f"Wrong GDELT shape: {gdelt_emb.shape}"
    assert nostr_emb.shape == (2, 768), f"Wrong Nostr shape: {nostr_emb.shape}"
    assert social_mask.shape == (2, 1), f"Wrong mask shape: {social_mask.shape}"

    loader.close()
    return True


def test_hierarchical_state_builder(gdelt_path, nostr_path):
    """Test HierarchicalStateBuilder can build state_dict."""
    print("\n" + "=" * 60)
    print("Test 3: HierarchicalStateBuilder with embeddings")
    print("=" * 60)

    builder = HierarchicalStateBuilder(
        n_assets=20,
        n_alphas=101,
        temporal_window=20,
        gdelt_embeddings_path=gdelt_path,
        nostr_embeddings_path=nostr_path,
        device='cpu',
    )

    print(f"✓ HierarchicalStateBuilder initialized")
    print(f"  Embedding loader: {'Loaded' if builder.embedding_loader else 'None'}")
    print(f"  Macro loader: {'Loaded' if builder.macro_loader else 'None'}")

    # Create mock state
    batch_size = 2
    n_assets = 20
    state_dim = 36

    market_state = torch.randn(batch_size, n_assets, state_dim)
    portfolio_state = torch.randn(batch_size, 22)
    timestamps = ['2024-01-01T12:00:00+00:00']

    # Build state_dict
    print("\n  Building state_dict...")
    state_dict = builder.build_state_dict(
        market_state=market_state,
        portfolio_state=portfolio_state,
        timestamps=timestamps,
    )

    # Validate state_dict
    print(f"\n  State dict keys: {list(state_dict.keys())}")
    print(f"  Shapes:")
    print(f"    alphas: {state_dict['alphas'].shape} (expected: (2, 20, 20, 101))")
    print(f"    news_embedding: {state_dict['news_embedding'].shape} (expected: (2, 20, 768))")
    print(f"    social_embedding: {state_dict['social_embedding'].shape} (expected: (2, 20, 768))")
    print(f"    has_social_signal: {state_dict['has_social_signal'].shape} (expected: (2, 20, 1))")
    print(f"    ohlcv_seq: {state_dict['ohlcv_seq'].shape} (expected: (2, 20, 20, 288, 5))")
    print(f"    global_features: {state_dict['global_features'].shape}")
    print(f"    portfolio_state: {state_dict['portfolio_state'].shape} (expected: (2, 22))")

    # Validate
    is_valid = builder.validate_state_dict(state_dict)
    print(f"\n  ✓ State dict validation: {'PASSED' if is_valid else 'FAILED'}")

    assert is_valid, "State dict validation failed!"

    return True


def test_agent_initialization(gdelt_path, nostr_path):
    """Test that agents can initialize with embedding paths."""
    print("\n" + "=" * 60)
    print("Test 4: CQLSACAgent initialization with embeddings")
    print("=" * 60)

    config = CQLSACConfig(
        n_assets=20,
        state_dim=36,
        gdelt_embeddings_path=gdelt_path,
        nostr_embeddings_path=nostr_path,
    )

    agent = CQLSACAgent(
        config=config,
        device='cpu',
    )

    print(f"✓ CQLSACAgent initialized")
    print(f"  use_hierarchical: {config.use_hierarchical}")
    print(f"  GDELT path: {config.gdelt_embeddings_path}")
    print(f"  Nostr path: {config.nostr_embeddings_path}")
    print(f"  Adapter: {'Loaded' if hasattr(agent, 'adapter') and agent.adapter else 'None'}")

    if config.use_hierarchical:
        assert agent.adapter is not None, "Adapter should be initialized in hierarchical mode"
        print(f"  ✓ Hierarchical adapter initialized")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("HIERARCHICAL STATE BUILDER + EMBEDDING LOADER INTEGRATION TESTS")
    print("=" * 80)

    try:
        # Test 1: Files exist
        gdelt_path, nostr_path = test_embedding_files_exist()

        # Test 2: EmbeddingLoader
        test_embedding_loader(gdelt_path, nostr_path)

        # Test 3: HierarchicalStateBuilder
        test_hierarchical_state_builder(gdelt_path, nostr_path)

        # Test 4: Agent initialization
        test_agent_initialization(gdelt_path, nostr_path)

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
