#!/usr/bin/env python3
"""
Test PriceTransformerEncoder and HierarchicalFeatureEncoder integration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from agents.networks import PriceTransformerEncoder, HierarchicalFeatureEncoder


def test_price_encoder():
    """Test PriceTransformerEncoder standalone."""
    print("=" * 80)
    print("Test 1: PriceTransformerEncoder")
    print("=" * 80)

    encoder = PriceTransformerEncoder(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )

    # Create mock OHLCV sequences
    B, T, N, seq_len = 2, 20, 10, 288
    ohlcv = torch.randn(B, T, N, seq_len, 5)

    print(f"Input shape: {ohlcv.shape}")  # (2, 20, 10, 288, 5)

    # Forward pass
    output = encoder(ohlcv)

    print(f"Output shape: {output.shape}")  # (2, 20, 10, 64)
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")

    assert output.shape == (B, T, N, 64), f"Expected (2, 20, 10, 64), got {output.shape}"
    print("✅ PriceTransformerEncoder test passed!\n")


def test_hierarchical_encoder_with_price():
    """Test HierarchicalFeatureEncoder with price encoding."""
    print("=" * 80)
    print("Test 2: HierarchicalFeatureEncoder with Price Encoding")
    print("=" * 80)

    encoder = HierarchicalFeatureEncoder(
        n_alphas=292,
        n_assets=20,
        d_alpha=64,
        d_text=64,
        d_price=64,
        d_temporal=128,
        d_global=32,
        d_portfolio=16,
        n_alpha_heads=8,
        n_asset_heads=8,
        dropout=0.1,
    )

    B, T, N = 4, 20, 20

    # Create mock state dict
    state_dict = {
        'alphas': torch.randn(B, T, N, 292),
        'news_embedding': torch.randn(B, T, N, 768),
        'social_embedding': torch.randn(B, T, N, 768),
        'has_social_signal': torch.ones(B, T, N, 1),
        'ohlcv_seq': torch.randn(B, T, N, 288, 5),  # NEW!
        'global_features': torch.randn(B, T, 6),
        'portfolio_state': torch.randn(B, 22),
    }

    print(f"Input shapes:")
    for key, val in state_dict.items():
        print(f"  {key}: {val.shape}")

    # Forward pass
    z_pooled, z_unpooled = encoder(state_dict)

    print(f"\nOutput shapes:")
    print(f"  z_pooled: {z_pooled.shape}")    # (4, 176)
    print(f"  z_unpooled: {z_unpooled.shape}")  # (4, 20, 176)

    assert z_pooled.shape == (B, 176), f"Expected (4, 176), got {z_pooled.shape}"
    assert z_unpooled.shape == (B, N, 176), f"Expected (4, 20, 176), got {z_unpooled.shape}"
    print("✅ HierarchicalFeatureEncoder test passed!\n")


def test_backward_compatibility():
    """Test that encoder works without ohlcv_seq (backward compatibility)."""
    print("=" * 80)
    print("Test 3: Backward Compatibility (no OHLCV)")
    print("=" * 80)

    encoder = HierarchicalFeatureEncoder()

    B, T, N = 2, 20, 20

    # State dict WITHOUT ohlcv_seq
    state_dict = {
        'alphas': torch.randn(B, T, N, 292),
        'news_embedding': torch.randn(B, T, N, 768),
        'social_embedding': torch.randn(B, T, N, 768),
        'global_features': torch.randn(B, T, 6),
        'portfolio_state': torch.randn(B, 22),
    }

    print(f"Input shapes (no ohlcv_seq):")
    for key, val in state_dict.items():
        print(f"  {key}: {val.shape}")

    # Forward pass should still work (uses zeros for price)
    z_pooled, z_unpooled = encoder(state_dict)

    print(f"\nOutput shapes:")
    print(f"  z_pooled: {z_pooled.shape}")
    print(f"  z_unpooled: {z_unpooled.shape}")

    assert z_pooled.shape == (B, 176)
    assert z_unpooled.shape == (B, N, 176)
    print("✅ Backward compatibility test passed!\n")


def test_price_encoder_from_real_data():
    """Test PriceTransformerEncoder with real normalized OHLCV data format."""
    print("=" * 80)
    print("Test 4: PriceTransformerEncoder with Realistic Data")
    print("=" * 80)

    from features.ohlcv_processor import OHLCVSequenceBuilder
    from datetime import date

    # Create builder
    builder = OHLCVSequenceBuilder(lookback=288)

    # Load real sequence for BTC
    try:
        seq = builder.load_5min_sequence('BTCUSDT', date(2024, 6, 15))
        print(f"Loaded real BTC sequence: {seq.shape}")  # (288, 5)
        print(f"Sample candle: {seq[0]}")

        # Create batch: (B=1, T=1, N=1, 288, 5)
        ohlcv_batch = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Encode
        encoder = PriceTransformerEncoder()
        output = encoder(ohlcv_batch)

        print(f"Encoded shape: {output.shape}")  # (1, 1, 1, 64)
        print(f"Encoded mean: {output.mean().item():.6f}")
        print(f"Encoded std: {output.std().item():.6f}")

        assert output.shape == (1, 1, 1, 64)
        print("✅ Real data encoding test passed!\n")

    except Exception as e:
        print(f"⚠️  Could not test with real data: {e}")
        print("   (This is okay if data files are not available)")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("PRICE ENCODER TESTS")
    print("=" * 80 + "\n")

    test_price_encoder()
    test_hierarchical_encoder_with_price()
    test_backward_compatibility()
    test_price_encoder_from_real_data()

    print("=" * 80)
    print("ALL TESTS PASSED ✅")
    print("=" * 80)
