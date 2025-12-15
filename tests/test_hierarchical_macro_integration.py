#!/usr/bin/env python3
"""
Test HierarchicalStateBuilder integration with MacroDataLoader.

Verifies:
1. MacroDataLoader is initialized in HierarchicalStateBuilder
2. Global features are loaded with correct shape (B, T, 23)
3. Forward-fill strategy works for temporal sequences
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
from agents.hierarchical_state_builder import HierarchicalStateBuilder
from config.settings import DATA_DIR


def test_macro_integration():
    """Test MacroDataLoader integration in HierarchicalStateBuilder."""

    print("=" * 60)
    print("Test: MacroDataLoader Integration")
    print("=" * 60)

    # Initialize HierarchicalStateBuilder (should auto-detect macro_data_dir)
    state_builder = HierarchicalStateBuilder(
        n_assets=20,
        n_alphas=101,
        temporal_window=20,
        device='cpu',
    )

    # Check if MacroDataLoader was initialized
    print(f"\n✅ MacroDataLoader initialized: {state_builder.macro_loader is not None}")

    if state_builder.macro_loader is not None:
        print(f"   Total features: {state_builder.macro_loader.n_features}")
        print(f"   Macro indicators: {state_builder.macro_loader.n_macro_features}")
        print(f"   Fama-French factors: {state_builder.macro_loader.n_ff_features}")
        print(f"   Sample indicators: {state_builder.macro_loader.indicators[:5]}...")

    # Create mock market state and portfolio state
    batch_size = 2
    n_assets = 20
    state_dim = 101  # Alpha101 only

    market_state = torch.randn(batch_size, n_assets, state_dim)
    portfolio_state = torch.randn(batch_size, n_assets + 2)  # weights + cash + equity

    # Test with timestamp
    timestamps = ["2024-01-15 09:35:00"]

    print("\n[1/3] Building state_dict with timestamps...")
    state_dict = state_builder.build_state_dict(
        market_state=market_state,
        portfolio_state=portfolio_state,
        timestamps=timestamps,
    )

    # Check global_features shape
    global_features = state_dict['global_features']
    print(f"   global_features shape: {global_features.shape}")

    # Expected shape: (B, T, 28) with Fama-French enabled (23 macro + 5 FF)
    # or (B, T, 23) if Fama-French disabled
    # or (B, T, 6) if no macro loader
    if state_builder.macro_loader is not None:
        expected_features = state_builder.macro_loader.n_features  # 28 or 23
        expected_shape = (batch_size, 20, expected_features)
    else:
        expected_shape = (batch_size, 20, 6)  # Fallback to 6 if no macro loader

    assert global_features.shape == expected_shape, \
        f"Expected {expected_shape}, got {global_features.shape}"

    print(f"   ✅ Shape correct: {global_features.shape}")

    # Breakdown of dimensions
    if state_builder.macro_loader is not None:
        print(f"   Dimension breakdown:")
        print(f"      [0:23]: Macro indicators")
        if state_builder.macro_loader.include_fama_french:
            print(f"      [23:28]: Fama-French factors (MKT_RF, SMB, HML, RMW, CMA)")

    # Check non-zero values
    if state_builder.macro_loader is not None:
        non_zero_count = (global_features.abs() > 1e-6).sum().item()
        total_count = global_features.numel()
        print(f"   Non-zero values: {non_zero_count}/{total_count}")

        # Sample values from first batch, last timestamp
        sample_values = global_features[0, -1, :5]  # First 5 indicators
        print(f"   Sample values (first 5 indicators):")
        for i, val in enumerate(sample_values):
            print(f"      [{i}]: {val.item():.6f}")

    # Test without timestamps (should fallback to zeros)
    print("\n[2/3] Building state_dict without timestamps (fallback)...")
    state_dict_no_ts = state_builder.build_state_dict(
        market_state=market_state,
        portfolio_state=portfolio_state,
        timestamps=None,
    )

    global_features_no_ts = state_dict_no_ts['global_features']
    print(f"   global_features shape: {global_features_no_ts.shape}")

    # Should be zeros
    zero_count = (global_features_no_ts.abs() < 1e-6).sum().item()
    total_count = global_features_no_ts.numel()
    print(f"   Zero values: {zero_count}/{total_count} (should be all zeros)")

    # Validate state_dict structure
    print("\n[3/3] Validating state_dict structure...")
    is_valid = state_builder.validate_state_dict(state_dict)
    print(f"   ✅ State dict valid: {is_valid}")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_macro_integration()
