#!/usr/bin/env python3
"""
Cross-sectional L1 Normalization for Alpha Factors.

Alpha 신호 정규화 단계:
1. 각 timestamp에서 **당시 존재하던 심볼만** 수집 (historical NaN 유지)
2. 평균 계산 → 빼서 중립화 (롱숏 중립)
3. L1 norm 적용 → |합|=1 되도록 정규화

이렇게 하면:
- 알파 신호 합이 0 (롱숏 중립)
- 절대값 합이 1 (다른 알파와 비교 가능, Cross-Attention 준비)
- Historical NaN 보존 (상장 전 심볼은 NaN 유지)

Input format: alpha_101_{idx}.parquet (per-alpha files)
Output format: alpha_101_{idx}.parquet (same, but normalized)

Usage:
    python scripts/normalize_alphas.py
    python scripts/normalize_alphas.py --in-place  # overwrite original
"""

import argparse
import logging
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_alpha_file(
    input_path: Path,
    output_path: Path,
) -> dict:
    """
    Normalize a single alpha file with cross-sectional L1 normalization.

    Args:
        input_path: Path to input alpha parquet file
        output_path: Path to save normalized alpha

    Returns:
        Stats dict with normalization results
    """
    # Load alpha file
    # Format: index=timestamp, columns=symbols, values=alpha values
    df = pd.read_parquet(input_path)

    original_shape = df.shape

    # Store original NaN mask (historical data - unlisted symbols at each time)
    original_nan_mask = df.isna()

    # Step 1: Cross-sectional mean neutralization (per timestamp)
    # Only use non-NaN values for mean calculation
    row_mean = df.mean(axis=1, skipna=True)
    demeaned = df.sub(row_mean, axis=0)

    # Step 2: Cross-sectional L1 normalization (per timestamp)
    l1_norm = demeaned.abs().sum(axis=1, skipna=True)
    # Prevent division by zero
    l1_norm = l1_norm.replace(0, np.nan)
    normalized = demeaned.div(l1_norm, axis=0)

    # Division-by-zero NaN → 0 (when L1 norm is 0, all values were 0)
    normalized = normalized.fillna(0)

    # Step 3: Restore historical NaN mask (CRITICAL!)
    # Symbols that didn't exist at a timestamp should remain NaN
    normalized[original_nan_mask] = np.nan

    # Save normalized alpha
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output_path, compression='snappy')

    # Compute stats for verification
    # Pick a random timestamp to check sum and L1 norm
    mid_idx = len(normalized) // 2
    sample_row = normalized.iloc[mid_idx]
    valid_values = sample_row.dropna()

    return {
        'shape': original_shape,
        'nan_count': original_nan_mask.sum().sum(),
        'sample_sum': valid_values.sum() if len(valid_values) > 0 else 0,
        'sample_l1': valid_values.abs().sum() if len(valid_values) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Cross-sectional L1 normalization for alpha factors (per-alpha format)'
    )
    parser.add_argument(
        '--cache-dir', type=str,
        default=str(DATA_DIR / 'features' / 'alpha_cache'),
        help='Input alpha cache directory'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default=str(DATA_DIR / 'features' / 'alpha_cache_normalized'),
        help='Output directory for normalized alphas'
    )
    parser.add_argument(
        '--in-place', action='store_true',
        help='Overwrite original files (default: save to new directory)'
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir) if not args.in_place else cache_dir

    logger.info("=" * 60)
    logger.info("Cross-sectional L1 Normalization (Per-Alpha Format)")
    logger.info(f"Input: {cache_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # Find all alpha files
    alpha_files = sorted(cache_dir.glob('alpha_101_*.parquet'))
    logger.info(f"Found {len(alpha_files)} alpha files")

    if not alpha_files:
        logger.error("No alpha files found!")
        return

    # Process each alpha file
    logger.info("\n[1/2] Normalizing alpha files...")
    all_stats = []

    for i, input_path in enumerate(alpha_files):
        output_path = output_dir / input_path.name

        try:
            stats = normalize_alpha_file(input_path, output_path)
            all_stats.append(stats)

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(
                    f"[{i+1}/{len(alpha_files)}] {input_path.name}: "
                    f"sum={stats['sample_sum']:.6f}, L1={stats['sample_l1']:.4f}"
                )
        except Exception as e:
            logger.error(f"Failed to normalize {input_path.name}: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("[2/2] Verification Summary")
    logger.info("=" * 60)

    if all_stats:
        avg_sum = np.mean([s['sample_sum'] for s in all_stats])
        avg_l1 = np.mean([s['sample_l1'] for s in all_stats])
        total_nans = sum(s['nan_count'] for s in all_stats)

        logger.info(f"Processed: {len(all_stats)} alpha files")
        logger.info(f"Average sample sum: {avg_sum:.6f} (should be ~0)")
        logger.info(f"Average sample L1 norm: {avg_l1:.4f} (should be ~1)")
        logger.info(f"Total historical NaNs preserved: {total_nans:,}")
        logger.info(f"Output directory: {output_dir}")

    logger.info("=" * 60)
    logger.info("Done! Normalized alphas saved to:")
    logger.info(f"  {output_dir}")
    logger.info("")
    logger.info("To use normalized alphas in training:")
    logger.info("  Set use_normalized_alphas=True in hierarchical_adapter.py")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
