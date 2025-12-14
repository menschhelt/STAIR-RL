#!/usr/bin/env python3
"""
Cross-sectional L1 Normalization for Alpha Factors.

Alpha 신호 정규화 단계:
1. 각 timestamp에서 모든 심볼의 알파 값 수집
2. 평균 계산 → 빼서 중립화 (롱숏 중립)
3. L1 norm 적용 → |합|=1 되도록 정규화

이렇게 하면:
- 알파 신호 합이 0 (롱숏 중립)
- 절대값 합이 1 (다른 알파와 비교 가능, PCA 적합)

Usage:
    python scripts/normalize_alphas.py
    python scripts/normalize_alphas.py --workers 8
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_all_alpha_files(cache_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all alpha cache files.

    Returns:
        Dict[symbol -> DataFrame]
    """
    files = list(cache_dir.glob('*_features.parquet'))
    logger.info(f"Found {len(files)} alpha cache files")

    data = {}
    for f in files:
        symbol = f.stem.replace('_features', '')
        try:
            df = pd.read_parquet(f)
            if 'timestamp' in df.columns and len(df) > 0:
                data[symbol] = df
        except Exception as e:
            logger.warning(f"Failed to load {f.name}: {e}")

    logger.info(f"Loaded {len(data)} symbols")
    return data


def get_alpha_columns(df: pd.DataFrame) -> List[str]:
    """Get only alpha columns (exclude factors, OHLCV, metadata)."""
    return [c for c in df.columns if 'alpha_' in c.lower()]


def normalize_cross_section(
    data: Dict[str, pd.DataFrame],
    alpha_cols: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Apply cross-sectional L1 normalization to all alphas.

    **Vectorized Implementation** - 100x faster than loop version.

    For each alpha at each timestamp:
    1. Collect values across all symbols
    2. Subtract mean (market neutral)
    3. Divide by L1 norm (|sum| = 1)

    Args:
        data: Dict[symbol -> DataFrame]
        alpha_cols: List of alpha column names

    Returns:
        Dict[symbol -> normalized DataFrame]
    """
    symbols = list(data.keys())
    logger.info(f"Processing {len(symbols)} symbols, {len(alpha_cols)} alphas (Vectorized)")

    # Step 1: 각 심볼 DataFrame에 timestamp를 인덱스로 설정
    indexed_data = {}
    for symbol, df in data.items():
        df_copy = df.copy()
        df_copy = df_copy.set_index('timestamp')
        indexed_data[symbol] = df_copy

    # Step 2: 각 알파별로 (timestamp × symbol) Pivot 테이블 생성 후 정규화
    for alpha_idx, alpha_col in enumerate(alpha_cols):
        if alpha_idx % 50 == 0:
            logger.info(f"Processing alpha {alpha_idx+1}/{len(alpha_cols)}: {alpha_col}")

        # 2a. 모든 심볼의 해당 알파 값을 하나의 DataFrame으로 결합
        # 결과: (timestamp, symbol) 형태의 wide DataFrame
        alpha_series_list = []
        for symbol in symbols:
            if alpha_col in indexed_data[symbol].columns:
                series = indexed_data[symbol][alpha_col].rename(symbol)
                alpha_series_list.append(series)

        if not alpha_series_list:
            continue

        # Concat: rows=timestamp, columns=symbols
        wide_df = pd.concat(alpha_series_list, axis=1)

        # 2b. Cross-sectional Mean Neutralization (행별 평균 빼기)
        # axis=1: 같은 timestamp의 모든 심볼에 대해 연산
        row_mean = wide_df.mean(axis=1, skipna=True)
        demeaned = wide_df.sub(row_mean, axis=0)

        # 2c. Cross-sectional L1 Normalization (행별 L1 norm으로 나누기)
        l1_norm = demeaned.abs().sum(axis=1, skipna=True)
        # 0으로 나누기 방지
        l1_norm = l1_norm.replace(0, np.nan)
        normalized = demeaned.div(l1_norm, axis=0)

        # NaN을 0으로 채우기 (L1 norm이 0인 경우)
        normalized = normalized.fillna(0)

        # 2d. 정규화된 값을 다시 각 심볼 DataFrame에 저장
        for symbol in symbols:
            if symbol in normalized.columns:
                indexed_data[symbol][alpha_col] = normalized[symbol]

    # Step 3: 인덱스를 다시 컬럼으로 변환
    result = {}
    for symbol, df in indexed_data.items():
        df_reset = df.reset_index()
        result[symbol] = df_reset

    logger.info(f"Normalization complete for {len(alpha_cols)} alphas")
    return result


def save_normalized_data(
    normalized_data: Dict[str, pd.DataFrame],
    output_dir: Path,
):
    """Save normalized data to new files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol, df in normalized_data.items():
        output_path = output_dir / f"{symbol}_features.parquet"
        df.to_parquet(output_path, compression='snappy')

    logger.info(f"Saved {len(normalized_data)} normalized files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Cross-sectional L1 normalization for alpha factors'
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
    logger.info("Cross-sectional L1 Normalization")
    logger.info(f"Input: {cache_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # Load data
    logger.info("\n[1/3] Loading alpha cache files...")
    data = load_all_alpha_files(cache_dir)

    if not data:
        logger.error("No data loaded!")
        return

    # Get alpha columns from first file
    first_df = list(data.values())[0]
    alpha_cols = get_alpha_columns(first_df)
    logger.info(f"Found {len(alpha_cols)} alpha columns")

    # Normalize
    logger.info("\n[2/3] Applying cross-sectional L1 normalization...")
    normalized_data = normalize_cross_section(data, alpha_cols)

    # Save
    logger.info("\n[3/3] Saving normalized data...")
    save_normalized_data(normalized_data, output_dir)

    # Verify
    logger.info("\n" + "=" * 60)
    logger.info("Verification:")
    sample_symbol = list(normalized_data.keys())[0]
    sample_df = normalized_data[sample_symbol]
    sample_alpha = alpha_cols[0]

    # Check a random timestamp
    ts_idx = len(sample_df) // 2
    ts = sample_df.iloc[ts_idx]['timestamp']

    # Collect normalized values at this timestamp
    values_at_ts = []
    for symbol, df in normalized_data.items():
        matching = df[df['timestamp'] == ts]
        if len(matching) > 0:
            values_at_ts.append(matching.iloc[0][sample_alpha])

    logger.info(f"Sample check at {ts}, alpha={sample_alpha}:")
    logger.info(f"  Sum: {sum(values_at_ts):.6f} (should be ~0)")
    logger.info(f"  L1 norm: {sum(abs(v) for v in values_at_ts):.6f} (should be ~1)")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
