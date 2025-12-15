#!/usr/bin/env python3
"""
BERT Embedding Pre-computation Script for STAIR-RL

Pre-computes text embeddings offline before training:
- GDELT news → FinBERT embeddings
- Nostr social → CryptoBERT embeddings

Usage:
    # GDELT embeddings (GPU 0)
    python scripts/precompute_embeddings.py \
        --source gdelt \
        --start 2021-01-01 --end 2024-12-31 \
        --device cuda:0 \
        --output /home/work/data/stair-local/embeddings/gdelt_embeddings.h5

    # Nostr embeddings (GPU 1, can run in parallel)
    python scripts/precompute_embeddings.py \
        --source nostr \
        --start 2021-01-01 --end 2024-12-31 \
        --device cuda:1 \
        --output /home/work/data/stair-local/embeddings/nostr_embeddings.h5

Output:
    HDF5 file with:
    - embeddings: (N, 768) float16
    - keys: list of "timestamp_assetIdx" strings
    - metadata: model name, creation date, etc.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from text_processing.finbert_processor import FinBERTProcessor
from text_processing.cryptobert_processor import CryptoBERTProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Asset symbol mapping for keyword detection
ASSET_KEYWORDS = {
    0: ['btc', 'bitcoin'],
    1: ['eth', 'ethereum'],
    2: ['sol', 'solana'],
    3: ['ada', 'cardano'],
    4: ['avax', 'avalanche'],
    5: ['dot', 'polkadot'],
    6: ['atom', 'cosmos'],
    7: ['uni', 'uniswap'],
    8: ['aave'],
    9: ['mkr', 'maker'],
    10: ['comp', 'compound'],
    11: ['arb', 'arbitrum'],
    12: ['op', 'optimism'],
    13: ['matic', 'polygon'],
    14: ['usdt', 'tether'],
    15: ['usdc'],
    16: ['dai'],
    17: ['link', 'chainlink'],
    18: ['ltc', 'litecoin'],
    19: ['xrp', 'ripple'],
}

# Reverse mapping for quick lookup
KEYWORD_TO_ASSET = {}
for asset_idx, keywords in ASSET_KEYWORDS.items():
    for kw in keywords:
        KEYWORD_TO_ASSET[kw] = asset_idx


def detect_assets_in_text(text: str) -> List[int]:
    """
    Detect which assets are mentioned in text.

    Args:
        text: Input text

    Returns:
        List of asset indices mentioned
    """
    if not text:
        return []

    text_lower = text.lower()
    mentioned = set()

    for keyword, asset_idx in KEYWORD_TO_ASSET.items():
        # Use word boundary matching
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text_lower):
            mentioned.add(asset_idx)

    return list(mentioned)


def round_to_5min(ts: pd.Timestamp) -> str:
    """
    Round timestamp to nearest 5-minute interval.

    Args:
        ts: Input timestamp

    Returns:
        ISO format string of rounded timestamp
    """
    # Floor to 5-minute
    minute = (ts.minute // 5) * 5
    rounded = ts.replace(minute=minute, second=0, microsecond=0)
    return rounded.isoformat()


def load_gdelt_data(
    data_dir: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Load GDELT data for date range.

    Args:
        data_dir: Directory containing GDELT parquet files
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with columns: published_at, title, content
    """
    logger.info(f"Loading GDELT data from {start_date} to {end_date}")

    start = pd.Timestamp(start_date, tz='UTC')
    end = pd.Timestamp(end_date, tz='UTC')

    # Find relevant files
    all_files = sorted(data_dir.glob('gdelt_*.parquet'))
    dfs = []

    for fpath in tqdm(all_files, desc="Loading GDELT files"):
        try:
            df = pd.read_parquet(fpath)

            # Filter by date
            if 'published_at' in df.columns:
                df = df[(df['published_at'] >= start) & (df['published_at'] <= end)]

                if len(df) > 0:
                    dfs.append(df[['published_at', 'title', 'content']])

        except Exception as e:
            logger.warning(f"Error loading {fpath}: {e}")

    if not dfs:
        logger.warning("No GDELT data found")
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(result):,} GDELT articles")

    return result


def load_nostr_data(
    data_dir: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Load Nostr data for date range.

    Args:
        data_dir: Directory containing Nostr parquet files
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with columns: created_at, content
    """
    logger.info(f"Loading Nostr data from {start_date} to {end_date}")

    start = pd.Timestamp(start_date, tz='UTC')
    end = pd.Timestamp(end_date, tz='UTC')

    # Find relevant files
    all_files = sorted(data_dir.glob('nostr_*.parquet'))
    dfs = []

    for fpath in tqdm(all_files, desc="Loading Nostr files"):
        try:
            df = pd.read_parquet(fpath)

            # Filter by date
            if 'created_at' in df.columns:
                df = df[(df['created_at'] >= start) & (df['created_at'] <= end)]

                # Filter only text posts (kind=1), not zaps
                if 'kind' in df.columns:
                    df = df[df['kind'] == 1]

                if len(df) > 0:
                    dfs.append(df[['created_at', 'content']])

        except Exception as e:
            logger.warning(f"Error loading {fpath}: {e}")

    if not dfs:
        logger.warning("No Nostr data found")
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(result):,} Nostr posts")

    return result


def process_texts_to_embeddings(
    df: pd.DataFrame,
    timestamp_col: str,
    text_cols: List[str],
    processor,
    batch_size: int = 256,
    n_assets: int = 20,
) -> Tuple[np.ndarray, List[str]]:
    """
    Process texts to embeddings with asset mapping.

    Args:
        df: DataFrame with timestamp and text columns
        timestamp_col: Name of timestamp column
        text_cols: List of text column names to concatenate
        processor: FinBERT or CryptoBERT processor
        batch_size: Batch size for embedding
        n_assets: Number of assets

    Returns:
        embeddings: (N, 768) array
        keys: List of "timestamp_assetIdx" strings
    """
    logger.info(f"Processing {len(df):,} texts to embeddings...")

    # Prepare (timestamp_key, asset_idx, text) tuples
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Detecting assets"):
        ts = row[timestamp_col]

        # Combine text columns
        texts = [str(row.get(col, '')) for col in text_cols if col in row.index]
        combined_text = ' '.join(filter(None, texts))

        if not combined_text.strip():
            continue

        # Round timestamp
        ts_key = round_to_5min(ts)

        # Detect mentioned assets
        mentioned_assets = detect_assets_in_text(combined_text)

        # If no specific asset mentioned, assign to BTC (general crypto news)
        if not mentioned_assets:
            mentioned_assets = [0]  # Default to BTC

        for asset_idx in mentioned_assets:
            records.append((ts_key, asset_idx, combined_text))

    if not records:
        logger.warning("No records to process")
        return np.array([]).reshape(0, 768), []

    logger.info(f"Generated {len(records):,} (timestamp, asset, text) records")

    # Aggregate by (timestamp, asset): collect all texts
    aggregated: Dict[str, List[str]] = {}
    for ts_key, asset_idx, text in records:
        key = f"{ts_key}_{asset_idx}"
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(text)

    logger.info(f"Aggregated to {len(aggregated):,} unique (timestamp, asset) pairs")

    # Process embeddings in batches
    keys = list(aggregated.keys())
    all_embeddings = []

    # For each key, we'll compute embedding of concatenated texts
    # (or use mean pooling of individual embeddings for better representation)

    batch_texts = []
    batch_keys = []

    for key in tqdm(keys, desc="Preparing batches"):
        texts = aggregated[key]

        # Option 1: Concatenate (simpler, faster)
        combined = ' '.join(texts[:5])  # Limit to first 5 texts

        # Truncate to reasonable length
        if len(combined) > 2000:
            combined = combined[:2000]

        batch_texts.append(combined)
        batch_keys.append(key)

    # Compute embeddings in batches
    logger.info(f"Computing embeddings for {len(batch_texts):,} texts...")

    for i in tqdm(range(0, len(batch_texts), batch_size), desc="Computing embeddings"):
        batch = batch_texts[i:i + batch_size]
        embeddings = processor.get_embeddings(batch)
        all_embeddings.append(embeddings)

    final_embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Final embeddings shape: {final_embeddings.shape}")

    return final_embeddings, batch_keys


def save_embeddings_h5(
    embeddings: np.ndarray,
    keys: List[str],
    output_path: Path,
    model_name: str,
):
    """
    Save embeddings to HDF5 file.

    Args:
        embeddings: (N, 768) array
        keys: List of "timestamp_assetIdx" strings
        output_path: Output HDF5 file path
        model_name: Name of model used
    """
    logger.info(f"Saving {len(keys):,} embeddings to {output_path}")

    # Create parent directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Store embeddings as float16 to save space
        f.create_dataset(
            'embeddings',
            data=embeddings.astype(np.float16),
            compression='gzip',
            compression_opts=9,
        )

        # Store keys as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('keys', data=keys, dtype=dt)

        # Metadata
        f.attrs['model_name'] = model_name
        f.attrs['embedding_dim'] = 768
        f.attrs['num_embeddings'] = len(keys)
        f.attrs['created_at'] = datetime.now().isoformat()

    # Print file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute BERT embeddings')
    parser.add_argument('--source', type=str, required=True, choices=['gdelt', 'nostr'],
                        help='Data source: gdelt or nostr')
    parser.add_argument('--start', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference (cuda:0, cuda:1, cpu)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for embedding')
    parser.add_argument('--output', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--data-dir', type=str, default='/home/work/data/stair-local',
                        help='Base data directory')

    args = parser.parse_args()

    logger.info(f"Pre-computing {args.source} embeddings")
    logger.info(f"Date range: {args.start} to {args.end}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output: {args.output}")

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    if args.source == 'gdelt':
        # Load GDELT data
        source_dir = data_dir / 'gdelt'
        df = load_gdelt_data(source_dir, args.start, args.end)

        if df.empty:
            logger.error("No data loaded")
            return 1

        # Initialize FinBERT processor
        logger.info("Initializing FinBERT processor...")
        processor = FinBERTProcessor(
            device=args.device,
            batch_size=args.batch_size,
        )

        # Process to embeddings
        embeddings, keys = process_texts_to_embeddings(
            df=df,
            timestamp_col='published_at',
            text_cols=['title', 'content'],
            processor=processor,
            batch_size=args.batch_size,
        )

        model_name = 'ProsusAI/finbert'

    else:  # nostr
        # Load Nostr data
        source_dir = data_dir / 'nostr'
        df = load_nostr_data(source_dir, args.start, args.end)

        if df.empty:
            logger.error("No data loaded")
            return 1

        # Initialize CryptoBERT processor
        logger.info("Initializing CryptoBERT processor...")
        processor = CryptoBERTProcessor(
            device=args.device,
            batch_size=args.batch_size,
        )

        # Process to embeddings
        embeddings, keys = process_texts_to_embeddings(
            df=df,
            timestamp_col='created_at',
            text_cols=['content'],
            processor=processor,
            batch_size=args.batch_size,
        )

        model_name = 'ElKulako/cryptobert'

    # Save to HDF5
    save_embeddings_h5(embeddings, keys, output_path, model_name)

    logger.info("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
