#!/usr/bin/env python3
"""
Mock Training Data Preprocessor for STAIR-RL

Converts mock parquet files to tensor format for TradingEnv.set_data().

Input:
    /home/work/data/stair-local/test_mock/features/mock_features/*.parquet

Output:
    /home/work/data/stair-local/test_mock/tensors/
    ├── train_data.npz  (2021-01-01 to 2021-01-31)
    └── val_data.npz    (2021-02-01 to 2021-02-28)

Format:
    {
        'states': np.ndarray (T, N, 36),       # Market features
        'returns': np.ndarray (T, N),          # Asset returns
        'prices': np.ndarray (T, N),           # Asset prices
        'funding_rates': np.ndarray (T, N),    # Zeros for mock
        'timestamps': List[datetime]           # Timestamps
    }

Usage:
    python scripts/prepare_mock_training_data.py \
        --input /home/work/data/stair-local/test_mock \
        --output /home/work/data/stair-local/test_mock/tensors
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockDataPreprocessor:
    """
    Convert mock parquet data to tensor format for training.
    """

    def __init__(
        self,
        input_dir: Path,
        train_start: str = '2021-01-01',
        train_end: str = '2021-01-31',
        val_start: str = '2021-02-01',
        val_end: str = '2021-02-28',
    ):
        """
        Initialize preprocessor.

        Args:
            input_dir: Input directory with mock_features/ subdirectory
            train_start: Training start date
            train_end: Training end date
            val_start: Validation start date
            val_end: Validation end date
        """
        self.input_dir = input_dir
        self.features_dir = input_dir / 'features' / 'mock_features'

        self.train_start = pd.Timestamp(train_start, tz='UTC')
        self.train_end = pd.Timestamp(train_end, tz='UTC')
        self.val_start = pd.Timestamp(val_start, tz='UTC')
        self.val_end = pd.Timestamp(val_end, tz='UTC')

        logger.info("Initialized MockDataPreprocessor")
        logger.info(f"  Train: {train_start} to {train_end}")
        logger.info(f"  Val: {val_start} to {val_end}")

    def load_features(self) -> Dict[str, pd.DataFrame]:
        """
        Load all feature parquet files.

        Returns:
            Dict of {symbol: features_df}
        """
        logger.info("Loading feature files...")

        all_features = {}
        feature_files = sorted(self.features_dir.glob('*_features.parquet'))

        if not feature_files:
            logger.error(f"No feature files found in {self.features_dir}")
            return {}

        for fpath in tqdm(feature_files, desc="Loading features"):
            symbol = fpath.stem.replace('_features', '')

            try:
                df = pd.read_parquet(fpath)
                df = df.sort_values('timestamp').reset_index(drop=True)
                all_features[symbol] = df
                logger.debug(f"  Loaded {symbol}: {len(df):,} rows")

            except Exception as e:
                logger.warning(f"Error loading {fpath}: {e}")

        logger.info(f"Loaded features for {len(all_features)} symbols")

        return all_features

    def align_data(
        self,
        all_features: Dict[str, pd.DataFrame],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Align features for all symbols in time range.

        Args:
            all_features: Dict of {symbol: features_df}
            start: Start timestamp
            end: End timestamp

        Returns:
            aligned_df: DataFrame with aligned data
            symbols: List of symbols in order
        """
        logger.info(f"Aligning data for {start.date()} to {end.date()}...")

        symbols = sorted(all_features.keys())

        # Find common timestamps
        common_timestamps = None

        for symbol in symbols:
            df = all_features[symbol]
            df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

            if common_timestamps is None:
                common_timestamps = set(df['timestamp'].values)
            else:
                common_timestamps = common_timestamps.intersection(df['timestamp'].values)

        if not common_timestamps:
            logger.error("No common timestamps found!")
            return pd.DataFrame(), symbols

        common_timestamps = sorted(common_timestamps)
        logger.info(f"  Found {len(common_timestamps):,} common timestamps")

        # Build aligned dataframe
        aligned_data = []

        for ts in tqdm(common_timestamps, desc="Aligning"):
            # Convert to pandas Timestamp with UTC timezone for comparison
            ts_pd = pd.Timestamp(ts, tz='UTC')

            for symbol in symbols:
                df = all_features[symbol]

                # Filter for this timestamp
                row = df[df['timestamp'] == ts_pd]

                if len(row) == 0:
                    continue

                row = row.iloc[0]

                # Build record with all features
                record = {
                    'timestamp': ts_pd,
                    'symbol': symbol,
                }

                # Add all feature columns
                for col in row.index:
                    if col not in ['timestamp', 'symbol']:
                        record[col] = row[col]

                aligned_data.append(record)

        aligned_df = pd.DataFrame(aligned_data)
        logger.info(f"  Aligned data: {len(aligned_df):,} rows")

        return aligned_df, symbols

    def build_tensors(
        self,
        aligned_df: pd.DataFrame,
        symbols: List[str],
    ) -> Dict:
        """
        Build tensor arrays from aligned dataframe.

        Args:
            aligned_df: Aligned dataframe
            symbols: List of symbols in order

        Returns:
            Dict with states, returns, prices, funding_rates, timestamps
        """
        logger.info("Building tensors...")

        # Get unique timestamps
        timestamps = sorted(aligned_df['timestamp'].unique())
        T = len(timestamps)
        N = len(symbols)

        logger.info(f"  Shape: T={T}, N={N}")

        # Initialize arrays
        states = np.zeros((T, N, 36), dtype=np.float32)
        prices = np.zeros((T, N), dtype=np.float32)
        returns = np.zeros((T, N), dtype=np.float32)
        funding_rates = np.zeros((T, N), dtype=np.float32)

        # Get feature column names (f01 to f36)
        feature_cols = [f'f{i:02d}' for i in range(1, 37)]

        # Fill arrays
        for t, ts in enumerate(tqdm(timestamps, desc="Building tensors")):
            for n, symbol in enumerate(symbols):
                # Get row for this (timestamp, symbol)
                row = aligned_df[(aligned_df['timestamp'] == ts) &
                                (aligned_df['symbol'] == symbol)]

                if len(row) == 0:
                    continue

                row = row.iloc[0]

                # Extract features (f01 to f36)
                features = []
                for i in range(1, 37):
                    # Find column that starts with f{i:02d}_
                    prefix = f'f{i:02d}_'
                    matching_cols = [c for c in row.index if c.startswith(prefix)]

                    if matching_cols:
                        features.append(row[matching_cols[0]])
                    else:
                        features.append(0.0)

                states[t, n, :] = features

                # Price (from f04_close_norm which is normalized, need actual close)
                # We'll use a synthetic price based on normalized value
                # In real case, we'd load actual OHLCV
                # For now, use normalized close as proxy
                prices[t, n] = row.get('f04_close_norm', 1.0) * 30000  # Mock price

        # Compute returns
        for n in range(N):
            price_series = prices[:, n]
            returns[1:, n] = np.log(price_series[1:] / (price_series[:-1] + 1e-8))
            returns[0, n] = 0.0

        # Convert timestamps to datetime list
        timestamp_list = [ts.to_pydatetime() for ts in timestamps]

        logger.info(f"  states: {states.shape}")
        logger.info(f"  returns: {returns.shape}")
        logger.info(f"  prices: {prices.shape}")

        return {
            'states': states,
            'returns': returns,
            'prices': prices,
            'funding_rates': funding_rates,
            'timestamps': timestamp_list,
        }

    def save_tensors(self, data: Dict, output_path: Path):
        """
        Save tensor data to .npz file.

        Args:
            data: Dict with states, returns, prices, funding_rates, timestamps
            output_path: Output .npz file path
        """
        logger.info(f"Saving tensors to {output_path}...")

        # Create parent directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert timestamps to strings for npz
        data_to_save = {
            'states': data['states'],
            'returns': data['returns'],
            'prices': data['prices'],
            'funding_rates': data['funding_rates'],
            'timestamps': np.array([ts.isoformat() for ts in data['timestamps']], dtype=str),
        }

        np.savez_compressed(output_path, **data_to_save)

        # Print file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Saved {output_path.name} ({size_mb:.1f} MB)")

    def process(self, output_dir: Path):
        """
        Main processing pipeline.

        Args:
            output_dir: Output directory for tensors
        """
        logger.info("=" * 60)
        logger.info("MOCK DATA PREPROCESSING")
        logger.info("=" * 60)

        # Load features
        all_features = self.load_features()

        if not all_features:
            logger.error("No features loaded!")
            return

        # Process training data
        logger.info("\n--- Processing Training Data ---")
        train_df, symbols = self.align_data(all_features, self.train_start, self.train_end)

        if train_df.empty:
            logger.error("No training data aligned!")
            return

        train_tensors = self.build_tensors(train_df, symbols)
        self.save_tensors(train_tensors, output_dir / 'train_data.npz')

        # Process validation data
        logger.info("\n--- Processing Validation Data ---")
        val_df, _ = self.align_data(all_features, self.val_start, self.val_end)

        if val_df.empty:
            logger.error("No validation data aligned!")
            return

        val_tensors = self.build_tensors(val_df, symbols)
        self.save_tensors(val_tensors, output_dir / 'val_data.npz')

        logger.info("\n✓ Preprocessing complete!")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"  {', '.join(symbols)}")
        logger.info(f"\nTraining data:")
        logger.info(f"  Shape: {train_tensors['states'].shape}")
        logger.info(f"  Timesteps: {len(train_tensors['timestamps'])}")
        logger.info(f"\nValidation data:")
        logger.info(f"  Shape: {val_tensors['states'].shape}")
        logger.info(f"  Timesteps: {len(val_tensors['timestamps'])}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess mock data to tensors')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory (contains features/mock_features/)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for tensors')
    parser.add_argument('--train-start', type=str, default='2021-01-01',
                        help='Training start date')
    parser.add_argument('--train-end', type=str, default='2021-01-31',
                        help='Training end date')
    parser.add_argument('--val-start', type=str, default='2021-02-01',
                        help='Validation start date')
    parser.add_argument('--val-end', type=str, default='2021-02-28',
                        help='Validation end date')

    args = parser.parse_args()

    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    preprocessor = MockDataPreprocessor(
        input_dir=Path(args.input),
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
    )

    preprocessor.process(Path(args.output))

    logger.info("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
