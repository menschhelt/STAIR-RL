"""
Tensor Builder - Constructs 3D tensors for RL state input.

Combines:
- Alpha factors from AlphaCalculator
- Universe from UniverseManager
- Text features from TextAggregator
- Rank normalization across the current universe

Output shape: (T, Slots, Features)
- T: Number of time steps
- Slots: 20 (universe size)
- Features: ~300+ (alphas + factors + text features)
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from config.settings import DATA_DIR
from features.alpha_calculator import AlphaCalculator
from universe.manager import UniverseManager


class RankNormalizer:
    """
    Applies cross-sectional rank normalization at each time step.

    At time t:
    1. Filter to current universe (top 20)
    2. Rank each feature across symbols
    3. Scale to [0, 1]

    This makes the agent's learning asset-agnostic:
    - "Rank 1 has highest momentum" instead of "BTC has momentum of 0.5"
    """

    def __init__(self, top_n: int = 20):
        """
        Initialize Rank Normalizer.

        Args:
            top_n: Universe size (default: 20)
        """
        self.top_n = top_n
        self.logger = logging.getLogger(self.__class__.__name__)

    def normalize_cross_section(
        self,
        features_dict: Dict[str, pd.DataFrame],
        universe: Dict[int, Optional[str]],
        timestamp: datetime,
    ) -> np.ndarray:
        """
        Rank-normalize features at a single timestamp.

        Args:
            features_dict: {symbol: features_df} with features at timestamp
            universe: {slot: symbol} mapping for current date
            timestamp: Current timestamp

        Returns:
            Array of shape (slots, features) with rank-normalized values
        """
        # Get feature names from first available symbol
        feature_names = []
        for df in features_dict.values():
            if not df.empty:
                # Exclude metadata columns
                exclude = {'timestamp', 'symbol', 'date'}
                feature_names = [c for c in df.columns if c not in exclude]
                break

        if not feature_names:
            return np.zeros((self.top_n, 1))

        # Build matrix (slots x features)
        matrix = np.full((self.top_n, len(feature_names)), np.nan)

        for slot in range(1, self.top_n + 1):
            symbol = universe.get(slot)
            if symbol and symbol in features_dict:
                df = features_dict[symbol]
                if not df.empty:
                    # Get row closest to timestamp
                    if 'timestamp' in df.columns:
                        idx = (df['timestamp'] - timestamp).abs().idxmin()
                        row = df.loc[idx, feature_names]
                    else:
                        row = df.iloc[-1][feature_names]

                    matrix[slot - 1] = row.values

        # Rank normalize each feature (column)
        normalized = np.zeros_like(matrix)
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            valid_mask = ~np.isnan(col)

            if valid_mask.sum() > 1:
                # Rank among valid values
                ranks = np.zeros_like(col)
                valid_values = col[valid_mask]
                rank_order = valid_values.argsort().argsort()
                ranks[valid_mask] = rank_order / (len(rank_order) - 1)  # Scale to [0, 1]
                normalized[:, j] = ranks
            else:
                # Not enough data to rank
                normalized[:, j] = 0.5  # Neutral value

        return normalized

    def normalize_batch(
        self,
        features_dict: Dict[str, pd.DataFrame],
        universe_history: Dict[date, Dict[int, Optional[str]]],
        timestamps: List[datetime],
    ) -> np.ndarray:
        """
        Rank-normalize features for multiple timestamps.

        Args:
            features_dict: {symbol: features_df}
            universe_history: {date: {slot: symbol}}
            timestamps: List of timestamps to process

        Returns:
            Array of shape (T, slots, features)
        """
        results = []

        for ts in timestamps:
            # Get universe for this date
            ts_date = ts.date() if isinstance(ts, datetime) else ts
            universe = universe_history.get(ts_date, {i: None for i in range(1, self.top_n + 1)})

            normalized = self.normalize_cross_section(features_dict, universe, ts)
            results.append(normalized)

        return np.stack(results, axis=0)


class TensorBuilder:
    """
    Constructs the final 3D tensor for RL training.

    Output shape: (T, Slots=20, Features)
    - T: Number of time steps
    - Slots: 20 (universe size)
    - Features: alphas + factors + text features

    Features (~300+):
    - Alpha 101: ~100 features
    - Alpha 191: ~190 features
    - 9 factors
    - Text features: 6 (3 Nostr + 3 GDELT)
    """

    def __init__(
        self,
        universe_manager: Optional[UniverseManager] = None,
        alpha_calculator: Optional[AlphaCalculator] = None,
        top_n: int = 20,
    ):
        """
        Initialize Tensor Builder.

        Args:
            universe_manager: UniverseManager instance
            alpha_calculator: AlphaCalculator instance
            top_n: Universe size
        """
        self.universe_manager = universe_manager or UniverseManager()
        self.alpha_calculator = alpha_calculator or AlphaCalculator()
        self.rank_normalizer = RankNormalizer(top_n=top_n)
        self.top_n = top_n

        # Output directory
        self.tensor_dir = DATA_DIR / 'features' / 'tensors'
        self.tensor_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def build_tensor(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str = '5m',
        include_text: bool = True,
        text_features: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, List[str], List[datetime]]:
        """
        Build 3D tensor for the date range.

        Args:
            start_date: Start of period
            end_date: End of period
            interval: Data interval
            include_text: Whether to include text features
            text_features: Pre-calculated text features DataFrame

        Returns:
            Tuple of:
            - numpy array of shape (T, 20, num_features)
            - list of feature names
            - list of timestamps
        """
        self.logger.info(f"Building tensor from {start_date} to {end_date}")

        # Load universe history
        universe_history = self._load_universe_history(start_date.date(), end_date.date())

        # Get all symbols in universe during period
        all_symbols = set()
        for date_universe in universe_history.values():
            for sym in date_universe.values():
                if sym:
                    all_symbols.add(sym)

        self.logger.info(f"Loading features for {len(all_symbols)} symbols")

        # Load cached features for all symbols
        features_dict = {}
        for symbol in all_symbols:
            df = self.alpha_calculator.load_cached_features(
                symbol,
                start_date=start_date,
                end_date=end_date,
            )
            if not df.empty:
                features_dict[symbol] = df

        if not features_dict:
            self.logger.error("No features found")
            return np.array([]), [], []

        # Get all unique timestamps
        all_timestamps = set()
        for df in features_dict.values():
            if 'timestamp' in df.columns:
                all_timestamps.update(df['timestamp'].tolist())

        timestamps = sorted(all_timestamps)
        self.logger.info(f"Processing {len(timestamps)} timestamps")

        # Build tensor
        tensor_slices = []
        feature_names = None

        for ts in timestamps:
            ts_date = ts.date() if isinstance(ts, datetime) else ts
            universe = universe_history.get(ts_date, {})

            # Get features at this timestamp
            ts_features = {}
            for symbol, df in features_dict.items():
                if 'timestamp' in df.columns:
                    ts_df = df[df['timestamp'] == ts]
                    if not ts_df.empty:
                        ts_features[symbol] = ts_df

            if not ts_features:
                continue

            # Get feature names once
            if feature_names is None:
                for df in ts_features.values():
                    exclude = {'timestamp', 'symbol', 'date'}
                    feature_names = [c for c in df.columns if c not in exclude]
                    break

            # Rank normalize
            normalized = self.rank_normalizer.normalize_cross_section(
                ts_features, universe, ts
            )

            tensor_slices.append(normalized)

        if not tensor_slices:
            return np.array([]), [], []

        tensor = np.stack(tensor_slices, axis=0)

        # Add text features if available
        if include_text and text_features is not None:
            tensor, feature_names = self._add_text_features(
                tensor, timestamps, text_features, feature_names
            )

        self.logger.info(
            f"Built tensor with shape {tensor.shape}: "
            f"(T={tensor.shape[0]}, Slots={tensor.shape[1]}, Features={tensor.shape[2]})"
        )

        return tensor, feature_names, timestamps

    def _load_universe_history(
        self,
        start_date: date,
        end_date: date,
    ) -> Dict[date, Dict[int, Optional[str]]]:
        """Load universe for each date in range."""
        history = {}
        current = start_date

        while current <= end_date:
            universe = self.universe_manager.get_universe_for_date(current)
            history[current] = universe
            current += timedelta(days=1)

        return history

    def _add_text_features(
        self,
        tensor: np.ndarray,
        timestamps: List[datetime],
        text_features: pd.DataFrame,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Add text sentiment features to tensor.

        Text features are the same for all slots (market-wide sentiment).

        Args:
            tensor: Original tensor (T, Slots, Features)
            timestamps: List of timestamps
            text_features: DataFrame with text features
            feature_names: Current feature names

        Returns:
            Updated tensor and feature names
        """
        # Text feature columns
        text_cols = [
            'nostr_bearish_weighted', 'nostr_neutral_weighted', 'nostr_bullish_weighted',
            'gdelt_positive_weighted', 'gdelt_negative_weighted', 'gdelt_neutral_weighted',
        ]

        # Filter to available columns
        available_cols = [c for c in text_cols if c in text_features.columns]
        if not available_cols:
            return tensor, feature_names

        # Build text feature array
        text_array = np.zeros((len(timestamps), self.top_n, len(available_cols)))

        for i, ts in enumerate(timestamps):
            # Find matching row in text_features
            if 'timestamp' in text_features.columns:
                mask = text_features['timestamp'] == ts
                if mask.any():
                    row = text_features.loc[mask, available_cols].iloc[0].values
                    # Broadcast to all slots (same sentiment for all)
                    text_array[i] = np.tile(row, (self.top_n, 1))

        # Concatenate with main tensor
        combined = np.concatenate([tensor, text_array], axis=2)

        # Update feature names
        combined_names = feature_names + available_cols

        return combined, combined_names

    def save_tensor(
        self,
        tensor: np.ndarray,
        feature_names: List[str],
        timestamps: List[datetime],
        name: str,
    ):
        """
        Save tensor to disk.

        Args:
            tensor: Tensor array
            feature_names: Feature names
            timestamps: Timestamps
            name: Name for the saved tensor
        """
        np.savez_compressed(
            self.tensor_dir / f"{name}.npz",
            tensor=tensor,
            feature_names=feature_names,
            timestamps=np.array([ts.isoformat() for ts in timestamps]),
        )
        self.logger.info(f"Saved tensor to {self.tensor_dir / name}.npz")

    def load_tensor(self, name: str) -> Tuple[np.ndarray, List[str], List[datetime]]:
        """
        Load tensor from disk.

        Args:
            name: Name of saved tensor

        Returns:
            Tuple of (tensor, feature_names, timestamps)
        """
        path = self.tensor_dir / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Tensor not found: {path}")

        data = np.load(path, allow_pickle=True)
        tensor = data['tensor']
        feature_names = data['feature_names'].tolist()
        timestamps = [datetime.fromisoformat(ts) for ts in data['timestamps']]

        return tensor, feature_names, timestamps


# ========== Standalone Execution ==========

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tensor Builder')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--name', type=str, default='train', help='Tensor name')
    parser.add_argument('--save', action='store_true', help='Save tensor to disk')

    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    builder = TensorBuilder()

    tensor, feature_names, timestamps = builder.build_tensor(
        start_date=start,
        end_date=end,
        include_text=False,  # Text features built separately
    )

    print(f"Tensor shape: {tensor.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Timestamps: {len(timestamps)}")

    if args.save:
        builder.save_tensor(tensor, feature_names, timestamps, args.name)
