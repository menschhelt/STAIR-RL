"""
PCA Compressor - Dimensionality reduction for alpha factors.

Compresses 300+ alpha factors into ~20 principal components
using rolling window PCA to avoid look-ahead bias.

Key features:
- Rolling window fit (default 252 days = 1 year)
- Look-ahead bias prevention
- Variance explained tracking
- Feature loading analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class AlphaPCACompressor:
    """
    Rolling window PCA for alpha factor compression.

    Compresses 300+ alpha factors to ~20 principal components while
    preventing look-ahead bias by only fitting on historical data.

    Usage:
        compressor = AlphaPCACompressor(n_components=20)

        # Fit on historical data
        compressor.fit(historical_alphas)

        # Transform current alphas
        compressed = compressor.transform(current_alphas)
    """

    def __init__(
        self,
        n_components: int = 20,
        lookback_days: int = 30,  # 30일 (5분봉 기준 8,640 샘플)
        min_observations: int = 4000,  # 약 2주치 (291 알파 * 3 ≈ 900 최소)
        explained_variance_threshold: float = 0.95,
    ):
        """
        Initialize PCA Compressor.

        Args:
            n_components: Number of principal components to extract
            lookback_days: Days of history for fitting PCA (30일 권장)
            min_observations: Minimum observations required for fitting (4000+ 권장)
            explained_variance_threshold: Target cumulative variance to explain
        """
        self.n_components = n_components
        self.lookback_days = lookback_days
        self.min_observations = min_observations
        self.explained_variance_threshold = explained_variance_threshold

        # Sklearn components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)

        # Fitted state
        self._is_fitted = False
        self._fitted_at: Optional[datetime] = None
        self._loadings: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None
        self._explained_variance_ratio: Optional[np.ndarray] = None

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

    def fit(
        self,
        alpha_matrix: np.ndarray,
        feature_names: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> 'AlphaPCACompressor':
        """
        Fit PCA on historical alpha matrix.

        Args:
            alpha_matrix: Shape (T, num_alphas) - historical alpha values
            feature_names: Names of alpha features
            timestamp: Timestamp of fit (for tracking)

        Returns:
            self for chaining
        """
        if alpha_matrix.shape[0] < self.min_observations:
            raise ValueError(
                f"Insufficient observations: {alpha_matrix.shape[0]} < {self.min_observations}"
            )

        # Handle NaN values
        alpha_matrix = self._handle_nan(alpha_matrix)

        # 1. Standardize (z-score)
        scaled = self.scaler.fit_transform(alpha_matrix)

        # 2. Fit PCA
        self.pca.fit(scaled)

        # 3. Store metadata
        self._loadings = self.pca.components_.T  # (num_alphas, n_components)
        self._explained_variance_ratio = self.pca.explained_variance_ratio_
        self._feature_names = feature_names or [f"alpha_{i}" for i in range(alpha_matrix.shape[1])]
        self._fitted_at = timestamp or datetime.utcnow()
        self._is_fitted = True

        # Log fit results
        cumulative_var = self.get_cumulative_variance()
        self.logger.info(
            f"PCA fitted: {self.n_components} components explain "
            f"{cumulative_var:.1%} of variance"
        )

        return self

    def transform(
        self,
        alpha_vector: np.ndarray,
        handle_unfitted: str = 'raise',
    ) -> np.ndarray:
        """
        Transform alpha values to principal components.

        Args:
            alpha_vector: Shape (num_alphas,) or (n_samples, num_alphas)
            handle_unfitted: How to handle unfitted state: 'raise', 'zeros', 'identity'

        Returns:
            compressed: Shape (n_components,) or (n_samples, n_components)
        """
        if not self._is_fitted:
            if handle_unfitted == 'raise':
                raise RuntimeError("PCA not fitted. Call fit() first.")
            elif handle_unfitted == 'zeros':
                return np.zeros(self.n_components)
            elif handle_unfitted == 'identity':
                return alpha_vector[:self.n_components]

        # Ensure 2D
        is_1d = alpha_vector.ndim == 1
        if is_1d:
            alpha_vector = alpha_vector.reshape(1, -1)

        # Handle NaN
        alpha_vector = self._handle_nan(alpha_vector)

        # Standardize using fitted scaler
        scaled = self.scaler.transform(alpha_vector)

        # Transform to PC space
        compressed = self.pca.transform(scaled)

        return compressed.squeeze() if is_1d else compressed

    def fit_transform(
        self,
        alpha_matrix: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Fit PCA and transform in one step.

        Args:
            alpha_matrix: Shape (T, num_alphas)
            feature_names: Names of alpha features

        Returns:
            compressed: Shape (T, n_components)
        """
        self.fit(alpha_matrix, feature_names)
        return self.transform(alpha_matrix)

    def fit_transform_rolling(
        self,
        alpha_df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        refit_interval_days: int = 1,
        samples_per_day: int = 288,  # 5분봉 = 288, 1시간봉 = 24
    ) -> Tuple[np.ndarray, List[str], List[datetime]]:
        """
        Apply rolling window PCA to time series.

        This method prevents look-ahead bias by only fitting on
        historical data at each time step.

        Args:
            alpha_df: DataFrame with timestamp and alpha columns
            timestamp_col: Name of timestamp column
            refit_interval_days: Days between refitting PCA (default: 1 = daily refit)
            samples_per_day: Samples per day (288 for 5min, 24 for 1h)

        Returns:
            Tuple of:
            - compressed: (T, n_components) array
            - component_names: ['PC1', 'PC2', ..., 'PC_n']
            - timestamps: List of timestamps
        """
        # Ensure sorted by timestamp
        alpha_df = alpha_df.sort_values(timestamp_col).reset_index(drop=True)

        # Get ONLY alpha columns (exclude factors, OHLCV, metadata)
        # Factors (9개) have economic meaning → keep raw, don't compress
        # Only alphas (291개) go through PCA compression
        alpha_cols = [c for c in alpha_df.columns if 'alpha_' in c.lower()]

        timestamps = alpha_df[timestamp_col].tolist()
        alpha_matrix = alpha_df[alpha_cols].values

        # Output
        n_samples = len(timestamps)
        compressed = np.full((n_samples, self.n_components), np.nan)

        # Convert days to samples (핵심 수정!)
        # 예: lookback_days=30, 5분봉 → 30 * 288 = 8,640 샘플
        lookback_samples = self.lookback_days * samples_per_day
        refit_interval_samples = refit_interval_days * samples_per_day

        self.logger.info(
            f"Rolling PCA: lookback={self.lookback_days}days ({lookback_samples} samples), "
            f"refit every {refit_interval_days}days ({refit_interval_samples} samples)"
        )

        # Track last fit time
        last_fit_idx = -refit_interval_samples

        for i in range(n_samples):
            # Check if we have enough history
            if i < self.min_observations:
                continue

            # Determine lookback window
            start_idx = max(0, i - lookback_samples)
            history = alpha_matrix[start_idx:i]

            # Refit if needed (샘플 단위로 비교)
            if i - last_fit_idx >= refit_interval_samples:
                try:
                    self.fit(history, feature_names=alpha_cols)
                    last_fit_idx = i
                    if i % (samples_per_day * 7) == 0:  # 주 1회 로그
                        self.logger.info(f"PCA refitted at sample {i}, history size: {len(history)}")
                except Exception as e:
                    self.logger.warning(f"PCA fit failed at index {i}: {e}")
                    continue

            # Transform current observation
            if self._is_fitted:
                compressed[i] = self.transform(alpha_matrix[i])

        component_names = [f'PC{i+1}' for i in range(self.n_components)]

        return compressed, component_names, timestamps

    def _handle_nan(self, matrix: np.ndarray) -> np.ndarray:
        """Handle NaN values in matrix."""
        if np.isnan(matrix).any():
            # Replace NaN with column mean
            col_means = np.nanmean(matrix, axis=0)
            nan_mask = np.isnan(matrix)
            matrix = matrix.copy()

            for j in range(matrix.shape[1]):
                matrix[nan_mask[:, j], j] = col_means[j] if not np.isnan(col_means[j]) else 0.0

        return matrix

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get variance explained by each component."""
        if not self._is_fitted:
            raise RuntimeError("PCA not fitted")
        return self._explained_variance_ratio

    def get_cumulative_variance(self) -> float:
        """Get total variance explained by all components."""
        if not self._is_fitted:
            raise RuntimeError("PCA not fitted")
        return float(self._explained_variance_ratio.sum())

    def get_loadings(self) -> pd.DataFrame:
        """
        Get factor loadings (contribution of each alpha to each PC).

        Returns:
            DataFrame with shape (num_alphas, n_components)
        """
        if not self._is_fitted:
            raise RuntimeError("PCA not fitted")

        return pd.DataFrame(
            self._loadings,
            index=self._feature_names,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
        )

    def get_top_contributors(
        self,
        component: int,
        n_top: int = 10,
    ) -> pd.Series:
        """
        Get top contributing alphas for a specific component.

        Args:
            component: Component index (1-based)
            n_top: Number of top contributors to return

        Returns:
            Series with alpha names and their loadings
        """
        if not self._is_fitted:
            raise RuntimeError("PCA not fitted")

        loadings = self._loadings[:, component - 1]
        abs_loadings = np.abs(loadings)

        top_idx = abs_loadings.argsort()[-n_top:][::-1]

        return pd.Series(
            loadings[top_idx],
            index=[self._feature_names[i] for i in top_idx],
            name=f'PC{component}_loadings',
        )

    def save(self, path: Union[str, Path]):
        """Save fitted PCA to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'n_components': self.n_components,
            'lookback_days': self.lookback_days,
            'min_observations': self.min_observations,
            'scaler': self.scaler,
            'pca': self.pca,
            'is_fitted': self._is_fitted,
            'fitted_at': self._fitted_at,
            'loadings': self._loadings,
            'feature_names': self._feature_names,
            'explained_variance_ratio': self._explained_variance_ratio,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        self.logger.info(f"PCA saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'AlphaPCACompressor':
        """Load fitted PCA from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        compressor = cls(
            n_components=state['n_components'],
            lookback_days=state['lookback_days'],
            min_observations=state['min_observations'],
        )

        compressor.scaler = state['scaler']
        compressor.pca = state['pca']
        compressor._is_fitted = state['is_fitted']
        compressor._fitted_at = state['fitted_at']
        compressor._loadings = state['loadings']
        compressor._feature_names = state['feature_names']
        compressor._explained_variance_ratio = state['explained_variance_ratio']

        return compressor


class IncrementalPCACompressor:
    """
    Incremental PCA for streaming data.

    Uses sklearn's IncrementalPCA for memory-efficient processing
    of large datasets that don't fit in memory.
    """

    def __init__(
        self,
        n_components: int = 20,
        batch_size: int = 1000,
    ):
        """
        Initialize Incremental PCA.

        Args:
            n_components: Number of components
            batch_size: Batch size for incremental fitting
        """
        from sklearn.decomposition import IncrementalPCA

        self.n_components = n_components
        self.batch_size = batch_size

        self.scaler = StandardScaler()
        self.pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

        self._is_fitted = False

    def partial_fit(self, alpha_batch: np.ndarray):
        """
        Incrementally fit PCA on a batch.

        Args:
            alpha_batch: Shape (batch_size, num_alphas)
        """
        # Handle NaN
        alpha_batch = np.nan_to_num(alpha_batch, nan=0.0)

        # Partial fit scaler and transform
        scaled = self.scaler.partial_fit(alpha_batch).transform(alpha_batch)

        # Partial fit PCA
        self.pca.partial_fit(scaled)
        self._is_fitted = True

    def transform(self, alpha_vector: np.ndarray) -> np.ndarray:
        """Transform using incrementally fitted PCA."""
        if not self._is_fitted:
            raise RuntimeError("PCA not fitted")

        alpha_vector = np.nan_to_num(alpha_vector, nan=0.0)

        is_1d = alpha_vector.ndim == 1
        if is_1d:
            alpha_vector = alpha_vector.reshape(1, -1)

        scaled = self.scaler.transform(alpha_vector)
        compressed = self.pca.transform(scaled)

        return compressed.squeeze() if is_1d else compressed


# ========== Standalone Testing ==========

if __name__ == '__main__':
    # Generate synthetic alpha data
    np.random.seed(42)

    n_samples = 500
    n_alphas = 300

    # Create correlated alphas (realistic scenario)
    base_factors = np.random.randn(n_samples, 20)
    loadings = np.random.randn(20, n_alphas)
    noise = np.random.randn(n_samples, n_alphas) * 0.5

    alpha_matrix = base_factors @ loadings + noise

    # Test PCA Compressor
    compressor = AlphaPCACompressor(n_components=20)

    # Fit and transform
    compressed = compressor.fit_transform(alpha_matrix)

    print(f"Original shape: {alpha_matrix.shape}")
    print(f"Compressed shape: {compressed.shape}")
    print(f"Cumulative variance explained: {compressor.get_cumulative_variance():.2%}")
    print(f"\nVariance by component:")
    for i, var in enumerate(compressor.get_explained_variance_ratio()[:5]):
        print(f"  PC{i+1}: {var:.2%}")

    # Test top contributors
    print(f"\nTop 5 contributors to PC1:")
    print(compressor.get_top_contributors(1, n_top=5))
