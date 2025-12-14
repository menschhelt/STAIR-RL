"""
Loading Matrix Calculator
=========================
Calculate factor loadings (exposures) for all assets using regression.

KEY DISTINCTION:
- Loading Matrix B: GLOBAL calculation (one regression per asset across ALL factors)
- Alpha Neutralization: PER-ALPHA operation (uses pre-computed B)

Regression Methods:
- OLS: Fast, interpretable
- Ridge: Handles multicollinearity
- Lasso: Feature selection

Adapted for stair-local Parquet-based data system.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging


class LoadingMatrixCalculator:
    """
    Calculate factor loadings for all assets in universe.

    Process:
    1. For each asset i:
       R_i = B_i * F + epsilon_i
       where R_i = asset returns (T x 1)
             B_i = factor loadings (K x 1)
             F = factor returns (T x K)

    2. Estimate B_i using OLS/Ridge/Lasso
    3. Cache result (update frequency: ~5 hours)
    """

    def __init__(
        self,
        regression_method: str = 'ols',
        regularization: float = 1e-6,
        lookback_period: int = 60,  # days
        min_observations: int = 30,
        update_frequency_minutes: int = 300  # 5 hours default
    ):
        """
        Args:
            regression_method: 'ols', 'ridge', or 'lasso'
            regularization: lambda for Ridge/Lasso, numerical stability for OLS
            lookback_period: Days of history for regression
            min_observations: Minimum data points required
            update_frequency_minutes: Recalculation interval (default: 300 = 5 hours)
        """
        self.regression_method = regression_method
        self.regularization = regularization
        self.lookback_period = lookback_period
        self.min_observations = min_observations
        self.update_frequency_minutes = update_frequency_minutes

        # Cache
        self.loading_matrix_cache: Dict[pd.Timestamp, Dict[str, Dict[str, float]]] = {}

        # Setup logging
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

    def _round_to_update_frequency(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        """
        Round timestamp to update_frequency boundary.

        Example: update_frequency = 300 minutes (5 hours)
        - 00:00 - 04:59 -> 00:00
        - 05:00 - 09:59 -> 05:00
        - 10:00 - 14:59 -> 10:00
        """
        total_minutes = timestamp.hour * 60 + timestamp.minute
        rounded_minutes = (total_minutes // self.update_frequency_minutes) * self.update_frequency_minutes

        return timestamp.replace(
            hour=rounded_minutes // 60,
            minute=rounded_minutes % 60,
            second=0,
            microsecond=0
        )

    def calculate_loading_matrix(
        self,
        asset_returns: Dict[str, pd.Series],  # {symbol: return_series}
        factor_returns: pd.DataFrame,          # DataFrame with factor columns
        timestamp: pd.Timestamp
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate loading matrix B for all assets.

        Args:
            asset_returns: {symbol: return series} - typically daily/hourly returns
            factor_returns: DataFrame with columns = factor names, index = timestamps
            timestamp: Current timestamp for caching

        Returns:
            {
                'BTCUSDT': {
                    'CMKT': 0.98,
                    'CSMB': 0.15,
                    'CMOM': -0.05,
                    'equities': 0.32,
                    ...
                },
                ...
            }
        """
        # Round timestamp to update_frequency boundary
        cache_key = self._round_to_update_frequency(timestamp)

        # Check cache
        if cache_key in self.loading_matrix_cache:
            self.logger.debug(f"Using cached loading matrix (key={cache_key}, requested={timestamp})")
            return self.loading_matrix_cache[cache_key]

        self.logger.info(
            f"Calculating loading matrix: {len(asset_returns)} assets, "
            f"{len(factor_returns.columns)} factors, "
            f"method={self.regression_method.upper()}"
        )
        self.logger.info(f"Factors in loading matrix: {list(factor_returns.columns)}")

        loading_matrix = {}

        for symbol, returns in asset_returns.items():
            try:
                loadings = self._calculate_loadings_for_asset(
                    returns, factor_returns, symbol
                )

                if loadings:
                    loading_matrix[symbol] = loadings

            except Exception as e:
                self.logger.warning(f"Failed to calculate loadings for {symbol}: {e}")
                continue

        # Cache result with rounded key
        self.loading_matrix_cache[cache_key] = loading_matrix

        self.logger.info(
            f"Calculated loadings for {len(loading_matrix)}/{len(asset_returns)} assets "
            f"(cached at {cache_key})"
        )

        return loading_matrix

    def _calculate_loadings_for_asset(
        self,
        asset_returns: pd.Series,
        factor_returns: pd.DataFrame,
        symbol: str
    ) -> Optional[Dict[str, float]]:
        """Calculate factor loadings for single asset."""

        # Get overlapping time period
        overlap_idx = asset_returns.index.intersection(factor_returns.index)

        if len(overlap_idx) < self.min_observations:
            self.logger.debug(
                f"{symbol}: Insufficient overlap ({len(overlap_idx)} < {self.min_observations})"
            )
            return None

        # Slice to lookback period
        lookback_start = overlap_idx[-self.lookback_period] if len(overlap_idx) > self.lookback_period else overlap_idx[0]
        lookback_idx = overlap_idx[overlap_idx >= lookback_start]

        Y = asset_returns.loc[lookback_idx].values
        X = factor_returns.loc[lookback_idx].values

        # Remove NaN
        valid_mask = ~(np.isnan(Y) | np.any(np.isnan(X), axis=1))
        Y = Y[valid_mask]
        X = X[valid_mask]

        if len(Y) < self.min_observations:
            self.logger.debug(f"{symbol}: Too few valid observations after NaN removal ({len(Y)})")
            return None

        # Run regression
        if self.regression_method == 'ols':
            beta = self._ols_regression(X, Y)
        elif self.regression_method == 'ridge':
            beta = self._ridge_regression(X, Y)
        elif self.regression_method == 'lasso':
            beta = self._lasso_regression(X, Y)
        else:
            raise ValueError(f"Unknown regression method: {self.regression_method}")

        # Convert to dict
        loadings = dict(zip(factor_returns.columns, beta))

        return loadings

    def _ols_regression(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        OLS: beta = (X^T X)^-1 X^T Y

        With regularization for numerical stability:
        beta = (X^T X + lambda*I)^-1 X^T Y
        """
        XtX = X.T @ X + self.regularization * np.eye(X.shape[1])
        XtY = X.T @ Y

        try:
            beta = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            self.logger.warning("OLS singular matrix, using lstsq")
            beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

        return beta

    def _ridge_regression(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Ridge: beta = (X^T X + lambda*I)^-1 X^T Y

        lambda controls regularization strength.
        """
        XtX = X.T @ X + self.regularization * np.eye(X.shape[1])
        XtY = X.T @ Y

        beta = np.linalg.solve(XtX, XtY)

        return beta

    def _lasso_regression(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Lasso: argmin ||Y - X*beta||^2 + lambda*||beta||_1

        Requires sklearn. Slower but provides feature selection.
        """
        try:
            from sklearn.linear_model import Lasso

            # Normalize X to avoid scale issues with L1 penalty
            X_std = X.std(axis=0)
            X_std[X_std < 1e-8] = 1.0  # Avoid division by zero
            X_normalized = X / X_std

            model = Lasso(alpha=self.regularization, fit_intercept=False, max_iter=5000)
            model.fit(X_normalized, Y)

            # Scale coefficients back
            beta = model.coef_ / X_std

            return beta

        except ImportError:
            self.logger.error("sklearn not installed, falling back to OLS")
            return self._ols_regression(X, Y)

    def get_loading_for_symbol(
        self,
        symbol: str,
        timestamp: pd.Timestamp
    ) -> Optional[Dict[str, float]]:
        """
        Get cached loading for specific symbol.

        Returns:
            {factor_name: loading} or None if not available
        """
        cache_key = self._round_to_update_frequency(timestamp)
        if cache_key in self.loading_matrix_cache:
            return self.loading_matrix_cache[cache_key].get(symbol)
        return None

    def calculate_residual_alpha(
        self,
        asset_return: float,
        loadings: Dict[str, float],
        factor_returns: Dict[str, float]
    ) -> float:
        """
        Calculate residual alpha (unexplained return).

        Formula: alpha_resid = R_i - sum(B_ij * F_j)

        Args:
            asset_return: Asset's actual return
            loadings: {factor: loading} for this asset
            factor_returns: {factor: return} for current period

        Returns:
            Residual alpha (unexplained portion of return)
        """
        expected_return = 0.0
        for factor, loading in loadings.items():
            if factor in factor_returns:
                expected_return += loading * factor_returns[factor]

        return asset_return - expected_return

    def cleanup_old_cache(self, current_timestamp: pd.Timestamp, max_age_days: int = 7):
        """Remove cache entries older than max_age_days."""
        from datetime import timedelta

        max_age = timedelta(days=max_age_days)

        keys_to_remove = []
        for cache_timestamp in self.loading_matrix_cache.keys():
            if (current_timestamp - cache_timestamp) > max_age:
                keys_to_remove.append(cache_timestamp)

        for key in keys_to_remove:
            del self.loading_matrix_cache[key]

        if keys_to_remove:
            self.logger.debug(f"Cleaned up {len(keys_to_remove)} old loading matrix cache entries")


# ========== Standalone Testing ==========

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from datetime import datetime, timezone, timedelta

    # Generate sample data
    np.random.seed(42)

    # Create sample factor returns (60 days, 3 factors)
    dates = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        periods=60,
        freq='D'
    )

    factor_returns = pd.DataFrame({
        'CMKT': np.random.randn(60) * 0.02,
        'CSMB': np.random.randn(60) * 0.01,
        'CMOM': np.random.randn(60) * 0.015,
    }, index=dates)

    # Create sample asset returns (correlated with factors)
    asset_returns = {
        'BTCUSDT': pd.Series(
            0.9 * factor_returns['CMKT'] + 0.3 * factor_returns['CSMB'] + np.random.randn(60) * 0.005,
            index=dates
        ),
        'ETHUSDT': pd.Series(
            1.1 * factor_returns['CMKT'] + 0.5 * factor_returns['CMOM'] + np.random.randn(60) * 0.007,
            index=dates
        ),
    }

    # Test loading matrix calculation
    calculator = LoadingMatrixCalculator(
        regression_method='ols',
        lookback_period=60,
        min_observations=30
    )

    timestamp = dates[-1]
    loadings = calculator.calculate_loading_matrix(asset_returns, factor_returns, timestamp)

    print("\n" + "=" * 60)
    print("LOADING MATRIX TEST")
    print("=" * 60)

    for symbol, beta in loadings.items():
        print(f"\n{symbol}:")
        for factor, loading in beta.items():
            print(f"  {factor}: {loading:.4f}")

    # Test residual alpha calculation
    print("\n" + "=" * 60)
    print("RESIDUAL ALPHA TEST")
    print("=" * 60)

    current_factor_returns = factor_returns.iloc[-1].to_dict()
    for symbol, returns in asset_returns.items():
        actual_return = returns.iloc[-1]
        alpha = calculator.calculate_residual_alpha(
            actual_return,
            loadings[symbol],
            current_factor_returns
        )
        print(f"{symbol}: actual={actual_return:.4f}, alpha={alpha:.4f}")
