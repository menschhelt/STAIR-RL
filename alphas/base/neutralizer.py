"""
Improved Neutralization System
Correct order: Mean Neutralization → Factor Neutralization → Normalize → Decay

TWO NEUTRALIZATION METHODS:
============================

(A) Priced Factor Removal (현재 기본 구현):
    ̃s_i,t = s_i,t - β̂_i,t^T λ̂_t

    - "팩터로 예상되는 수익분"을 빼서 신호 정정
    - β̂, λ̂를 어떤 방식(OLS/Ridge/Lasso/KF)으로 추정했든 계산 가능
    - 경제적 해석이 직관적 (가격되는 요인 기여분 제거)
    - 단점: β̂, λ̂의 추정오차가 남음 (완전 직교 아님)

    구현: epsilon = alpha - B @ factor_exposures

(B) Joint Projection (직교 중립화):
    ̃s_t = (I - P_X)s_t,  P_X = X(X^T W X)^(-1) X^T W

    - 신호를 노출행렬 X (β̂_t, 섹터, 스타일 로딩)의 부분공간에 정직교
    - X^T W ̃s_t = 0 이 수학적으로 보장 (완전 중립화)
    - OLS/WLS로 정확히 구현 (가중 직교 투영)
    - Ridge: 수축 투영 (완전 직교는 아니지만 수치 안정성 ↑)
    - Lasso: 선택된 변수로 post-OLS 재중립화 필요

    구현: epsilon = alpha - X @ (X^T W X)^(-1) @ X^T @ W @ alpha
         = (I - P_X) @ alpha

권장 사용:
- 엄격한 중립화 필요: (B) + OLS/WLS
- 경제적 해석 중요: (A)
- 실무 조합: (A) 설명 + (B) 최종 거래 신호
"""

from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from scipy import linalg
from .base import BaseNeutralizer

# Factor calculator is now in core/ directory, not here
# We don't need to import it in neutralizer since factors are passed in

# Try to import sklearn for Lasso/Ridge
try:
    from sklearn.linear_model import Ridge, Lasso, LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available, only OLS regression will work")


class NoneNeutralizer(BaseNeutralizer):
    """No neutralization - passthrough"""

    def neutralize(self, alpha_dict: Dict[str, pd.Series],
                  loading_matrix: Optional[Dict] = None) -> Dict[str, pd.Series]:
        """Return alpha as-is without neutralization"""
        return alpha_dict


class HierarchicalMeanNeutralizer(BaseNeutralizer):
    """
    Hierarchical mean neutralization using 21Shares GCCS classification
    Levels: market, level_1, level_2a, level_2b, level_3
    """

    def __init__(self, classification_data: Optional[pd.DataFrame] = None, level: str = 'market'):
        """
        Args:
            classification_data: DataFrame with symbol classifications (index=symbol)
                                 columns: level_1, level_2a, level_2b, level_3
            level: Neutralization level ('market', 'level_1', 'level_2a', 'level_2b', 'level_3')
        """
        self.classification_data = classification_data
        self.level = level

    def neutralize(self, alpha_dict: Dict[str, pd.Series],
                  loading_matrix: Optional[Dict] = None) -> Dict[str, pd.Series]:
        """
        Mean-neutralize within sectors at specified hierarchical level
        """
        if self.level == 'market' or self.classification_data is None:
            # Simple market neutralization (subtract overall mean)
            return self._market_neutralize(alpha_dict)

        # Sector-based neutralization
        return self._sector_neutralize(alpha_dict, self.level)

    def _market_neutralize(self, alpha_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Market neutralization: subtract overall mean"""
        result = {}
        all_pairs = list(alpha_dict.keys())

        if not all_pairs:
            return alpha_dict

        # Skip neutralization if universe is too small (< 3 pairs)
        if len(all_pairs) < 3:
            return alpha_dict

        # Common index
        common_index = alpha_dict[all_pairs[0]].index
        for pair in all_pairs[1:]:
            common_index = common_index.intersection(alpha_dict[pair].index)

        # For each timestamp, subtract mean
        for timestamp in common_index:
            cross_section_values = []
            for pair in all_pairs:
                if timestamp in alpha_dict[pair].index:
                    cross_section_values.append(alpha_dict[pair].loc[timestamp])

            if cross_section_values:
                universe_mean = np.mean(cross_section_values)

                for pair in all_pairs:
                    if pair not in result:
                        result[pair] = alpha_dict[pair].copy()
                    if timestamp in result[pair].index:
                        result[pair].loc[timestamp] -= universe_mean

        return result

    def _sector_neutralize(self, alpha_dict: Dict[str, pd.Series], level: str) -> Dict[str, pd.Series]:
        """Sector-based mean neutralization at specified level"""
        result = {pair: alpha_dict[pair].copy() for pair in alpha_dict}
        all_pairs = list(alpha_dict.keys())

        if not all_pairs:
            return alpha_dict

        # Common index
        common_index = alpha_dict[all_pairs[0]].index
        for pair in all_pairs[1:]:
            common_index = common_index.intersection(alpha_dict[pair].index)

        # For each timestamp
        for timestamp in common_index:
            # Group pairs by sector
            sector_groups = {}
            for pair in all_pairs:
                if timestamp not in alpha_dict[pair].index:
                    continue

                # Get symbol from pair (e.g., "BTC/USDT:USDT" -> "BTC")
                symbol = pair.split('/')[0]

                # Get sector classification
                if symbol in self.classification_data.index and level in self.classification_data.columns:
                    sector = self.classification_data.loc[symbol, level]
                    if sector not in sector_groups:
                        sector_groups[sector] = []
                    sector_groups[sector].append(pair)

            # Neutralize within each sector
            for sector, pairs_in_sector in sector_groups.items():
                if len(pairs_in_sector) > 0:
                    sector_values = [alpha_dict[p].loc[timestamp] for p in pairs_in_sector]
                    sector_mean = np.mean(sector_values)

                    for pair in pairs_in_sector:
                        result[pair].loc[timestamp] -= sector_mean

        return result


class FactorNeutralizer(BaseNeutralizer):
    """
    Factor-based OLS neutralization (WITHOUT built-in normalization)
    Returns residuals from regression: epsilon = alpha - B*x

    Supports 400+ factors via FactorManager
    """

    def __init__(self,
                 factor_list: Optional[List[str]] = None,
                 factor_groups: Optional[List[str]] = None,
                 regression_method: str = "ols",
                 regularization: float = 1e-6,
                 neutralization_mode: str = "priced_factor"):
        """
        Args:
            factor_list: Specific factors to use (None = use all from loading_matrix)
            factor_groups: Factor groups hint for documentation ['candle', 'sector', 'macro']
            regression_method: Regression method ('ols', 'lasso', 'ridge')
            regularization: Regularization parameter for Lasso/Ridge
            neutralization_mode: Neutralization method:
                - 'priced_factor' (A): ̃s = s - β̂^T λ̂  (경제적 해석, 추정오차 존재)
                - 'joint_projection' (B): ̃s = (I - P_X)s  (정직교 중립화, 완전한 중립화)

        Note: Factors are now calculated by FactorManager in core/, not here.
              This neutralizer only uses the pre-computed loading_matrix passed to neutralize().
        """
        self.factor_list = factor_list  # None = all factors
        self.factor_groups = factor_groups or ['candle', 'macro']  # For documentation only
        self.regression_method = regression_method
        self.regularization = regularization
        self.neutralization_mode = neutralization_mode  # 'priced_factor' or 'joint_projection'
        self.factor_calculator = None  # Will be set externally if needed

    def neutralize(self, alpha_dict: Dict[str, pd.Series],
                  loading_matrix: Optional[Dict] = None) -> Dict[str, pd.Series]:
        """
        Factor neutralization: regress out systematic factors

        Returns RAW RESIDUALS (not normalized) for downstream normalization
        """
        if loading_matrix is None:
            print("⚠️ No factor loading data provided for factor neutralization")
            return alpha_dict

        result = {}
        all_pairs = list(alpha_dict.keys())

        if len(all_pairs) < 2:
            return alpha_dict

        # Common index
        common_index = alpha_dict[all_pairs[0]].index
        for pair in all_pairs[1:]:
            common_index = common_index.intersection(alpha_dict[pair].index)

        if len(common_index) == 0:
            return alpha_dict

        # Initialize result
        for pair in all_pairs:
            result[pair] = pd.Series(index=common_index, dtype=float)

        # Build factor loadings matrix
        factor_loadings_dict = self._build_factor_loadings_matrix(loading_matrix, all_pairs, common_index)

        # Neutralize each timestamp
        for timestamp in common_index:
            epsilon_dict = self._neutralize_cross_section(alpha_dict, factor_loadings_dict, timestamp, all_pairs)

            if epsilon_dict:
                for pair, value in epsilon_dict.items():
                    result[pair].loc[timestamp] = value

        return result

    def _build_factor_loadings_matrix(self, loading_matrix: Dict, pairs: list, timestamps: pd.Index) -> Dict:
        """
        Build factor loadings matrix from pre-computed loading_matrix.

        Note: loading_matrix is already computed by FactorManager with all factor loadings.
              We just need to extract and align the data.
        """
        factor_loadings_dict = {}

        for pair in pairs:
            if pair not in loading_matrix:
                continue

            dataframe = loading_matrix[pair]

            if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
                continue

            # loading_matrix already contains factor loadings computed by FactorManager
            # Just reindex to match timestamps
            try:
                # Select factors if factor_list is specified
                if self.factor_list:
                    # Only use specified factors
                    available_factors = [f for f in self.factor_list if f in dataframe.columns]
                    if available_factors:
                        factor_loadings = dataframe[available_factors]
                    else:
                        # No matching factors, skip this pair
                        continue
                else:
                    # Use all available columns as factors
                    factor_loadings = dataframe

                factor_loadings_dict[pair] = factor_loadings.reindex(timestamps).fillna(0)

            except Exception as e:
                print(f"⚠️ Failed to process factor loadings for {pair}: {e}")
                continue

        return factor_loadings_dict

    def _neutralize_cross_section(self, alpha_dict: Dict, factor_loadings_dict: Dict,
                                 timestamp, pairs: list) -> Optional[Dict]:
        """Cross-sectional factor neutralization for specific timestamp"""

        # Collect alpha values and factor loadings
        alpha_values = []
        factor_matrix_rows = []
        valid_pairs = []

        for pair in pairs:
            alpha_available = (pair in alpha_dict and
                             timestamp in alpha_dict[pair].index and
                             not pd.isna(alpha_dict[pair].loc[timestamp]))

            factor_available = (pair in factor_loadings_dict and
                              timestamp in factor_loadings_dict[pair].index)

            if alpha_available and factor_available:
                alpha_val = alpha_dict[pair].loc[timestamp]
                factor_row = factor_loadings_dict[pair].loc[timestamp].values

                # Convert to float array to avoid dtype issues
                try:
                    factor_row = np.asarray(factor_row, dtype=float)
                    if not (np.isnan(alpha_val) or np.any(np.isnan(factor_row)) or np.any(np.isinf(factor_row))):
                        alpha_values.append(alpha_val)
                        factor_matrix_rows.append(factor_row)
                        valid_pairs.append(pair)
                except (ValueError, TypeError):
                    # Skip pairs with non-numeric factor data
                    continue

        if len(valid_pairs) < 3:
            return None

        # Prepare data for regression
        alpha_vector = np.array(alpha_values)  # N x 1
        B_matrix = np.array(factor_matrix_rows)  # N x K

        try:
            # Select regression method
            if self.regression_method == "lasso" and SKLEARN_AVAILABLE:
                # Lasso regression (L1 regularization)
                model = Lasso(alpha=self.regularization, fit_intercept=False, max_iter=10000)
                model.fit(B_matrix, alpha_vector)
                factor_exposures = model.coef_

            elif self.regression_method == "ridge" and SKLEARN_AVAILABLE:
                # Ridge regression (L2 regularization)
                model = Ridge(alpha=self.regularization, fit_intercept=False)
                model.fit(B_matrix, alpha_vector)
                factor_exposures = model.coef_

            else:
                # OLS regression (default or fallback)
                if self.regression_method != "ols" and not SKLEARN_AVAILABLE:
                    print(f"⚠️ sklearn not available, falling back to OLS (requested: {self.regression_method})")

                # x = (B'B)^(-1) B' alpha
                BtB = B_matrix.T @ B_matrix
                Bt_alpha = B_matrix.T @ alpha_vector

                # Regularization for numerical stability
                regularization = self.regularization * np.eye(BtB.shape[0])
                BtB_reg = BtB + regularization

                try:
                    factor_exposures = linalg.solve(BtB_reg, Bt_alpha)
                except linalg.LinAlgError:
                    factor_exposures = np.linalg.pinv(BtB_reg) @ Bt_alpha

            # ===== METHOD A vs B =====
            if self.neutralization_mode == "joint_projection":
                # (B) Joint Projection: ̃s = (I - P_X)s
                # P_X = X(X^T W X)^(-1) X^T W
                # For OLS: W = I, so P_X = X(X^T X)^(-1) X^T
                # epsilon = alpha - X @ (X^T X)^(-1) @ X^T @ alpha
                #         = (I - P_X) @ alpha

                # Build projection matrix
                try:
                    if self.regression_method == "ols":
                        # OLS: P_X = B(B^T B)^(-1) B^T
                        BtB = B_matrix.T @ B_matrix
                        regularization = self.regularization * np.eye(BtB.shape[0])
                        BtB_reg = BtB + regularization

                        try:
                            BtB_inv = linalg.inv(BtB_reg)
                        except linalg.LinAlgError:
                            BtB_inv = np.linalg.pinv(BtB_reg)

                        P_X = B_matrix @ BtB_inv @ B_matrix.T

                    elif self.regression_method == "ridge" and SKLEARN_AVAILABLE:
                        # Ridge: P_X with shrinkage
                        BtB = B_matrix.T @ B_matrix
                        regularization = self.regularization * np.eye(BtB.shape[0])
                        BtB_ridge = BtB + regularization

                        try:
                            BtB_inv = linalg.inv(BtB_ridge)
                        except linalg.LinAlgError:
                            BtB_inv = np.linalg.pinv(BtB_ridge)

                        P_X = B_matrix @ BtB_inv @ B_matrix.T

                    elif self.regression_method == "lasso" and SKLEARN_AVAILABLE:
                        # Lasso: post-OLS on selected variables
                        # Step 1: Lasso selection
                        model_lasso = Lasso(alpha=self.regularization, fit_intercept=False, max_iter=10000)
                        model_lasso.fit(B_matrix, alpha_vector)
                        selected_features = np.abs(model_lasso.coef_) > 1e-8

                        if selected_features.sum() == 0:
                            # No features selected, fallback to identity
                            epsilon = alpha_vector
                            epsilon_dict = {pair: epsilon[i] for i, pair in enumerate(valid_pairs)}
                            return epsilon_dict

                        # Step 2: post-OLS on selected features
                        B_selected = B_matrix[:, selected_features]
                        BtB_selected = B_selected.T @ B_selected
                        regularization_selected = self.regularization * np.eye(BtB_selected.shape[0])
                        BtB_selected_reg = BtB_selected + regularization_selected

                        try:
                            BtB_inv_selected = linalg.inv(BtB_selected_reg)
                        except linalg.LinAlgError:
                            BtB_inv_selected = np.linalg.pinv(BtB_selected_reg)

                        P_X = B_selected @ BtB_inv_selected @ B_selected.T

                    else:
                        # Fallback to OLS
                        BtB = B_matrix.T @ B_matrix
                        regularization = self.regularization * np.eye(BtB.shape[0])
                        BtB_reg = BtB + regularization

                        try:
                            BtB_inv = linalg.inv(BtB_reg)
                        except linalg.LinAlgError:
                            BtB_inv = np.linalg.pinv(BtB_reg)

                        P_X = B_matrix @ BtB_inv @ B_matrix.T

                    # Apply projection: epsilon = (I - P_X) @ alpha
                    I = np.eye(len(alpha_vector))
                    epsilon = (I - P_X) @ alpha_vector

                    # Verify orthogonality: X^T @ epsilon should be ~0
                    orthogonality_check = np.linalg.norm(B_matrix.T @ epsilon)
                    if orthogonality_check > 0.01:  # Threshold for numerical stability
                        print(f"⚠️ Orthogonality check failed at {timestamp}: ||X^T epsilon|| = {orthogonality_check:.6f}")

                except Exception as e:
                    print(f"⚠️ Joint projection failed at {timestamp}, falling back to Method A: {e}")
                    # Fallback to Method A
                    predicted_alpha = B_matrix @ factor_exposures
                    epsilon = alpha_vector - predicted_alpha

            else:
                # (A) Priced Factor Removal: ̃s = s - β̂^T λ̂
                # epsilon = alpha - B @ factor_exposures
                predicted_alpha = B_matrix @ factor_exposures
                epsilon = alpha_vector - predicted_alpha

            # Return RAW residuals (no normalization here!)
            epsilon_dict = {pair: epsilon[i] for i, pair in enumerate(valid_pairs)}

            return epsilon_dict

        except Exception as e:
            print(f"⚠️ {self.regression_method.upper()} neutralization failed at {timestamp}: {e}")
            return None


class CombinedNeutralizer(BaseNeutralizer):
    """
    Combined neutralizer that applies both mean and factor neutralization
    Order: 1. Mean Neutralization → 2. Factor Neutralization
    """

    def __init__(
        self,
        mean_neutralizer: Optional[HierarchicalMeanNeutralizer] = None,
        factor_neutralizer: Optional[FactorNeutralizer] = None,
        apply_mean: bool = True,
        apply_factor: bool = False
    ):
        self.mean_neutralizer = mean_neutralizer
        self.factor_neutralizer = factor_neutralizer
        self.apply_mean = apply_mean
        self.apply_factor = apply_factor

    def neutralize(self, alpha_dict: Dict[str, pd.Series],
                  loading_matrix: Optional[Dict] = None) -> Dict[str, pd.Series]:
        """
        Apply neutralization in correct order:
        1. Mean neutralization
        2. Factor neutralization
        """
        result = alpha_dict

        # Step 1: Mean neutralization
        if self.apply_mean and self.mean_neutralizer is not None:
            result = self.mean_neutralizer.neutralize(result, loading_matrix)

        # Step 2: Factor neutralization
        if self.apply_factor and self.factor_neutralizer is not None:
            result = self.factor_neutralizer.neutralize(result, loading_matrix)

        return result
