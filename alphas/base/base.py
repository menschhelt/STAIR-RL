
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Optional, Dict, List, Union
import importlib
from abc import ABC, abstractmethod
from scipy import stats

# Type alias for data that can be Series or DataFrame
DataType = Union[pd.Series, pd.DataFrame]


class BaseAlpha(ABC):
    """알파 팩터 베이스 클래스 - Wide DataFrame (cross-sectional) 지원"""

    # 각 알파별 처리 설정 (하위 클래스에서 오버라이드 가능)
    neutralizer_type: str = "mean"  # "none", "mean", "factor"
    decay_period: int = 3  # 선형 디케이 기간 (일수)

    # 기본 파라미터 정의 (하위 클래스에서 오버라이드)
    default_params: Dict[str, int] = {}

    def __init__(self, params: Dict[str, int] = None):
        """
        Args:
            params: Config에서 전달된 파라미터 (optional)
        """
        self.params = {**self.default_params, **(params or {})}

    @abstractmethod
    def calculate(self, data: Union[DataFrame, Dict[str, DataFrame]], pair: str = None) -> DataType:
        """
        Raw 알파 계산

        Args:
            data: OHLCV DataFrame (per-symbol) 또는 Dict[str, DataFrame] (wide format)
            pair: 페어명 (wide format에서는 None)

        Returns:
            pd.Series (per-symbol) 또는 pd.DataFrame (wide format)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """알파 이름"""
        pass

    @property
    def required_warmup_candles(self) -> int:
        if not self.params:
            return 200
        return max(self.params.values()) if self.params else 200

    # ========== 시계열 연산자 (ts_) - DataFrame 지원 ==========

    def ts_delta(self, data: DataType, period: int = 1) -> DataType:
        """시계열 차분 (lag difference)"""
        return data.diff(periods=period)

    def ts_delay(self, data: DataType, period: int) -> DataType:
        """지연 (lag) - ts_delta와 구분"""
        return data.shift(period)

    def ts_rank(self, data: DataType, window: int) -> DataType:
        """시계열 롤링 랭킹 (0~1 사이 값) - 시간축 기준"""
        return data.rolling(window).rank(pct=True)

    def ts_sum(self, data: DataType, window: int) -> DataType:
        """시계열 롤링 합계"""
        return data.rolling(window).sum()

    def ts_mean(self, data: DataType, window: int) -> DataType:
        """시계열 롤링 평균"""
        return data.rolling(window).mean()

    def ts_min(self, data: DataType, window: int) -> DataType:
        """시계열 롤링 최소값"""
        return data.rolling(window).min()

    def ts_max(self, data: DataType, window: int) -> DataType:
        """시계열 롤링 최대값"""
        return data.rolling(window).max()

    def ts_std(self, data: DataType, window: int) -> DataType:
        """시계열 롤링 표준편차"""
        return data.rolling(window).std()

    def ts_corr(self, x: DataType, y: DataType, window: int) -> DataType:
        """시계열 롤링 상관계수"""
        result = x.rolling(window).corr(y)
        return result.replace([np.inf, -np.inf], 0)

    def ts_cov(self, x: DataType, y: DataType, window: int) -> DataType:
        """시계열 롤링 공분산"""
        result = x.rolling(window).cov(y)
        return result.replace([np.inf, -np.inf], 0)

    def ts_argmax(self, data: DataType, window: int) -> DataType:
        """시계열 롤링 최대값 인덱스"""
        if isinstance(data, pd.DataFrame):
            return data.rolling(window).apply(lambda x: np.argmax(x), raw=True)
        # Series - optimized version
        result = np.full(len(data), np.nan)
        values = data.values
        for i in range(window - 1, len(data)):
            result[i] = np.argmax(values[i - window + 1 : i + 1])
        return pd.Series(result, index=data.index)

    def ts_argmin(self, data: DataType, window: int) -> DataType:
        """시계열 롤링 최소값 인덱스"""
        if isinstance(data, pd.DataFrame):
            return data.rolling(window).apply(lambda x: np.argmin(x), raw=True)
        result = np.full(len(data), np.nan)
        values = data.values
        for i in range(window - 1, len(data)):
            result[i] = np.argmin(values[i - window + 1 : i + 1])
        return pd.Series(result, index=data.index)

    def ts_product(self, data: DataType, window: int) -> DataType:
        """시계열 롤링 곱"""
        return data.rolling(window).apply(np.prod, raw=True)

    def ts_decayed_linear(self, data: DataType, window: int) -> DataType:
        """선형 감쇠 가중평균"""
        weights = np.arange(1, window + 1, dtype=float)
        weights = weights / weights.sum()
        return data.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)

    def ts_zscore(self, data: DataType, window: int) -> DataType:
        """시계열 롤링 z-score"""
        mean = data.rolling(window).mean()
        std = data.rolling(window).std()
        return (data - mean) / std.replace(0, np.nan)

    def ts_scale(self, data: DataType, window: int, constant: float = 0) -> DataType:
        """시계열 롤링 0~1 스케일"""
        min_val = data.rolling(window).min()
        max_val = data.rolling(window).max()
        range_val = max_val - min_val
        return (data - min_val) / range_val.replace(0, np.nan) + constant

    def ts_quantile(self, data: DataType, window: int, q: float = 0.5) -> DataType:
        """시계열 롤링 분위수"""
        return data.rolling(window).quantile(q)

    # ========== 크로스섹션 연산자 (axis=1) ==========

    def rank(self, data: DataType) -> DataType:
        """
        크로스섹션 랭킹 (0~1 사이 값)
        - DataFrame: 각 행(timestamp)에서 열(symbol)간 순위
        - Series: 전체 값 내 순위
        """
        if isinstance(data, pd.DataFrame):
            return data.rank(axis=1, pct=True)
        return data.rank(pct=True)

    def normalize(self, data: DataType, use_std: bool = False) -> DataType:
        """크로스섹션 정규화 (평균 제거, 선택적 std 나눔)"""
        if isinstance(data, pd.DataFrame):
            mean = data.mean(axis=1)
            result = data.sub(mean, axis=0)
            if use_std:
                std = data.std(axis=1).replace(0, np.nan)
                result = result.div(std, axis=0)
            return result
        mean = data.mean()
        result = data - mean
        if use_std:
            std = data.std()
            if std > 0:
                result = result / std
        return result

    def scale(self, data: DataType, scale_val: float = 1) -> DataType:
        """크로스섹션 스케일 (절대값 합이 scale_val이 되도록)"""
        if isinstance(data, pd.DataFrame):
            abs_sum = data.abs().sum(axis=1).replace(0, np.nan)
            return data.div(abs_sum, axis=0) * scale_val
        abs_sum = data.abs().sum()
        if abs_sum > 0:
            return data / abs_sum * scale_val
        return data * 0

    def scale_down(self, data: DataType, constant: float = 0) -> DataType:
        """크로스섹션 0~1 스케일"""
        if isinstance(data, pd.DataFrame):
            min_val = data.min(axis=1)
            max_val = data.max(axis=1)
            range_val = (max_val - min_val).replace(0, np.nan)
            return data.sub(min_val, axis=0).div(range_val, axis=0) - constant
        min_val, max_val = data.min(), data.max()
        range_val = max_val - min_val
        if range_val > 0:
            return (data - min_val) / range_val - constant
        return data * 0

    def winsorize(self, data: DataType, std_mult: float = 4) -> DataType:
        """크로스섹션 윈저라이징 (이상치 처리)"""
        if isinstance(data, pd.DataFrame):
            mean = data.mean(axis=1)
            std = data.std(axis=1)
            lower = mean - std_mult * std
            upper = mean + std_mult * std
            # DataFrame clip with axis
            result = data.copy()
            for i in range(len(data)):
                result.iloc[i] = data.iloc[i].clip(lower=lower.iloc[i], upper=upper.iloc[i])
            return result
        mean, std = data.mean(), data.std()
        return data.clip(lower=mean - std_mult * std, upper=mean + std_mult * std)

    def zscore(self, data: DataType) -> DataType:
        """크로스섹션 z-score"""
        if isinstance(data, pd.DataFrame):
            mean = data.mean(axis=1)
            std = data.std(axis=1).replace(0, np.nan)
            return data.sub(mean, axis=0).div(std, axis=0)
        mean, std = data.mean(), data.std()
        if std > 0:
            return (data - mean) / std
        return data * 0

    def quantile_transform(self, data: DataType, driver: str = 'gaussian') -> DataType:
        """크로스섹션 분위수 변환"""
        if isinstance(data, pd.DataFrame):
            ranked = data.rank(axis=1, pct=True)
            if driver == 'gaussian':
                # Clip to avoid inf at 0 and 1
                return ranked.clip(0.001, 0.999).apply(lambda x: stats.norm.ppf(x))
            return ranked
        ranked = data.rank(pct=True).clip(0.001, 0.999)
        if driver == 'gaussian':
            return pd.Series(stats.norm.ppf(ranked.values), index=data.index)
        return ranked

    # ========== 조건/논리 연산자 ==========

    def condition(self, cond: DataType, true_val, false_val) -> DataType:
        """조건부 연산 (if-then-else)"""
        if isinstance(cond, pd.DataFrame):
            if isinstance(true_val, (int, float)):
                true_val = cond * 0 + true_val
            if isinstance(false_val, (int, float)):
                false_val = cond * 0 + false_val
        else:
            if isinstance(true_val, (int, float)):
                true_val = pd.Series(true_val, index=cond.index)
            if isinstance(false_val, (int, float)):
                false_val = pd.Series(false_val, index=cond.index)
        return true_val.where(cond, false_val)

    def sign(self, data: DataType) -> DataType:
        """부호 함수"""
        return np.sign(data)

    def delay(self, data: DataType, period: int) -> DataType:
        """지연 (lag)"""
        return data.shift(period)

    # ========== 기본 산술 연산자 ==========

    def add(self, x: DataType, y) -> DataType:
        """덧셈"""
        if isinstance(y, (int, float)):
            return x + y
        return x.add(y, fill_value=0)

    def sub(self, x: DataType, y) -> DataType:
        """뺄셈"""
        if isinstance(y, (int, float)):
            return x - y
        return x.sub(y, fill_value=0)

    def mul(self, x, y) -> DataType:
        """곱셈"""
        if isinstance(y, (int, float)):
            return x * y
        if isinstance(x, (int, float)):
            return y * x
        return x.mul(y, fill_value=0)

    def div(self, x, y) -> DataType:
        """나눗셈 (0으로 나누기 처리)"""
        if isinstance(y, (int, float)):
            if abs(y) < 1e-10:
                return x * 0
            return x / y
        if isinstance(x, (int, float)):
            result = x / y.replace(0, np.nan)
        else:
            result = x.div(y.replace(0, np.nan))
        return result.replace([np.inf, -np.inf], 0).fillna(0)

    def pow(self, x: DataType, exp: float) -> DataType:
        """거듭제곱"""
        result = np.power(x, exp)
        if isinstance(result, (pd.DataFrame, pd.Series)):
            return result.replace([np.inf, -np.inf], 0)
        return np.where(np.isinf(result), 0, result)

    def power(self, x: DataType, y) -> DataType:
        """거듭제곱 (alias)"""
        result = np.power(x, y)
        if isinstance(result, (pd.DataFrame, pd.Series)):
            return result.replace([np.inf, -np.inf], 0)
        return np.where(np.isinf(result), 0, result)

    def log(self, data: DataType) -> DataType:
        """자연로그"""
        return np.log(data.replace(0, np.nan)).fillna(0)

    def abs(self, data: DataType) -> DataType:
        """절댓값"""
        if isinstance(data, pd.DataFrame):
            return data.abs()
        return np.abs(data)

    def sqrt(self, data: DataType) -> DataType:
        """제곱근"""
        if isinstance(data, pd.DataFrame):
            return np.sqrt(data.clip(lower=0))
        return np.sqrt(np.maximum(data, 0))

    def neg(self, data: DataType) -> DataType:
        """음수 변환"""
        return -data

    # ========== 비교 연산자 ==========

    def lt(self, x: DataType, y) -> DataType:
        """미만"""
        return x < y

    def gt(self, x: DataType, y) -> DataType:
        """초과"""
        return x > y

    def le(self, x: DataType, y) -> DataType:
        """이하"""
        return x <= y

    def ge(self, x: DataType, y) -> DataType:
        """이상"""
        return x >= y

    def eq(self, x: DataType, y) -> DataType:
        """같음"""
        return x == y

    def ne(self, x: DataType, y) -> DataType:
        """다름"""
        return x != y

    def or_(self, x: DataType, y: DataType) -> DataType:
        """논리 OR"""
        return x | y

    def and_(self, x: DataType, y: DataType) -> DataType:
        """논리 AND"""
        return x & y

    def min(self, x: DataType, y) -> DataType:
        """element-wise 최소"""
        return np.minimum(x, y)

    def max(self, x: DataType, y) -> DataType:
        """element-wise 최대"""
        return np.maximum(x, y)

    # ========== 유틸리티 함수 ==========

    def sma(self, data: DataType, n: int, m: int = 1) -> DataType:
        """지수이동평균 (SMA style)"""
        return data.ewm(alpha=m/n, adjust=False).mean()

    def ema(self, data: DataType, span: int, alpha: float = None) -> DataType:
        """지수 이동 평균"""
        if alpha is not None:
            return data.ewm(alpha=alpha, adjust=False).mean()
        return data.ewm(span=span, adjust=False).mean()

    def ma(self, data: DataType, window: int) -> DataType:
        """단순 이동 평균"""
        return data.rolling(window).mean()

    def sequence(self, n: int) -> np.ndarray:
        """1~n 등차수열"""
        return np.arange(1, n+1)

    def returns(self, close: DataType) -> DataType:
        """수익률"""
        return close.pct_change(fill_method=None)

    def vwap_calc(self, high: DataType, low: DataType, close: DataType, volume: DataType) -> DataType:
        """VWAP 계산"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    def grouped_demean(self, data: DataType, group=None) -> DataType:
        """그룹별 평균 제거 (간단 구현: 전체 평균 제거)"""
        if isinstance(data, pd.DataFrame):
            mean = data.mean(axis=1)
            return data.sub(mean, axis=0)
        return data - data.mean()

    # ========== 회귀 관련 ==========

    def slope(self, data: DataType, window: int) -> DataType:
        """선형 회귀 기울기"""
        def calc_slope(y):
            if len(y) < 2:
                return np.nan
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]
        return data.rolling(window).apply(calc_slope, raw=True)

    def regbeta(self, data: DataType, x: np.ndarray) -> DataType:
        """회귀 베타"""
        window = len(x)
        return data.rolling(window).apply(lambda y: np.polyfit(x, y, deg=1)[0], raw=True)

    def residual(self, data: DataType, window: int, x: DataType = None) -> DataType:
        """선형 회귀 잔차"""
        if x is None:
            def calc_residual(y):
                if len(y) < 2:
                    return np.nan
                x_vals = np.arange(len(y))
                coeffs = np.polyfit(x_vals, y, 1)
                fitted = np.polyval(coeffs, x_vals)
                return y[-1] - fitted[-1]
            return data.rolling(window).apply(calc_residual, raw=True)
        else:
            # 두 시리즈 간 잔차 - simplified for DataFrame
            return data - x  # placeholder

    def ts_rsquare(self, data: DataType, window: int, y: DataType = None) -> DataType:
        """시계열 롤링 R-squared"""
        def calc_rsquare(vals):
            if len(vals) < 2:
                return np.nan
            x = np.arange(len(vals))
            if np.std(vals) == 0:
                return 0
            corr = np.corrcoef(x, vals)[0, 1]
            return corr ** 2 if not np.isnan(corr) else np.nan
        return data.rolling(window).apply(calc_rsquare, raw=True)

    def ts_linear_reg_with_seq(self, data: DataType, window: int, mode: int = 0) -> DataType:
        """시계열 선형 회귀"""
        def calc_reg(y):
            if len(y) < 2:
                return np.nan
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, 1)
            if mode == 0:
                return coeffs[0] * (len(y) - 1) + coeffs[1]
            return coeffs[0]
        return data.rolling(window).apply(calc_reg, raw=True)

    def reg_resi(self, y: DataType, x: DataType, window: int) -> DataType:
        """회귀 잔차 (alias)"""
        return self.residual(y, window, x)

    # ========== 기타 ==========

    def lowday(self, data: DataType, window: int) -> DataType:
        """롤링 윈도우에서 최저점 이후 경과 일수"""
        return data.rolling(window).apply(lambda x: len(x) - np.argmin(x) - 1, raw=True)

    def highday(self, data: DataType, window: int) -> DataType:
        """롤링 윈도우에서 최고점 이후 경과 일수"""
        return data.rolling(window).apply(lambda x: len(x) - np.argmax(x) - 1, raw=True)

    def twise_a_scale(self, data: DataType, scale_val: float = 1.0) -> DataType:
        """스케일링 (표준화)"""
        if isinstance(data, pd.DataFrame):
            mean = data.mean(axis=1)
            std = data.std(axis=1).replace(0, np.nan)
            return data.sub(mean, axis=0).div(std, axis=0) * scale_val
        std = data.std()
        if std == 0 or pd.isna(std):
            return data * 0
        return ((data - data.mean()) / std) * scale_val


class BaseNeutralizer(ABC):
    """중립화 베이스 클래스"""

    @abstractmethod
    def neutralize(self, alpha_dict: Dict[str, pd.Series],
                  loading_matrix: Optional[Dict] = None) -> Dict[str, pd.Series]:
        pass


class AlphaProcessor:
    """알파 처리 엔진"""

    def __init__(self, neutralizer: BaseNeutralizer):
        self.neutralizer = neutralizer

    def l1_normalize_cross_section(self, alpha_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """L1 정규화"""
        result = {}
        all_pairs = list(alpha_dict.keys())

        if len(all_pairs) < 3:
            return alpha_dict

        all_timestamps = set()
        for pair in all_pairs:
            all_timestamps.update(alpha_dict[pair].index)

        for pair in all_pairs:
            result[pair] = pd.Series(0.0, index=alpha_dict[pair].index, dtype=float)

        for timestamp in all_timestamps:
            cross_section_values = {}
            for pair in all_pairs:
                if timestamp in alpha_dict[pair].index:
                    value = alpha_dict[pair].loc[timestamp]
                    if pd.notna(value):
                        cross_section_values[pair] = value

            if cross_section_values:
                values_array = np.array(list(cross_section_values.values()))
                l1_norm = np.sum(np.abs(values_array))

                if l1_norm > 0:
                    for pair, value in cross_section_values.items():
                        result[pair].loc[timestamp] = value / l1_norm

        return result

    def l2_normalize_cross_section(self, alpha_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """L2 정규화"""
        result = {}
        all_pairs = list(alpha_dict.keys())

        if len(all_pairs) < 3:
            return alpha_dict

        all_timestamps = set()
        for pair in all_pairs:
            all_timestamps.update(alpha_dict[pair].index)

        for pair in all_pairs:
            result[pair] = pd.Series(0.0, index=alpha_dict[pair].index, dtype=float)

        for timestamp in all_timestamps:
            cross_section_values = {}
            for pair in all_pairs:
                if timestamp in alpha_dict[pair].index:
                    value = alpha_dict[pair].loc[timestamp]
                    if pd.notna(value):
                        cross_section_values[pair] = value

            if cross_section_values:
                values_array = np.array(list(cross_section_values.values()))
                l2_norm = np.sqrt(np.sum(values_array ** 2))

                if l2_norm > 0:
                    for pair, value in cross_section_values.items():
                        result[pair].loc[timestamp] = value / l2_norm

        return result

    def apply_linear_decay(self, alpha_series: pd.Series, decay_period: int) -> pd.Series:
        """선형 디케이"""
        if len(alpha_series) < decay_period:
            return alpha_series

        weights = np.linspace(1, 0, decay_period)
        weights = weights / weights.sum()

        result = alpha_series.copy()
        for i in range(decay_period-1, len(alpha_series)):
            window_data = alpha_series.iloc[i-decay_period+1:i+1]
            if len(window_data) == decay_period:
                result.iloc[i] = np.sum(window_data * weights)

        return result

    def process_alpha(self, raw_alpha_dict: Dict[str, pd.Series],
                     loading_matrix: Optional[Dict] = None,
                     decay_period: int = 3,
                     norm_type: str = 'l1') -> Dict[str, pd.Series]:
        """알파 처리 파이프라인"""
        result = raw_alpha_dict.copy()
        result = self.neutralizer.neutralize(result, loading_matrix)

        if norm_type == 'l2':
            result = self.l2_normalize_cross_section(result)
        else:
            result = self.l1_normalize_cross_section(result)

        for pair in result:
            result[pair] = self.apply_linear_decay(result[pair], decay_period)

        return result


class AlphaLoader:
    """알파 동적 로딩 클래스"""

    @staticmethod
    def load_alpha(alpha_name: str, params: Dict[str, int] = None, alphas_folder: str = "alphas") -> BaseAlpha:
        try:
            module_path = f"{alphas_folder}.{alpha_name}"
            module = importlib.import_module(module_path)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, BaseAlpha) and
                    attr != BaseAlpha):
                    return attr(params=params)

            raise ValueError(f"No BaseAlpha subclass found in {module_path}")

        except ImportError as e:
            raise ImportError(f"Could not load alpha {alpha_name}: {e}")
