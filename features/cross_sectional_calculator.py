"""
Cross-Sectional Alpha Calculator

Wide-format 기반 크로스섹션 알파 계산기.
모든 심볼 데이터를 timestamp × symbol 형태로 로드하여
rank(), zscore() 등의 크로스섹션 연산을 정확하게 수행.

핵심 차이점:
- rank(x): 같은 timestamp에서 모든 심볼 간 랭킹 (axis=1)
- ts_rank(x, window): 단일 심볼의 시계열 롤링 랭킹 (axis=0)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import pyarrow.parquet as pq
from datetime import datetime, timezone
import logging
from concurrent.futures import ThreadPoolExecutor
import importlib

from config.settings import DATA_DIR


class WideDataLoader:
    """Wide-format 데이터 로더 - timestamp × symbol 형태"""

    def __init__(self, binance_data_dir: Optional[Path] = None):
        self.binance_data_dir = binance_data_dir or DATA_DIR / 'binance'
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_wide_data(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str = '5m',
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        모든 심볼의 OHLCV 데이터를 wide format으로 로드.

        Returns:
            Dict[str, pd.DataFrame]: {
                'open': DataFrame (timestamp × symbol),
                'high': DataFrame (timestamp × symbol),
                'low': DataFrame (timestamp × symbol),
                'close': DataFrame (timestamp × symbol),
                'volume': DataFrame (timestamp × symbol),
            }
        """
        # 모든 파티션에서 데이터 로드
        dfs = []
        current = start_date

        while current <= end_date:
            partition_key = current.strftime('%Y%m')
            file_path = self.binance_data_dir / f"binance_futures_{interval}_{partition_key}.parquet"

            if file_path.exists():
                df = pq.read_table(file_path).to_pandas()
                if symbols:
                    df = df[df['symbol'].isin(symbols)]
                dfs.append(df)
                self.logger.info(f"Loaded {len(df)} rows from {partition_key}")

            # 다음 월로 이동
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                current = datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)

        if not dfs:
            return {}

        # 전체 데이터 병합
        all_data = pd.concat(dfs, ignore_index=True)
        all_data = all_data[
            (all_data['timestamp'] >= start_date) &
            (all_data['timestamp'] <= end_date)
        ]

        self.logger.info(f"Total rows: {len(all_data)}, Symbols: {all_data['symbol'].nunique()}")

        # Wide format으로 pivot
        wide_data = {}
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in all_data.columns:
                wide_data[col] = all_data.pivot(
                    index='timestamp',
                    columns='symbol',
                    values=col
                ).sort_index()

        # quote_volume도 있으면 추가
        if 'quote_volume' in all_data.columns:
            wide_data['quote_volume'] = all_data.pivot(
                index='timestamp',
                columns='symbol',
                values='quote_volume'
            ).sort_index()

        return wide_data


class CrossSectionalOperators:
    """
    크로스섹션 연산자 - Wide format (timestamp × symbol) DataFrame 처리

    모든 연산은 DataFrame을 받아 DataFrame을 반환.
    - 크로스섹션 연산 (rank, zscore 등): axis=1 (심볼 간)
    - 시계열 연산 (ts_rank, ts_sum 등): axis=0 (시간 방향)
    """

    # ========== 크로스섹션 연산자 (같은 timestamp에서 심볼 간) ==========

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        크로스섹션 랭킹 - 같은 timestamp에서 모든 심볼 간 순위 (0~1)
        axis=1: 각 행(timestamp)에서 열(symbols) 간 랭킹
        """
        return df.rank(axis=1, pct=True)

    def zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """크로스섹션 Z-score - 같은 timestamp에서 표준화"""
        row_mean = df.mean(axis=1)
        row_std = df.std(axis=1)
        return df.sub(row_mean, axis=0).div(row_std.replace(0, np.nan), axis=0).fillna(0)

    def demean(self, df: pd.DataFrame) -> pd.DataFrame:
        """크로스섹션 평균 제거"""
        row_mean = df.mean(axis=1)
        return df.sub(row_mean, axis=0)

    def scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """크로스섹션 스케일링 - L1 norm으로 나눔"""
        l1_norm = df.abs().sum(axis=1)
        return df.div(l1_norm.replace(0, np.nan), axis=0).fillna(0)

    # ========== 시계열 연산자 (단일 심볼의 시간 방향) ==========

    def ts_delta(self, df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
        """시계열 차분"""
        return df.diff(periods=period)

    def ts_rank(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """시계열 롤링 랭킹 - 각 심볼의 시간 방향 순위"""
        return df.rolling(window).rank(pct=True)

    def ts_sum(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """시계열 롤링 합계"""
        return df.rolling(window).sum()

    def ts_mean(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """시계열 롤링 평균"""
        return df.rolling(window).mean()

    def ts_min(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """시계열 롤링 최소값"""
        return df.rolling(window).min()

    def ts_max(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """시계열 롤링 최대값"""
        return df.rolling(window).max()

    def ts_std(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """시계열 롤링 표준편차"""
        return df.rolling(window).std()

    def ts_corr(self, x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
        """시계열 롤링 상관계수"""
        return x.rolling(window).corr(y)

    def ts_cov(self, x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
        """시계열 롤링 공분산"""
        return x.rolling(window).cov(y)

    def ts_argmax(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """시계열 롤링 최대값 인덱스"""
        return df.rolling(window).apply(lambda x: np.argmax(x), raw=True)

    def ts_argmin(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """시계열 롤링 최소값 인덱스"""
        return df.rolling(window).apply(lambda x: np.argmin(x), raw=True)

    def ts_product(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """시계열 롤링 곱"""
        return df.rolling(window).apply(lambda x: np.prod(x), raw=True)

    def ts_decayed_linear(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """선형 감쇠 가중평균"""
        weights = np.arange(1, window + 1, dtype=float)
        weights = weights / weights.sum()
        return df.rolling(window).apply(lambda x: np.dot(x, weights) if len(x) == window else np.nan, raw=True)

    # ========== 기본 연산자 ==========

    def add(self, x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        return x + y

    def sub(self, x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        return x - y

    def mul(self, x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        return x * y

    def div(self, x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        """나눗셈 - 0으로 나누기 처리"""
        if isinstance(y, pd.DataFrame):
            result = x / y.replace(0, np.nan)
        else:
            if abs(y) < 1e-10:
                return pd.DataFrame(0, index=x.index, columns=x.columns)
            result = x / y
        return result.replace([np.inf, -np.inf], 0).fillna(0)

    def pow(self, x: pd.DataFrame, exp: float) -> pd.DataFrame:
        return x ** exp

    def log(self, df: pd.DataFrame) -> pd.DataFrame:
        return np.log(df.replace(0, np.nan)).fillna(0)

    def abs(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.abs()

    def sign(self, df: pd.DataFrame) -> pd.DataFrame:
        return np.sign(df).fillna(0)

    def neg(self, df: pd.DataFrame) -> pd.DataFrame:
        return -df

    def delay(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        return df.shift(period)

    def condition(self, cond: pd.DataFrame, true_val: Union[pd.DataFrame, float],
                  false_val: Union[pd.DataFrame, float]) -> pd.DataFrame:
        """조건부 연산"""
        if isinstance(true_val, (int, float)):
            true_val = pd.DataFrame(true_val, index=cond.index, columns=cond.columns)
        if isinstance(false_val, (int, float)):
            false_val = pd.DataFrame(false_val, index=cond.index, columns=cond.columns)
        return true_val.where(cond, false_val)

    def lt(self, x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        return x < y

    def gt(self, x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        return x > y

    def le(self, x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        return x <= y

    def ge(self, x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        return x >= y

    def eq(self, x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        return x == y

    def and_(self, x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        return x & y

    def or_(self, x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        return x | y

    def min(self, x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        return np.minimum(x, y)

    def max(self, x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        return np.maximum(x, y)

    # ========== 추가 유틸리티 함수 ==========

    def sma(self, df: pd.DataFrame, n: int, m: int = 1) -> pd.DataFrame:
        """지수이동평균"""
        return df.ewm(alpha=m/n, adjust=False).mean()

    def ema(self, df: pd.DataFrame, span: int, alpha: float = None) -> pd.DataFrame:
        """지수 이동 평균"""
        if alpha is not None:
            return df.ewm(alpha=alpha, adjust=False).mean()
        return df.ewm(span=span, adjust=False).mean()

    def ma(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """이동 평균 (Simple Moving Average)"""
        return df.rolling(window).mean()

    def returns(self, close: pd.DataFrame) -> pd.DataFrame:
        """수익률 계산"""
        return close.pct_change()

    def vwap_calc(self, high: pd.DataFrame, low: pd.DataFrame,
                  close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        """VWAP 계산"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    def slope(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """선형 회귀 기울기"""
        def calc_slope(y):
            if len(y) < 2:
                return np.nan
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]
        return df.rolling(window).apply(calc_slope, raw=True)

    def regbeta(self, df: pd.DataFrame, x: np.ndarray) -> pd.DataFrame:
        """회귀 베타 계산"""
        window = len(x)
        return df.rolling(window).apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)

    def sequence(self, n: int) -> np.ndarray:
        """1~n 등차수열 생성"""
        return np.arange(1, n+1)

    def ts_quantile(self, df: pd.DataFrame, window: int, q: float = 0.5) -> pd.DataFrame:
        """시계열 롤링 분위수"""
        return df.rolling(window).quantile(q)

    def lowday(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """롤링 윈도우에서 최저점 이후 경과 일수"""
        return df.rolling(window).apply(
            lambda x: len(x) - np.argmin(x) - 1 if len(x) > 0 else np.nan, raw=True
        )

    def highday(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """롤링 윈도우에서 최고점 이후 경과 일수"""
        return df.rolling(window).apply(
            lambda x: len(x) - np.argmax(x) - 1 if len(x) > 0 else np.nan, raw=True
        )

    def residual(self, y: pd.DataFrame, window: int, x: pd.DataFrame = None) -> pd.DataFrame:
        """선형 회귀 잔차"""
        if x is None:
            # 시간 대비 잔차
            def calc_residual(data):
                if len(data) < 2:
                    return np.nan
                x_vals = np.arange(len(data))
                coeffs = np.polyfit(x_vals, data, 1)
                fitted = np.polyval(coeffs, x_vals)
                return data[-1] - fitted[-1]
            return y.rolling(window).apply(calc_residual, raw=True)
        else:
            # 두 시리즈 간 잔차 - 심볼별로 계산
            result = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
            for col in y.columns:
                if col in x.columns:
                    y_col = y[col].values
                    x_col = x[col].values
                    res = np.full(len(y_col), np.nan)
                    for i in range(window - 1, len(y_col)):
                        y_win = y_col[i - window + 1:i + 1]
                        x_win = x_col[i - window + 1:i + 1]
                        if not np.any(np.isnan(y_win)) and not np.any(np.isnan(x_win)):
                            coeffs = np.polyfit(x_win, y_win, 1)
                            fitted = np.polyval(coeffs, x_win)
                            res[i] = y_win[-1] - fitted[-1]
                    result[col] = res
            return result

    def reg_resi(self, y: pd.DataFrame, x: pd.DataFrame, window: int) -> pd.DataFrame:
        """회귀 잔차 (residual의 별칭)"""
        return self.residual(y, window, x)

    def ts_rsquare(self, df: pd.DataFrame, window: int, y: pd.DataFrame = None) -> pd.DataFrame:
        """시계열 롤링 R-squared"""
        if y is None:
            def calc_rsquare_time(data):
                if len(data) < 2:
                    return np.nan
                x_vals = np.arange(len(data))
                if np.std(data) == 0:
                    return 0
                corr = np.corrcoef(x_vals, data)[0, 1]
                return corr ** 2 if not np.isnan(corr) else np.nan
            return df.rolling(window).apply(calc_rsquare_time, raw=True)
        else:
            # 두 DataFrame 간 R² - 심볼별
            result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
            for col in df.columns:
                if col in y.columns:
                    x_col = df[col].values
                    y_col = y[col].values
                    res = np.full(len(x_col), np.nan)
                    for i in range(window - 1, len(x_col)):
                        x_win = x_col[i - window + 1:i + 1]
                        y_win = y_col[i - window + 1:i + 1]
                        if not np.any(np.isnan(x_win)) and not np.any(np.isnan(y_win)):
                            if np.std(x_win) > 0 and np.std(y_win) > 0:
                                corr = np.corrcoef(x_win, y_win)[0, 1]
                                res[i] = corr ** 2 if not np.isnan(corr) else np.nan
                    result[col] = res
            return result

    def ts_linear_reg_with_seq(self, df: pd.DataFrame, window: int, mode: int = 0) -> pd.DataFrame:
        """시계열 선형 회귀"""
        def calc_reg(y):
            if len(y) < 2:
                return np.nan
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, 1)
            if mode == 0:
                return coeffs[0] * (len(y) - 1) + coeffs[1]
            return coeffs[0]
        return df.rolling(window).apply(calc_reg, raw=True)

    def grouped_demean(self, df: pd.DataFrame, group: str = None) -> pd.DataFrame:
        """그룹별 평균 제거 - 크로스섹션 demean으로 처리"""
        return self.demean(df)

    def twise_a_scale(self, df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
        """스케일링 (크로스섹션 Z-score)"""
        return self.zscore(df) * scale


class CrossSectionalAlphaCalculator:
    """
    크로스섹션 알파 계산기

    Wide-format 데이터에서 알파를 계산.
    기존 알파 정의 파일을 재사용하되, 연산자를 크로스섹션 버전으로 대체.
    """

    def __init__(
        self,
        binance_data_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.data_loader = WideDataLoader(binance_data_dir)
        self.ops = CrossSectionalOperators()
        self.cache_dir = cache_dir or DATA_DIR / 'features' / 'cross_sectional_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

    def _setup_logging(self):
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def load_data(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str = '5m',
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Wide format으로 데이터 로드"""
        return self.data_loader.load_wide_data(start_date, end_date, interval, symbols)

    def calculate_alpha_101_001(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Alpha 101_001: mul(-1,ts_corr(rank(ts_delta(log(volume),2)),rank(div(sub(close,open),open)),6))
        """
        close = data['close']
        open_ = data['open']
        volume = data['volume']

        # 1. log(volume)의 2기간 차분 후 크로스섹션 랭킹
        log_volume = self.ops.log(volume)
        volume_delta = self.ops.ts_delta(log_volume, 2)
        volume_rank = self.ops.rank(volume_delta)  # 크로스섹션 랭킹!

        # 2. (close - open) / open의 크로스섹션 랭킹
        price_change = self.ops.div(self.ops.sub(close, open_), open_)
        price_rank = self.ops.rank(price_change)  # 크로스섹션 랭킹!

        # 3. 6기간 상관계수 * -1
        corr = self.ops.ts_corr(volume_rank, price_rank, 6)
        alpha = self.ops.mul(corr, -1)

        return alpha.fillna(0)

    def calculate_alpha_101_006(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Alpha 101_006: condition based on volume and price change
        """
        close = data['close']
        volume = data['volume']

        # amount = volume * close
        amount = self.ops.mul(volume, close)

        # 조건: ts_mean(amount, 20) < volume
        amount_mean_20 = self.ops.ts_mean(amount, 20)
        condition_check = self.ops.lt(amount_mean_20, volume)

        # ts_delta(close, 7)
        close_delta_7 = self.ops.ts_delta(close, 7)

        # ts_rank(abs(ts_delta(close, 7)), 60)
        abs_close_delta = self.ops.abs(close_delta_7)
        ts_rank_60 = self.ops.ts_rank(abs_close_delta, 60)

        # -1 * ts_rank_60 * sign(delta)
        neg_ts_rank = self.ops.mul(ts_rank_60, -1)
        sign_delta = self.ops.sign(close_delta_7)
        true_value = self.ops.mul(neg_ts_rank, sign_delta)

        # 조건부 결과
        alpha = self.ops.condition(condition_check, true_value, -1)

        return alpha.fillna(0)

    def calculate_all_alphas(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str = '5m',
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        모든 알파를 wide format으로 계산.

        Returns:
            Dict[str, pd.DataFrame]: {alpha_name: DataFrame (timestamp × symbol)}
        """
        # 데이터 로드
        self.logger.info("Loading wide-format data...")
        data = self.load_data(start_date, end_date, interval, symbols)

        if not data:
            self.logger.error("No data loaded!")
            return {}

        self.logger.info(f"Data loaded: {len(data['close'])} timestamps, {len(data['close'].columns)} symbols")

        # 알파 계산
        alphas = {}

        # Alpha 101 (101개)
        self.logger.info("Calculating Alpha 101...")
        for i in range(101):
            alpha_name = f"alpha_101_{i:03d}"
            try:
                alpha_df = self._calculate_alpha_from_definition(alpha_name, data)
                if alpha_df is not None and not alpha_df.empty:
                    alphas[alpha_name] = alpha_df
            except Exception as e:
                self.logger.debug(f"Error calculating {alpha_name}: {e}")

        # Alpha 191 (190개, 000~189)
        self.logger.info("Calculating Alpha 191...")
        for i in range(190):
            alpha_name = f"alpha_191_{i:03d}"
            try:
                alpha_df = self._calculate_alpha_from_definition(alpha_name, data)
                if alpha_df is not None and not alpha_df.empty:
                    alphas[alpha_name] = alpha_df
            except Exception as e:
                self.logger.debug(f"Error calculating {alpha_name}: {e}")

        self.logger.info(f"Calculated {len(alphas)} alphas")
        return alphas

    def _calculate_alpha_from_definition(
        self,
        alpha_name: str,
        data: Dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """
        기존 알파 정의를 읽고 wide-format으로 계산.

        이 함수는 각 알파의 calculate() 로직을 크로스섹션 연산자로 재구현.
        """
        # 알파 파일에서 정의 로드 시도
        try:
            if 'alpha_101' in alpha_name:
                module_path = f"alphas.alpha_101.{alpha_name}"
            else:
                module_path = f"alphas.alpha_191.{alpha_name}"

            module = importlib.import_module(module_path)

            # 클래스 찾기
            alpha_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and attr_name.startswith('alpha_'):
                    alpha_class = attr
                    break

            if alpha_class is None:
                return None

            # 인스턴스 생성 (params 없이)
            instance = alpha_class()

            # 크로스섹션 연산자로 계산
            return self._execute_alpha_cross_sectional(instance, data)

        except Exception as e:
            self.logger.debug(f"Could not load {alpha_name}: {e}")
            return None

    def _execute_alpha_cross_sectional(
        self,
        alpha_instance,
        data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        알파 인스턴스의 계산 로직을 크로스섹션 버전으로 실행.

        알파 인스턴스의 메서드들을 크로스섹션 연산자로 대체.
        """
        # 알파 인스턴스의 연산자를 크로스섹션 버전으로 교체
        ops = self.ops

        # 원본 메서드 저장 및 교체
        original_methods = {}
        methods_to_replace = [
            'rank', 'ts_delta', 'ts_rank', 'ts_sum', 'ts_mean', 'ts_min', 'ts_max',
            'ts_std', 'ts_corr', 'ts_cov', 'ts_argmax', 'ts_argmin', 'ts_product',
            'ts_decayed_linear', 'add', 'sub', 'mul', 'div', 'pow', 'log', 'abs',
            'sign', 'neg', 'delay', 'condition', 'lt', 'gt', 'le', 'ge', 'eq',
            'and_', 'or_', 'min', 'max', 'sma', 'ema', 'ma', 'returns', 'vwap_calc',
            'slope', 'regbeta', 'sequence', 'ts_quantile', 'lowday', 'highday',
            'residual', 'reg_resi', 'ts_rsquare', 'ts_linear_reg_with_seq',
            'grouped_demean', 'twise_a_scale', 'demean', 'zscore', 'scale'
        ]

        for method_name in methods_to_replace:
            if hasattr(alpha_instance, method_name):
                original_methods[method_name] = getattr(alpha_instance, method_name)
            if hasattr(ops, method_name):
                setattr(alpha_instance, method_name, getattr(ops, method_name))

        # 가짜 dataframe 생성 (딕셔너리 접근용)
        class WideDataFrame:
            def __init__(self, data_dict):
                self._data = data_dict
                self.columns = list(data_dict.keys())

            def __getitem__(self, key):
                return self._data.get(key, pd.DataFrame())

            def __contains__(self, key):
                return key in self._data

        fake_df = WideDataFrame(data)

        try:
            # 알파 계산 실행
            result = alpha_instance.calculate(fake_df, 'WIDE_FORMAT')
            return result.fillna(0)
        finally:
            # 원본 메서드 복원
            for method_name, original in original_methods.items():
                setattr(alpha_instance, method_name, original)

    def save_to_cache(
        self,
        alphas: Dict[str, pd.DataFrame],
        filename: str = 'cross_sectional_alphas.parquet'
    ):
        """알파를 캐시 파일로 저장"""
        # Wide format을 long format으로 변환 후 저장
        all_data = []

        for alpha_name, df in alphas.items():
            # Melt to long format
            melted = df.reset_index().melt(
                id_vars='timestamp',
                var_name='symbol',
                value_name=alpha_name
            )
            all_data.append(melted.set_index(['timestamp', 'symbol']))

        if all_data:
            combined = pd.concat(all_data, axis=1)
            combined.to_parquet(self.cache_dir / filename)
            self.logger.info(f"Saved {len(alphas)} alphas to {self.cache_dir / filename}")

    def calculate_factors(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        9개 팩터 계산 (wide format)

        1. returns: 수익률
        2. volatility: 변동성 (20일)
        3. momentum: 모멘텀 (20일 수익률)
        4. volume_momentum: 거래량 모멘텀
        5. price_range: 가격 범위 / 종가
        6. vwap_deviation: VWAP 대비 편차
        7. mkt: 시장 수익률 (BTC 수익률)
        8. smb: Small Minus Big (시가총액 관련)
        9. hml: High Minus Low (가치 관련)
        """
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        factors = {}

        # 1. Returns
        factors['returns'] = self.ops.returns(close)

        # 2. Volatility (20일 수익률 표준편차)
        returns = factors['returns']
        factors['volatility'] = self.ops.ts_std(returns, 20)

        # 3. Momentum (20일 수익률)
        factors['momentum'] = close.pct_change(20)

        # 4. Volume Momentum
        factors['volume_momentum'] = volume.pct_change(5)

        # 5. Price Range
        price_range = self.ops.div(self.ops.sub(high, low), close)
        factors['price_range'] = price_range

        # 6. VWAP Deviation
        vwap = self.ops.vwap_calc(high, low, close, volume)
        factors['vwap_deviation'] = self.ops.div(self.ops.sub(close, vwap), vwap)

        # 7-9. Fama-French style factors (simplified)
        # mkt: 전체 평균 수익률
        factors['mkt'] = returns.mean(axis=1).to_frame().reindex(columns=close.columns, method=None).fillna(method='ffill', axis=1)
        # 실제로는 모든 컬럼에 같은 값
        mkt_series = returns.mean(axis=1)
        factors['mkt'] = pd.DataFrame(
            np.tile(mkt_series.values.reshape(-1, 1), (1, len(close.columns))),
            index=close.index,
            columns=close.columns
        )

        # smb, hml은 데이터 부족으로 0으로 설정
        factors['smb'] = pd.DataFrame(0, index=close.index, columns=close.columns)
        factors['hml'] = pd.DataFrame(0, index=close.index, columns=close.columns)

        return factors


# ========== CLI ==========

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Cross-Sectional Alpha Calculator')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, nargs='*', help='Specific symbols')
    parser.add_argument('--output', type=str, default='cross_sectional_alphas.parquet')

    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    calculator = CrossSectionalAlphaCalculator()

    alphas = calculator.calculate_all_alphas(
        start_date=start,
        end_date=end,
        symbols=args.symbols,
    )

    calculator.save_to_cache(alphas, args.output)
    print(f"Calculated {len(alphas)} alphas")
