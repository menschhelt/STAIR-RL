#!/usr/bin/env python3
"""
Fast Feature Builder using BaseAlpha classes

Wide-format 기반으로 모든 알파를 한번에 계산.
BaseAlpha 클래스들을 동적 로드하여 실제 알파 공식 사용.
"""

import sys
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from typing import Dict, List, Optional
import importlib
import time
import gc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR


class FastFeatureBuilder:
    """
    빠른 피처 빌더 - Wide format 기반 크로스섹션 알파 계산
    BaseAlpha 클래스를 동적으로 로드하여 실제 알파 공식 사용
    """

    def __init__(self):
        self.binance_data_dir = DATA_DIR / 'binance'
        self.output_dir = DATA_DIR / 'features' / 'alpha_cache'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger('FastFeatureBuilder')
        self._setup_logging()

    def _setup_logging(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def load_wide_data(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str = '5m',
    ) -> Dict[str, pd.DataFrame]:
        """모든 심볼의 OHLCV를 wide format으로 로드"""
        self.logger.info(f"Loading data from {start_date} to {end_date}...")

        dfs = []
        current = start_date

        while current <= end_date:
            partition_key = current.strftime('%Y%m')
            file_path = self.binance_data_dir / f"binance_futures_{interval}_{partition_key}.parquet"

            if file_path.exists():
                df = pq.read_table(file_path).to_pandas()
                dfs.append(df)
                self.logger.info(f"  Loaded {len(df):,} rows from {partition_key}")

            if current.month == 12:
                current = datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                current = datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)

        if not dfs:
            return {}

        self.logger.info("Concatenating dataframes...")
        all_data = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()

        all_data = all_data[
            (all_data['timestamp'] >= start_date) &
            (all_data['timestamp'] <= end_date)
        ]

        n_timestamps = all_data['timestamp'].nunique()
        n_symbols = all_data['symbol'].nunique()
        self.logger.info(f"Total: {len(all_data):,} rows, {n_timestamps:,} timestamps, {n_symbols} symbols")

        # 메모리 절약: float32로 먼저 변환
        self.logger.info("Converting to float32 before pivot...")
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in all_data.columns:
                all_data[col] = all_data[col].astype(np.float32)
        if 'quote_volume' in all_data.columns:
            all_data['quote_volume'] = all_data['quote_volume'].astype(np.float32)
        gc.collect()

        # Wide format pivot
        wide_data = {}
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in all_data.columns:
                self.logger.info(f"  Pivoting {col}...")
                wide_data[col] = all_data.pivot(
                    index='timestamp',
                    columns='symbol',
                    values=col
                ).sort_index()
                gc.collect()

        if 'quote_volume' in all_data.columns:
            self.logger.info("  Pivoting quote_volume...")
            wide_data['quote_volume'] = all_data.pivot(
                index='timestamp',
                columns='symbol',
                values='quote_volume'
            ).sort_index()
            gc.collect()

        self.logger.info(f"Pivot complete! Shape: {wide_data['close'].shape}")

        del all_data
        gc.collect()

        # 미리 계산된 필드 추가 (많은 알파에서 필요)
        self.logger.info("Pre-calculating derived fields...")

        # vwap
        typical_price = (wide_data['high'] + wide_data['low'] + wide_data['close']) / 3
        wide_data['vwap'] = (typical_price * wide_data['volume']).cumsum() / wide_data['volume'].cumsum()

        # returns
        wide_data['returns'] = wide_data['close'].pct_change()

        # amount
        wide_data['amount'] = wide_data['volume'] * wide_data['close']

        self.logger.info("  Added: vwap, returns, amount")

        return wide_data

    def _load_alpha_class(self, alpha_name: str):
        """동적으로 알파 클래스 로드"""
        try:
            if alpha_name.startswith('alpha_101'):
                module_path = f'alphas.alpha_101.{alpha_name}'
            elif alpha_name.startswith('alpha_191'):
                module_path = f'alphas.alpha_191.{alpha_name}'
            else:
                return None

            module = importlib.import_module(module_path)
            alpha_class = getattr(module, alpha_name)
            return alpha_class
        except Exception as e:
            self.logger.debug(f"Failed to load {alpha_name}: {e}")
            return None

    def _calculate_single_alpha(
        self,
        alpha_name: str,
        data: Dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """BaseAlpha 클래스를 사용하여 단일 알파 계산"""
        try:
            alpha_class = self._load_alpha_class(alpha_name)
            if alpha_class is None:
                return None

            alpha_instance = alpha_class()
            result = alpha_instance.calculate(data, pair=None)

            if result is not None:
                # inf와 NaN 모두 0으로 처리
                result = result.replace([np.inf, -np.inf], 0).fillna(0)
                return result.astype(np.float32)
            return None

        except Exception as e:
            self.logger.debug(f"Alpha {alpha_name} failed: {e}")
            return None

    def calculate_factors(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """9개 팩터 계산"""
        from alphas.base.base import BaseAlpha

        # BaseAlpha 인스턴스를 팩터 계산용으로 사용
        class FactorCalculator(BaseAlpha):
            def calculate(self, data, pair=None):
                pass
            @property
            def name(self):
                return "factor_calc"

        ops = FactorCalculator()
        factors = {}

        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        # 1. returns
        factors['factor_returns'] = ops.returns(close).fillna(0)

        # 2. volatility (20일)
        returns = factors['factor_returns']
        factors['factor_volatility'] = ops.ts_std(returns, 20).fillna(0)

        # 3. momentum (20일 수익률)
        factors['factor_momentum'] = close.pct_change(20).fillna(0)

        # 4. volume_momentum
        factors['factor_volume_momentum'] = volume.pct_change(5).fillna(0)

        # 5. price_range
        factors['factor_price_range'] = ops.div(ops.sub(high, low), close).fillna(0)

        # 6. vwap_deviation
        vwap = ops.vwap_calc(high, low, close, volume)
        factors['factor_vwap_deviation'] = ops.div(ops.sub(close, vwap), vwap).fillna(0)

        # 7. mkt (시장 평균 수익률)
        mkt_returns = returns.mean(axis=1)
        factors['factor_mkt'] = pd.DataFrame(
            np.tile(mkt_returns.values.reshape(-1, 1), (1, len(close.columns))),
            index=close.index, columns=close.columns
        )

        # 8. smb (placeholder)
        factors['factor_smb'] = pd.DataFrame(0, index=close.index, columns=close.columns, dtype=np.float32)

        # 9. hml (placeholder)
        factors['factor_hml'] = pd.DataFrame(0, index=close.index, columns=close.columns, dtype=np.float32)

        return factors

    def _get_all_alpha_names(self) -> List[str]:
        """전체 알파 이름 리스트"""
        names = []
        for i in range(101):
            names.append(f'alpha_101_{i:03d}')
        for i in range(191):
            names.append(f'alpha_191_{i:03d}')
        return names

    def calculate_alpha_batch(
        self,
        data: Dict[str, pd.DataFrame],
        alpha_names: List[str],
        n_jobs: int = 2
    ) -> Dict[str, pd.DataFrame]:
        """알파 배치 계산 - 병렬 처리 (n_jobs=2)"""
        from joblib import Parallel, delayed

        def calc_one(name):
            try:
                result = self._calculate_single_alpha(name, data)
                return (name, result)
            except Exception as e:
                return (name, None)

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(calc_one)(name) for name in alpha_names
        )

        alphas = {}
        for name, result in results:
            if result is not None:
                alphas[name] = result

        return alphas

    def build(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str = '5m'
    ):
        """전체 피처 빌드 파이프라인"""
        from joblib import Parallel, delayed
        total_start = time.time()

        # 1. 데이터 로드
        data = self.load_wide_data(start_date, end_date, interval)
        if not data:
            self.logger.error("No data loaded!")
            return

        symbols = data['close'].columns.tolist()
        timestamps = data['close'].index
        self.logger.info(f"Data loaded: {len(timestamps)} timestamps, {len(symbols)} symbols")

        # 2. 팩터 계산
        self.logger.info("Calculating factors...")
        factor_start = time.time()
        factors = self.calculate_factors(data)
        self.logger.info(f"Factors calculated in {time.time() - factor_start:.1f}s")

        # 3. 알파를 배치로 계산하고 즉시 저장
        all_alpha_names = self._get_all_alpha_names()
        alpha_cache_dir = Path('/home/work/data/stair-local/features/alpha_cache')
        alpha_cache_dir.mkdir(parents=True, exist_ok=True)

        # 기존 캐시 삭제
        for f in alpha_cache_dir.glob('*.parquet'):
            f.unlink()

        batch_size = 20
        alpha_start = time.time()
        completed_alphas = []
        failed_alphas = []

        self.logger.info(f"Calculating {len(all_alpha_names)} alphas in batches of {batch_size}...")

        for i in range(0, len(all_alpha_names), batch_size):
            batch_names = all_alpha_names[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(all_alpha_names) + batch_size - 1) // batch_size

            self.logger.info(f"Batch {batch_num}/{total_batches}: {batch_names[0]} ~ {batch_names[-1]}")

            # 배치 계산
            batch_alphas = self.calculate_alpha_batch(data, batch_names)

            # 즉시 파일로 저장
            for alpha_name, alpha_df in batch_alphas.items():
                cache_path = alpha_cache_dir / f"{alpha_name}.parquet"
                alpha_df.to_parquet(cache_path, compression='snappy')
                completed_alphas.append(alpha_name)

            # 실패한 알파 기록
            for name in batch_names:
                if name not in batch_alphas:
                    failed_alphas.append(name)

            self.logger.info(f"  Saved {len(batch_alphas)} alphas, total: {len(completed_alphas)}/{len(all_alpha_names)}")

            # 메모리 해제
            del batch_alphas
            gc.collect()

        self.logger.info(f"All alphas calculated in {time.time()-alpha_start:.1f}s")
        self.logger.info(f"  Success: {len(completed_alphas)}, Failed: {len(failed_alphas)}")
        if failed_alphas:
            self.logger.warning(f"  Failed alphas: {failed_alphas[:10]}...")

        # 4. 심볼별 파일 생성
        self.logger.info("Combining features per symbol...")

        def save_symbol(sym):
            sym_data = {'timestamp': timestamps}
            # OHLCV
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data:
                    sym_data[col] = data[col][sym].values
            # Factors
            for factor_name, factor_df in factors.items():
                if sym in factor_df.columns:
                    sym_data[factor_name] = factor_df[sym].values
            # Alphas (캐시에서 읽기)
            for alpha_name in completed_alphas:
                cache_path = alpha_cache_dir / f"{alpha_name}.parquet"
                if cache_path.exists():
                    alpha_df = pd.read_parquet(cache_path)
                    if sym in alpha_df.columns:
                        sym_data[alpha_name] = alpha_df[sym].values

            df = pd.DataFrame(sym_data)
            df['symbol'] = sym
            output_path = self.output_dir / f"{sym}_features.parquet"
            df.to_parquet(output_path, compression='snappy')
            return sym

        # 병렬 저장
        save_start = time.time()
        results = Parallel(n_jobs=20, backend='loky', verbose=10)(
            delayed(save_symbol)(sym) for sym in symbols
        )
        self.logger.info(f"Saved {len(results)} symbol files in {time.time()-save_start:.1f}s")

        total_elapsed = time.time() - total_start
        self.logger.info(f"Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        self.logger.info(f"  {len(symbols)} symbols, {len(completed_alphas)} alphas")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fast Feature Builder')
    parser.add_argument('--start', type=str, default='2023-06-01', help='Start date')
    parser.add_argument('--end', type=str, default='2025-11-30', help='End date')
    parser.add_argument('--interval', type=str, default='5m', help='Data interval')

    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    builder = FastFeatureBuilder()
    builder.build(start, end, args.interval)
