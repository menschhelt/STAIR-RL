#!/usr/bin/env python3
"""
Build only alpha_101_100 to complete alpha_101 series
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


class SingleAlphaBuilder:
    """빠른 단일 알파 빌더"""

    def __init__(self):
        self.binance_data_dir = DATA_DIR / 'binance'
        self.alpha_cache_dir = Path('/home/work/data/stair-local/features/alpha_cache')
        self.alpha_cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger('SingleAlphaBuilder')
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

        # 미리 계산된 필드 추가
        self.logger.info("Pre-calculating derived fields...")

        # vwap
        typical_price = (wide_data['high'] + wide_data['low'] + wide_data['close']) / 3
        wide_data['vwap'] = (typical_price * wide_data['volume']).cumsum() / wide_data['volume'].cumsum()

        # returns
        wide_data['returns'] = wide_data['close'].pct_change(fill_method=None)

        # amount
        wide_data['amount'] = wide_data['volume'] * wide_data['close']

        self.logger.info("  Added: vwap, returns, amount")

        return wide_data

    def detect_symbol_start_dates(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, datetime]:
        """심볼별 시작 날짜 감지"""
        symbol_starts = {}
        close_df = data['close']

        self.logger.info(f"Detecting start dates for {len(close_df.columns)} symbols...")

        for symbol in close_df.columns:
            first_valid_idx = close_df[symbol].first_valid_index()
            if first_valid_idx is not None:
                symbol_starts[symbol] = first_valid_idx

        self.logger.info(f"Found start dates for {len(symbol_starts)} symbols")
        return symbol_starts

    def apply_historical_mask(
        self,
        data: Dict[str, pd.DataFrame],
        symbol_starts: Dict[str, datetime]
    ) -> tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """히스토리컬 마스크 적용"""
        self.logger.info("Applying historical mask to prevent lookahead bias...")

        masked_data = {}
        close_df = data['close']
        historical_mask = pd.DataFrame(True, index=close_df.index, columns=close_df.columns)

        for symbol in close_df.columns:
            if symbol in symbol_starts:
                start_date = symbol_starts[symbol]
                mask = close_df.index < start_date
                historical_mask.loc[mask, symbol] = False

        for col_name, df in data.items():
            if col_name not in ['open', 'high', 'low', 'close', 'volume']:
                masked_data[col_name] = df
                continue

            masked_df = df.copy()
            masked_df[~historical_mask] = np.nan
            masked_data[col_name] = masked_df

        self.logger.info(f"Historical mask created: {historical_mask.shape}, "
                        f"Avg symbols per timestamp: {historical_mask.sum(axis=1).mean():.1f}")

        return masked_data, historical_mask

    def _load_alpha_class(self, alpha_name: str):
        """동적으로 알파 클래스 로드"""
        try:
            module_path = f'alphas.alpha_101.{alpha_name}'
            module = importlib.import_module(module_path)
            alpha_class = getattr(module, alpha_name)
            return alpha_class
        except Exception as e:
            self.logger.error(f"Failed to load {alpha_name}: {e}")
            return None

    def _calculate_single_alpha(
        self,
        alpha_name: str,
        data: Dict[str, pd.DataFrame],
        historical_mask: Optional[pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
        """단일 알파 계산"""
        try:
            alpha_class = self._load_alpha_class(alpha_name)
            if alpha_class is None:
                return None

            alpha_instance = alpha_class()
            result = alpha_instance.calculate(data, pair=None)

            if result is not None:
                result = result.replace([np.inf, -np.inf], 0).fillna(0)

                if historical_mask is not None:
                    result[~historical_mask] = np.nan

                return result.astype(np.float32)
            return None

        except Exception as e:
            self.logger.error(f"Alpha {alpha_name} failed: {e}")
            return None

    def build(
        self,
        start_date: datetime,
        end_date: datetime,
        alpha_name: str = 'alpha_101_100',
        interval: str = '5m'
    ):
        """단일 알파 빌드"""
        total_start = time.time()

        # 기존 캐시 확인
        cache_path = self.alpha_cache_dir / f"{alpha_name}.parquet"
        if cache_path.exists():
            self.logger.info(f"{alpha_name} already exists, skipping...")
            return

        # 데이터 로드
        data = self.load_wide_data(start_date, end_date, interval)
        if not data:
            self.logger.error("No data loaded!")
            return

        # 히스토리컬 마스크 적용
        symbol_starts = self.detect_symbol_start_dates(data)
        data, historical_mask = self.apply_historical_mask(data, symbol_starts)

        # 알파 계산
        self.logger.info(f"Calculating {alpha_name}...")
        alpha_start = time.time()
        alpha_df = self._calculate_single_alpha(alpha_name, data, historical_mask)

        if alpha_df is not None:
            # 저장
            alpha_df.to_parquet(cache_path, compression='snappy')
            self.logger.info(f"✓ Saved {alpha_name} in {time.time()-alpha_start:.1f}s")
        else:
            self.logger.error(f"✗ Failed to calculate {alpha_name}")

        total_elapsed = time.time() - total_start
        self.logger.info(f"Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build single alpha_101_100')
    parser.add_argument('--start', type=str, default='2023-06-01', help='Start date')
    parser.add_argument('--end', type=str, default='2025-11-30', help='End date')
    parser.add_argument('--alpha', type=str, default='alpha_101_100', help='Alpha name')
    parser.add_argument('--interval', type=str, default='5m', help='Data interval')

    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    builder = SingleAlphaBuilder()
    builder.build(start, end, args.alpha, args.interval)
