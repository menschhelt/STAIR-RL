import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_069(BaseAlpha):
    """
    alpha101_069: mul(pow(rank(ts_delta({disk:vwap},1.29456)),ts_rank(ts_corr(grouped_demean({disk:close},{disk:industry_group_lv2}),ts_mean({disk:amount},50),17.8256),17.9171)),-1)
    
    VWAP 변화 랭킹과 업종별 표준화된 종가-거래대금 상관관계의 시계열 랭킹의 거듭제곱의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_069"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_069 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: VWAP 변화 랭킹
        # 1. ts_delta(vwap, 1.29456) -> 반올림하여 1
        delta_window = int(round(1.29456))
        vwap_delta = self.ts_delta(data['vwap'], delta_window)
        
        # 2. rank(vwap_delta)
        base_rank = self.rank(vwap_delta)
        
        # 두 번째 부분: 업종별 표준화된 종가-거래대금 상관관계의 시계열 랭킹
        # 3. grouped_demean(close, industry_group_lv2) -> close - mean(close)
        close_demeaned = data['close'] - data['close'].mean()
        
        # 4. ts_mean(amount, 50)
        amount_mean = self.ts_mean(data['amount'], 50)
        
        # 5. ts_corr(close_demeaned, amount_mean, 17.8256) -> 반올림하여 18
        corr_window = int(round(17.8256))
        corr_result = self.ts_corr(close_demeaned, amount_mean, corr_window)
        
        # 6. ts_rank(corr_result, 17.9171) -> 반올림하여 18
        rank_window = int(round(17.9171))
        power_rank = self.ts_rank(corr_result, rank_window)
        
        # 7. pow(base_rank, power_rank)
        powered = self.pow(base_rank, power_rank)
        
        # 8. mul(powered, -1)
        alpha = self.mul(powered, -1)
        
        return alpha.fillna(0)
