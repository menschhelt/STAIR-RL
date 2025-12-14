import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_068(BaseAlpha):
    """
    alpha101_068: mul(pow(rank(ts_max(ts_delta(grouped_demean({disk:vwap},{disk:industry_group_lv2}),2.72412),4.79344)),ts_rank(ts_corr(add(mul({disk:close},0.490655),mul({disk:vwap},sub(1,0.490655))),ts_mean({disk:amount},20),4.92416),9.0615)),-1)
    
    업종별 표준화된 VWAP 변화의 최대값 랭킹과 가중 가격-거래대금 상관관계의 시계열 랭킹의 거듭제곱의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_068"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_068 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 업종별 표준화된 VWAP 변화의 최대값
        # 1. grouped_demean(vwap, industry_group_lv2) -> vwap - mean(vwap)
        vwap_demeaned = data['vwap'] - data['vwap'].mean()
        
        # 2. ts_delta(vwap_demeaned, 2.72412) -> 반올림하여 3
        delta_window = int(round(2.72412))
        vwap_delta = self.ts_delta(vwap_demeaned, delta_window)
        
        # 3. ts_max(vwap_delta, 4.79344) -> 반올림하여 5
        max_window = int(round(4.79344))
        vwap_max = self.ts_max(vwap_delta, max_window)
        
        # 4. rank(vwap_max)
        base_rank = self.rank(vwap_max)
        
        # 두 번째 부분: 가중 가격-거래대금 상관관계의 시계열 랭킹
        weight = 0.490655
        
        # 5. 가중 가격: close * 0.490655 + vwap * (1-0.490655)
        weighted_price = self.add(
            self.mul(data['close'], weight),
            self.mul(data['vwap'], 1 - weight)
        )
        
        # 6. ts_mean(amount, 20)
        amount_mean = self.ts_mean(data['amount'], 20)
        
        # 7. ts_corr(weighted_price, amount_mean, 4.92416) -> 반올림하여 5
        corr_window = int(round(4.92416))
        corr_result = self.ts_corr(weighted_price, amount_mean, corr_window)
        
        # 8. ts_rank(corr_result, 9.0615) -> 반올림하여 9
        rank_window = int(round(9.0615))
        power_rank = self.ts_rank(corr_result, rank_window)
        
        # 9. pow(base_rank, power_rank)
        powered = self.pow(base_rank, power_rank)
        
        # 10. mul(powered, -1)
        alpha = self.mul(powered, -1)
        
        return alpha.fillna(0)
