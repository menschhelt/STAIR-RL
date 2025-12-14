import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_077(BaseAlpha):
    """
    alpha101_077: pow(rank(ts_corr(ts_sum(add(mul({disk:low},0.352233),mul({disk:vwap},sub(1,0.352233))),19.7428),ts_sum(ts_mean({disk:amount},40),19.7428),6.83313)),rank(ts_corr(rank({disk:vwap}),rank({disk:volume}),5.77492)))
    
    가중 저가-VWAP와 거래대금의 합계 상관관계 랭킹을 VWAP-거래량 랭킹 상관관계 랭킹으로 거듭제곱
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_077"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_077 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        weight = 0.352233
        
        # 첫 번째 부분: 가중 저가-VWAP와 거래대금의 합계 상관관계
        # 1. 가중 저가-VWAP: low * 0.352233 + vwap * (1-0.352233)
        weighted_low_vwap = self.add(
            self.mul(data['low'], weight),
            self.mul(data['vwap'], 1 - weight)
        )
        
        # 2. ts_sum(weighted_low_vwap, 19.7428) -> 반올림하여 20
        sum_window = int(round(19.7428))
        weighted_sum = self.ts_sum(weighted_low_vwap, sum_window)
        
        # 3. ts_mean(amount, 40)
        amount_mean = self.ts_mean(data['amount'], 40)
        
        # 4. ts_sum(amount_mean, 19.7428) -> 반올림하여 20
        amount_sum = self.ts_sum(amount_mean, sum_window)
        
        # 5. ts_corr(weighted_sum, amount_sum, 6.83313) -> 반올림하여 7
        corr_window1 = int(round(6.83313))
        first_corr = self.ts_corr(weighted_sum, amount_sum, corr_window1)
        
        # 6. rank(first_corr)
        base_rank = self.rank(first_corr)
        
        # 두 번째 부분: VWAP-거래량 랭킹 상관관계
        # 7. rank(vwap)
        vwap_rank = self.rank(data['vwap'])
        
        # 8. rank(volume)
        volume_rank = self.rank(data['volume'])
        
        # 9. ts_corr(vwap_rank, volume_rank, 5.77492) -> 반올림하여 6
        corr_window2 = int(round(5.77492))
        second_corr = self.ts_corr(vwap_rank, volume_rank, corr_window2)
        
        # 10. rank(second_corr)
        power_rank = self.rank(second_corr)
        
        # 11. pow(base_rank, power_rank)
        alpha = self.pow(base_rank, power_rank)
        
        return alpha.fillna(0)
