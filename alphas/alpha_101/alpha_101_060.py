import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_060(BaseAlpha):
    """
    alpha101_060: lt(rank(sub({disk:vwap},ts_min({disk:vwap},16.1219))),rank(ts_corr({disk:vwap},ts_mean({disk:amount},180),17.9282)))
    
    VWAP과 최소 VWAP 차이의 랭킹이 VWAP-거래대금 상관관계 랭킹보다 작은지 비교
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_060"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_060 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: VWAP - 최소 VWAP
        # 1. ts_min(vwap, 16.1219)
        vwap_min_window = int(round(16.1219))  # 16
        vwap_min = self.ts_min(data['vwap'], vwap_min_window)
        
        # 2. sub(vwap, vwap_min)
        vwap_diff = self.sub(data['vwap'], vwap_min)
        
        # 3. rank(vwap_diff)
        first_rank = self.rank(vwap_diff)
        
        # 두 번째 부분: VWAP-거래대금 상관관계
        # 4. ts_mean(amount, 180)
        amount_mean = self.ts_mean(data['amount'], 180)
        
        # 5. ts_corr(vwap, amount_mean, 17.9282)
        corr_window = int(round(17.9282))  # 18
        corr_result = self.ts_corr(data['vwap'], amount_mean, corr_window)
        
        # 6. rank(corr_result)
        second_rank = self.rank(corr_result)
        
        # 7. lt(first_rank, second_rank)
        alpha = self.lt(first_rank, second_rank)
        
        # 불린을 숫자로 변환
        alpha = alpha.astype(float)
        
        return alpha.fillna(0)
