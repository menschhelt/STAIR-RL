import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_074(BaseAlpha):
    """
    alpha101_074: lt(rank(ts_corr({disk:vwap},{disk:volume},4.24304)),rank(ts_corr(rank({disk:low}),rank(ts_mean({disk:amount},50)),12.4413)))
    
    VWAP-거래량 상관관계 랭킹이 저가-거래대금 랭킹 상관관계 랭킹보다 작은지 비교
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_074"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_074 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: VWAP-거래량 상관관계
        # 1. ts_corr(vwap, volume, 4.24304) -> 반올림하여 4
        corr_window1 = int(round(4.24304))
        vwap_corr = self.ts_corr(data['vwap'], data['volume'], corr_window1)
        
        # 2. rank(vwap_corr)
        first_rank = self.rank(vwap_corr)
        
        # 두 번째 부분: 저가-거래대금 랭킹 상관관계
        # 3. rank(low)
        low_rank = self.rank(data['low'])
        
        # 4. ts_mean(amount, 50)
        amount_mean = self.ts_mean(data['amount'], 50)
        
        # 5. rank(amount_mean)
        amount_rank = self.rank(amount_mean)
        
        # 6. ts_corr(low_rank, amount_rank, 12.4413) -> 반올림하여 12
        corr_window2 = int(round(12.4413))
        low_corr = self.ts_corr(low_rank, amount_rank, corr_window2)
        
        # 7. rank(low_corr)
        second_rank = self.rank(low_corr)
        
        # 8. lt(first_rank, second_rank)
        alpha = self.lt(first_rank, second_rank)
        
        # 불린을 숫자로 변환
        alpha = alpha.astype(float)
        
        return alpha.fillna(0)
