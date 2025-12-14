import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_093(BaseAlpha):
    """
    alpha101_093: mul(pow(rank(sub({disk:vwap},ts_min({disk:vwap},11.5783))),ts_rank(ts_corr(ts_rank({disk:vwap},19.6462),ts_rank(ts_mean({disk:amount},60),4.02992),18.0926),2.70756)),-1)
    
    VWAP과 최소 VWAP 차이의 랭킹을 VWAP-거래대금 시계열 랭킹 상관관계의 시계열 랭킹으로 거듭제곱한 값의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_093"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_093 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 밑: VWAP과 최소 VWAP 차이의 랭킹
        # 1. ts_min(vwap, 11.5783) -> 반올림하여 12
        min_window = int(round(11.5783))
        vwap_min = self.ts_min(data['vwap'], min_window)
        
        # 2. sub(vwap, vwap_min)
        vwap_diff = self.sub(data['vwap'], vwap_min)
        
        # 3. rank(vwap_diff)
        base = self.rank(vwap_diff)
        
        # 지수: VWAP-거래대금 시계열 랭킹 상관관계의 시계열 랭킹
        # 4. ts_rank(vwap, 19.6462) -> 반올림하여 20
        vwap_rank_window = int(round(19.6462))
        vwap_tsrank = self.ts_rank(data['vwap'], vwap_rank_window)
        
        # 5. ts_mean(amount, 60)
        amount_mean = self.ts_mean(data['amount'], 60)
        
        # 6. ts_rank(amount_mean, 4.02992) -> 반올림하여 4
        amount_rank_window = int(round(4.02992))
        amount_tsrank = self.ts_rank(amount_mean, amount_rank_window)
        
        # 7. ts_corr(vwap_tsrank, amount_tsrank, 18.0926) -> 반올림하여 18
        corr_window = int(round(18.0926))
        corr_result = self.ts_corr(vwap_tsrank, amount_tsrank, corr_window)
        
        # 8. ts_rank(corr_result, 2.70756) -> 반올림하여 3
        final_rank_window = int(round(2.70756))
        power = self.ts_rank(corr_result, final_rank_window)
        
        # 9. pow(base, power)
        powered = self.pow(base, power)
        
        # 10. mul(powered, -1)
        alpha = self.mul(powered, -1)
        
        return alpha.fillna(0)
