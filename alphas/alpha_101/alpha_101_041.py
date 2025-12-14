import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_041(BaseAlpha):
    """
    alpha101_041: div(rank(sub({disk:vwap},{disk:close})),rank(add({disk:vwap},{disk:close})))
    
    VWAP과 종가의 차이 랭킹을 합계 랭킹으로 나눈 비율
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_041"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_041 계산
        """
        # 1. sub(vwap, close)
        vwap_close_diff = self.sub(data['vwap'], data['close'])
        
        # 2. rank(vwap_close_diff)
        diff_rank = self.rank(vwap_close_diff)
        
        # 3. add(vwap, close)
        vwap_close_sum = self.add(data['vwap'], data['close'])
        
        # 4. rank(vwap_close_sum)
        sum_rank = self.rank(vwap_close_sum)
        
        # 5. div(diff_rank, sum_rank)
        alpha = self.div(diff_rank, sum_rank)
        
        return alpha.fillna(0)