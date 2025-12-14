import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_123(BaseAlpha):
    """
    alpha191_123: div(sub({disk:close},{disk:vwap}),ts_decayed_linear(rank(ts_max({disk:close},30)),2))
    
    (종가 - VWAP) / 종가 최대값 랭킹의 2일 decayed linear
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_123"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_123 계산
        """
        close = data['close']
        vwap = data['vwap']
        
        # 분자: close - vwap
        numerator = self.sub(close, vwap)
        
        # 분모: ts_decayed_linear(rank(ts_max(close, 30)), 2)
        close_max_30 = self.ts_max(close, 30)
        close_max_rank = self.rank(close_max_30)
        denominator = self.ts_decayed_linear(close_max_rank, 2)
        
        alpha = self.div(numerator, denominator)
        
        return alpha.fillna(0)
