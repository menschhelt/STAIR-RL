import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_016(BaseAlpha):
    """
    alpha191_016: pow(rank(sub({disk:vwap},max({disk:vwap},15))),ts_delta({disk:close},5))
    
    (VWAP - 15일 최대 VWAP) 랭킹의 (5일 종가 변화)제곱
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_016"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_016 계산
        """
        vwap = data['vwap']
        close = data['close']
        
        # 15일 최대 VWAP
        vwap_max_15 = self.ts_max(vwap, 15)
        
        # VWAP - 최대 VWAP
        vwap_diff = self.sub(vwap, vwap_max_15)
        
        # 랭킹
        vwap_diff_rank = self.rank(vwap_diff)
        
        # 5일 종가 변화
        close_delta = self.ts_delta(close, 5)
        
        # 거듭제곱
        alpha = self.pow(vwap_diff_rank, close_delta)
        
        return alpha.fillna(0)
