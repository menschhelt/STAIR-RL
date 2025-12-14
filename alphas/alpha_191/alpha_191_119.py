import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_119(BaseAlpha):
    """
    alpha191_119: div(rank(sub({disk:vwap},{disk:close})),rank(add({disk:vwap},{disk:close})))
    
    VWAP-종가 차이 랭킹 / VWAP+종가 합계 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_119"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_119 계산
        """
        vwap = data['vwap']
        close = data['close']
        
        # VWAP - close
        vwap_close_diff = self.sub(vwap, close)
        rank1 = self.rank(vwap_close_diff)
        
        # VWAP + close
        vwap_close_sum = self.add(vwap, close)
        rank2 = self.rank(vwap_close_sum)
        
        # 비율
        alpha = self.div(rank1, rank2)
        
        return alpha.fillna(0)
