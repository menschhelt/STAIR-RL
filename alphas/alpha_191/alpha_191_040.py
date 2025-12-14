import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_040(BaseAlpha):
    """
    alpha191_040: mul(rank(max(ts_delta({disk:vwap},3),5)),-1)
    
    -1 * VWAP 3일 변화의 5일 최대값에 대한 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_040"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_040 계산
        """
        vwap = data['vwap']
        
        # VWAP 3일 변화
        vwap_delta = self.ts_delta(vwap, 3)
        
        # 5일 최대값
        vwap_max = self.ts_max(vwap_delta, 5)
        
        # 랭킹
        vwap_rank = self.rank(vwap_max)
        
        # -1 곱하기
        alpha = self.mul(vwap_rank, -1)
        
        return alpha.fillna(0)
