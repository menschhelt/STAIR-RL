import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_174(BaseAlpha):
    """
    alpha191_174: ts_mean(max(max(sub({disk:high},{disk:low}),abs(sub(delay({disk:close},1),{disk:high}))),abs(sub(delay({disk:close},1),{disk:low}))),6)
    
    True Range의 6일 평균 (ATR과 유사)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_174"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_174 계산
        """
        high = data['high']
        low = data['low']
        close_lag1 = self.delay(data['close'], 1)
        
        # True Range 계산
        # 1. high - low
        range1 = self.sub(high, low)
        
        # 2. abs(delay(close,1) - high)
        range2 = self.abs(self.sub(close_lag1, high))
        
        # 3. abs(delay(close,1) - low)
        range3 = self.abs(self.sub(close_lag1, low))
        
        # 최대값 계산
        max_range = self.max(self.max(range1, range2), range3)
        
        # 6일 평균
        alpha = self.ts_mean(max_range, 6)
        
        return alpha.fillna(0)
