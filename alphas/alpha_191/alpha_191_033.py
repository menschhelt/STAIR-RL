import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_033(BaseAlpha):
    """
    alpha191_033: div(ts_mean({disk:close},12),{disk:close})
    
    12일 평균 종가 / 현재 종가
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_033"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_033 계산
        """
        close = data['close']
        
        # 12일 평균
        close_mean_12 = self.ts_mean(close, 12)
        
        alpha = self.div(close_mean_12, close)
        
        return alpha.fillna(0)
