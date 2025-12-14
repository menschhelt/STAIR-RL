import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_064(BaseAlpha):
    """
    alpha191_064: div(ts_mean({disk:close},6),{disk:close})
    
    6일 평균 종가 / 현재 종가 (033과 동일하지만 기간이 다름)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_064"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_064 계산
        """
        close = data['close']
        
        # 6일 평균
        close_mean_6 = self.ts_mean(close, 6)
        
        alpha = self.div(close_mean_6, close)
        
        return alpha.fillna(0)
