import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_020(BaseAlpha):
    """
    alpha191_020: ts_linear_reg_with_seq(ts_mean({disk:close},6),6,0)
    
    6일 평균 종가의 6일 선형 회귀 (절편)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_020"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_020 계산
        """
        close = data['close']
        
        # 6일 평균
        close_mean_6 = self.ts_mean(close, 6)
        
        # 6일 선형 회귀 (절편)
        alpha = self.ts_linear_reg_with_seq(close_mean_6, 6, 0)
        
        return alpha.fillna(0)
