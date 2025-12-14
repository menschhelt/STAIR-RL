import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_188(BaseAlpha):
    """
    alpha191_188: ts_mean(abs(sub({disk:close},ts_mean({disk:close},6))),6)
    
    종가와 6일 평균의 절대 편차를 6일 평균
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_188"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_188 계산
        """
        close = data['close']
        
        # 6일 평균
        close_mean = self.ts_mean(close, 6)
        
        # 절대 편차
        abs_deviation = self.abs(self.sub(close, close_mean))
        
        # 절대 편차의 6일 평균
        alpha = self.ts_mean(abs_deviation, 6)
        
        return alpha.fillna(0)
