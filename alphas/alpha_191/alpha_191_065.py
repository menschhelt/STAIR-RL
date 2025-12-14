import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_065(BaseAlpha):
    """
    alpha191_065: div(sub({disk:close},ts_mean({disk:close},6)),mul(ts_mean({disk:close},6),100))
    
    (현재 종가 - 6일 평균) / (6일 평균 * 100) = 6일 평균 대비 편차율(%)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_065"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_065 계산
        """
        close = data['close']
        
        # 6일 평균
        close_mean_6 = self.ts_mean(close, 6)
        
        # 현재 종가 - 6일 평균
        price_diff = self.sub(close, close_mean_6)
        
        # 분모: 6일 평균 * 100
        denominator = self.mul(close_mean_6, 100)
        
        alpha = self.div(price_diff, denominator)
        
        return alpha.fillna(0)
