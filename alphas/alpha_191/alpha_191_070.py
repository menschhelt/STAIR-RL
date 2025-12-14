import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_070(BaseAlpha):
    """
    alpha191_070: div(sub({disk:close},ts_mean({disk:close},24)),mul(ts_mean({disk:close},24),100))
    
    (현재 종가 - 24일 평균) / (24일 평균 * 100) = 24일 평균 대비 편차율(%)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_070"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_070 계산
        """
        close = data['close']
        
        # 24일 평균
        close_mean_24 = self.ts_mean(close, 24)
        
        # 현재 종가 - 24일 평균
        price_diff = self.sub(close, close_mean_24)
        
        # 분모: 24일 평균 * 100
        denominator = self.mul(close_mean_24, 100)
        
        alpha = self.div(price_diff, denominator)
        
        return alpha.fillna(0)
