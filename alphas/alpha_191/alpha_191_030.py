import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_030(BaseAlpha):
    """
    alpha191_030: div(sub({disk:close},ts_mean({disk:close},12)),mul(ts_mean({disk:close},12),100))
    
    (현재 종가 - 12일 평균) / (12일 평균 * 100) = 12일 평균 대비 편차율(%)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_030"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_030 계산
        """
        close = data['close']
        
        # 12일 평균
        close_mean_12 = self.ts_mean(close, 12)
        
        # 현재 종가 - 12일 평균
        price_diff = self.sub(close, close_mean_12)
        
        # 분모: 12일 평균 * 100
        denominator = self.mul(close_mean_12, 100)
        
        alpha = self.div(price_diff, denominator)
        
        return alpha.fillna(0)
