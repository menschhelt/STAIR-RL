import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_146(BaseAlpha):
    """
    alpha191_146: ts_linear_reg_with_seq(ts_mean({disk:close},12),12,0)
    
    12일 종가 평균의 12일 선형회귀 기울기
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_146"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_146 계산
        """
        close = data['close']
        
        # 12일 종가 평균
        close_mean = self.ts_mean(close, 12)
        
        # 선형회귀 기울기 (ts_linear_reg_with_seq가 없는 경우 근사)
        # 12일 기간에 대한 선형 회귀 기울기를 계산
        if hasattr(self, 'ts_linear_reg_with_seq'):
            alpha = self.ts_linear_reg_with_seq(close_mean, 12, 0)
        else:
            # 기울기 근사: (현재값 - 12일전값) / 12
            close_mean_lag12 = self.delay(close_mean, 12)
            alpha = self.div(self.sub(close_mean, close_mean_lag12), 12)
        
        return alpha.fillna(0)
