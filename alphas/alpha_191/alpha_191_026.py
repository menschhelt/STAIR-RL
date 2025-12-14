import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_026(BaseAlpha):
    """
    alpha191_026: ts_decayed_linear(div(sub({disk:close},delay({disk:close},3)),mul(delay({disk:close},3),add(100,div(sub({disk:close},delay({disk:close},6)),mul(delay({disk:close},6),100))))),12)
    
    복잡한 가격 변화율의 12일 decayed linear
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_026"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_026 계산
        """
        close = data['close']
        
        # 3일 전 종가
        close_lag3 = self.delay(close, 3)
        # 6일 전 종가  
        close_lag6 = self.delay(close, 6)
        
        # 6일 수익률 (%)
        returns_6d = self.div(
            self.sub(close, close_lag6),
            self.mul(close_lag6, 100)
        )
        
        # 분모: delay(close, 3) * (100 + 6일 수익률)
        denominator = self.mul(
            close_lag3,
            (100 + returns_6d)
        )
        
        # 분자: close - delay(close, 3)
        numerator = self.sub(close, close_lag3)
        
        # 비율
        ratio = self.div(numerator, denominator)
        
        # 12일 decayed linear
        alpha = self.ts_decayed_linear(ratio, 12)
        
        return alpha.fillna(0)
