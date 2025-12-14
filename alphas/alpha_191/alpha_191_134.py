import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_134(BaseAlpha):
    """
    alpha191_134: sma(delay(div({disk:close},delay({disk:close},20)),1),20,1)
    
    20일 수익률의 지연 및 평활화 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_134"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_134 계산
        """
        close = data['close']
        
        # 20일 지연 종가
        close_lag20 = self.delay(close, 20)
        
        # 20일 수익률 비율
        returns_ratio = self.div(close, close_lag20)
        
        # 1일 지연
        returns_ratio_lag1 = self.delay(returns_ratio, 1)
        
        # 20일 SMA
        alpha = self.sma(returns_ratio_lag1, 20, 1)
        
        return alpha.fillna(0)
