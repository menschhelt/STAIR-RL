import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_013(BaseAlpha):
    """
    alpha191_013: sub({disk:close},delay({disk:close},5))
    
    5일 가격 변화 (현재 종가 - 5일 전 종가)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_013"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_013 계산
        """
        close = data['close']
        close_lag5 = self.delay(close, 5)
        
        alpha = self.sub(close, close_lag5)
        
        return alpha.fillna(0)
