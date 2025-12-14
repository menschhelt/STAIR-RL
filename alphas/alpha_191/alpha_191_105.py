import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_105(BaseAlpha):
    """
    alpha191_105: sub({disk:close},delay({disk:close},20))
    
    20일 가격 변화 (현재 종가 - 20일 전 종가) - 013과 유사하지만 기간이 다름
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_105"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_105 계산
        """
        close = data['close']
        close_lag20 = self.delay(close, 20)
        
        alpha = self.sub(close, close_lag20)
        
        return alpha.fillna(0)
