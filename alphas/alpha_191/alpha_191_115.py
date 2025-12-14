import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_115(BaseAlpha):
    """
    alpha191_115: ts_linear_reg_with_seq({disk:close},20,0)
    
    종가의 20일 선형 회귀 (절편) - 020과 유사하지만 기간이 다름
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_115"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_115 계산
        """
        close = data['close']
        
        # 20일 선형 회귀 (절편)
        alpha = self.ts_linear_reg_with_seq(close, 20, 0)
        
        return alpha.fillna(0)
