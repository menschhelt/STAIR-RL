import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_014(BaseAlpha):
    """
    alpha191_014: div({disk:open},sub(delay({disk:close},1),1))
    
    시가 / (전일 종가 - 1) - 수정된 갭 비율
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_014"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_014 계산
        """
        open_price = data['open']
        close_lag1 = self.delay(data['close'], 1)
        
        # 전일 종가 - 1
        denominator = self.sub(close_lag1, 1)
        
        alpha = self.div(open_price, denominator)
        
        return alpha.fillna(0)
