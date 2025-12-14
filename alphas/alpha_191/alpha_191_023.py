import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_023(BaseAlpha):
    """
    alpha191_023: sma(sub({disk:close},delay({disk:close},5)),5,1)
    
    5일 가격 변화의 5일 SMA
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_023"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_023 계산
        """
        close = data['close']
        close_lag5 = self.delay(close, 5)
        
        # 5일 가격 변화
        price_change = self.sub(close, close_lag5)
        
        # 5일 SMA
        alpha = self.sma(price_change, 5, 1)
        
        return alpha.fillna(0)
