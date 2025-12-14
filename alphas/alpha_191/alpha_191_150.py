import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_150(BaseAlpha):
    """
    alpha191_150: sma(sub({disk:close},delay({disk:close},20)),20,1)
    
    20일 가격 변화의 20일 이동평균
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_150"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_150 계산
        """
        close = data['close']
        
        # 20일 가격 변화
        close_lag20 = self.delay(close, 20)
        price_change = self.sub(close, close_lag20)
        
        # 20일 SMA
        alpha = self.sma(price_change, 20, 1)
        
        return alpha.fillna(0)
