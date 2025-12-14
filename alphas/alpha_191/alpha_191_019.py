import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_019(BaseAlpha):
    """
    alpha191_019: div(sub({disk:close},delay({disk:close},6)),mul(delay({disk:close},6),100))
    
    6일 수익률 (%) = (현재 종가 - 6일 전 종가) / (6일 전 종가 * 100)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_019"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_019 계산
        """
        close = data['close']
        close_lag6 = self.delay(close, 6)
        
        # 가격 변화
        price_change = self.sub(close, close_lag6)
        
        # 분모: delay(close, 6) * 100
        denominator = self.mul(close_lag6, 100)
        
        alpha = self.div(price_change, denominator)
        
        return alpha.fillna(0)
