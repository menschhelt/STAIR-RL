import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_133(BaseAlpha):
    """
    alpha191_133: div(sub({disk:close},delay({disk:close},12)),mul(delay({disk:close},12),{disk:volume}))
    
    12일 수익률을 거래량으로 정규화
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_133"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_133 계산
        """
        close = data['close']
        volume = data['volume']
        
        # 12일 지연 종가
        close_lag12 = self.delay(close, 12)
        
        # 12일 수익률
        returns_12 = self.sub(close, close_lag12)
        
        # 거래량으로 정규화
        normalizer = self.mul(close_lag12, volume)
        alpha = self.div(returns_12, normalizer)
        
        return alpha.fillna(0)
