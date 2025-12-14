import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_062(BaseAlpha):
    """
    alpha191_062: div(sma(max(sub({disk:close},delay({disk:close},1)),0),6,1),mul(sma(abs(sub({disk:close},delay({disk:close},1))),6,1),100))
    
    6일 상승분 SMA / (6일 절대변화 SMA * 100) = 상승 비율
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_062"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_062 계산
        """
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # 일일 변화
        daily_change = self.sub(close, close_lag1)
        
        # 상승분만
        positive_change = self.max(daily_change, 0)
        
        # 절대 변화
        abs_change = self.abs(daily_change)
        
        # 6일 SMA
        positive_sma = self.sma(positive_change, 6, 1)
        abs_sma = self.sma(abs_change, 6, 1)
        
        # 비율
        alpha = self.div(positive_sma, self.mul(abs_sma, 100))
        
        return alpha.fillna(0)
