import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_173(BaseAlpha):
    """
    alpha191_173: sma(condition(gt({disk:close},delay({disk:close},1)),ts_std({disk:close},20),0),20,1)
    
    상승일의 20일 변동성을 20일 평활화
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_173"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_173 계산
        """
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # 상승 조건
        is_up = self.gt(close, close_lag1)
        
        # 20일 변동성
        volatility = self.ts_std(close, 20)
        
        # 조건: 상승일이면 변동성, 아니면 0
        conditional_vol = self.condition(is_up, volatility, 0)
        
        # 20일 SMA
        alpha = self.sma(conditional_vol, 20, 1)
        
        return alpha.fillna(0)
