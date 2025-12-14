import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_095(BaseAlpha):
    """
    alpha191_095: sma(sma(div(sub({disk:close},ts_min({disk:low},9)),mul(sub(ts_max({disk:high},9),ts_min({disk:low},9)),100)),3,1),3,1)
    
    9일 스토캐스틱 %K의 3일 SMA의 3일 SMA (이중 스무딩)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_095"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_095 계산
        """
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 9일 최고가와 최저가
        high_max_9 = self.ts_max(high, 9)
        low_min_9 = self.ts_min(low, 9)
        
        # 스토캐스틱 %K 계산
        numerator = self.sub(close, low_min_9)
        denominator = self.mul(
            self.sub(high_max_9, low_min_9),
            100
        )
        
        stoch_k = self.div(numerator, denominator)
        
        # 첫 번째 3일 SMA
        sma1 = self.sma(stoch_k, 3, 1)
        
        # 두 번째 3일 SMA (이중 스무딩)
        alpha = self.sma(sma1, 3, 1)
        
        return alpha.fillna(0)
