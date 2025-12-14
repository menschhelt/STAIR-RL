import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_071(BaseAlpha):
    """
    alpha191_071: sma(div(sub(ts_max({disk:high},6),{disk:close}),mul(sub(ts_max({disk:high},6),ts_min({disk:low},6)),100)),15,1)
    
    (6일 최고가 - 종가) / ((6일 최고가 - 6일 최저가) * 100)의 15일 SMA (046과 유사하지만 기간이 다름)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_071"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_071 계산
        """
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 6일 최고가와 최저가
        high_max_6 = self.ts_max(high, 6)
        low_min_6 = self.ts_min(low, 6)
        
        # 분자: ts_max(high, 6) - close
        numerator = self.sub(high_max_6, close)
        
        # 분모: (ts_max(high, 6) - ts_min(low, 6)) * 100
        denominator = self.mul(
            self.sub(high_max_6, low_min_6),
            100
        )
        
        # 비율
        ratio = self.div(numerator, denominator)
        
        # 15일 SMA
        alpha = self.sma(ratio, 15, 1)
        
        return alpha.fillna(0)
