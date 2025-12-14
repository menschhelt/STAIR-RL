import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_059(BaseAlpha):
    """
    alpha191_059: ts_sum(div(sub(sub({disk:close},{disk:low}),sub({disk:high},{disk:close})),mul(sub({disk:high},{disk:low}),{disk:volume})),20)
    
    20일간 거래량 가중 Williams %R 합계 (010과 유사하지만 기간이 다름)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_059"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_059 계산
        """
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Williams %R 계산: (close - low) - (high - close)
        numerator = self.sub(
            self.sub(close, low),
            self.sub(high, close)
        )
        
        # 분모: (high - low) * volume
        denominator = self.mul(
            self.sub(high, low),
            volume
        )
        
        # 비율
        ratio = self.div(numerator, denominator)
        
        # 20일 합계
        alpha = self.ts_sum(ratio, 20)
        
        return alpha.fillna(0)
