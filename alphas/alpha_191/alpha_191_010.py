import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_010(BaseAlpha):
    """
    alpha191_010: ts_sum(div(sub(sub({disk:close},{disk:low}),sub({disk:high},{disk:close})),mul(sub({disk:high},{disk:low}),{disk:volume})),6)
    
    6일간 거래량 가중 Williams %R의 합계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_010"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_010 계산
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
        
        # 6일 합계
        alpha = self.ts_sum(ratio, 6)
        
        return alpha.fillna(0)
