import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_187(BaseAlpha):
    """
    alpha191_187: mul(div(sub({disk:high},sub({disk:low},sma(sub({disk:high},{disk:low}),11,2))),sma(sub({disk:high},{disk:low}),11,2)),100)
    
    가격 범위의 SMA 대비 상대적 위치를 백분율로 표현
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_187"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_187 계산
        """
        high = data['high']
        low = data['low']
        
        # 가격 범위
        price_range = self.sub(high, low)
        
        # 11일 SMA
        range_sma = self.sma(price_range, 11, 2)
        
        # 분자: high - (low - range_sma) = high - low + range_sma
        numerator = self.sub(high, self.sub(low, range_sma))
        
        # 분모: range_sma
        denominator = range_sma
        
        # 비율을 백분율로
        ratio = self.div(numerator, denominator)
        alpha = self.mul(ratio, 100)
        
        return alpha.fillna(100)
