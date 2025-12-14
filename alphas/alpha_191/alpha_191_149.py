import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_149(BaseAlpha):
    """
    alpha191_149: div(add({disk:close},add({disk:high},{disk:low})),mul(3,{disk:volume}))
    
    (종가 + 고가 + 저가) / (3 * 거래량) - 단위 거래량당 평균 가격
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_149"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_149 계산
        """
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # 분자: close + high + low
        price_sum = self.add(close, self.add(high, low))
        
        # 분모: 3 * volume
        volume_scaled = self.mul(3, volume)
        
        # 비율 계산
        alpha = self.div(price_sum, volume_scaled)
        
        return alpha.fillna(0)
