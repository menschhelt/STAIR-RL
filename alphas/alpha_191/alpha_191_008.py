import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_008(BaseAlpha):
    """
    alpha191_008: sma(mul(div(add({disk:high},{disk:low}),sub(2,div(add(delay({disk:high},1),delay({disk:low},1)),2))),div(sub({disk:high},{disk:low}),{disk:volume})),7,2)
    
    복잡한 SMA 계산 - 가격 변화와 거래량 관련
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_008"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_008 계산
        """
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # 현재 high + low
        current_hl = self.add(high, low)
        
        # 지연된 high + low
        lag_high = self.delay(high, 1)
        lag_low = self.delay(low, 1)
        lag_hl = self.add(lag_high, lag_low)
        
        # 분모: 2 - (delay(high+low)) / 2 - 스칼라가 첫 번째이므로 직접 연산
        denominator = 2 - self.div(lag_hl, 2)
        
        # 첫 번째 비율
        ratio1 = self.div(current_hl, denominator)
        
        # 두 번째 비율: (high - low) / volume
        range_vol = self.div(self.sub(high, low), volume)
        
        # 곱하기
        product = self.mul(ratio1, range_vol)
        
        # SMA(7, 2) - 7일 SMA, 스무딩 팩터 2
        alpha = self.sma(product, 7, 2)
        
        return alpha.fillna(0)
