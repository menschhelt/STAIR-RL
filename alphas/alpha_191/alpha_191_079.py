import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_079(BaseAlpha):
    """
    alpha191_079: div(sub({disk:volume},delay({disk:volume},5)),mul(delay({disk:volume},5),100))
    
    5일 거래량 변화율 (%) = (현재 거래량 - 5일 전 거래량) / (5일 전 거래량 * 100)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_079"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_079 계산
        """
        volume = data['volume']
        volume_lag5 = self.delay(volume, 5)
        
        # 5일 거래량 변화
        volume_change = self.sub(volume, volume_lag5)
        
        # 분모: delay(volume, 5) * 100
        denominator = self.mul(volume_lag5, 100)
        
        alpha = self.div(volume_change, denominator)
        
        return alpha.fillna(0)
