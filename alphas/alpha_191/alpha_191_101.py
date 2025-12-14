import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_101(BaseAlpha):
    """
    alpha191_101: div(sma(max(sub({disk:volume},delay({disk:volume},1)),0),6,1),mul(sma(abs(sub({disk:volume},delay({disk:volume},1))),6,1),100))
    
    6일 거래량 증가분 SMA / (6일 거래량 절대변화 SMA * 100) = 거래량 상승 비율
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_101"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_101 계산
        """
        volume = data['volume']
        volume_lag1 = self.delay(volume, 1)
        
        # 일일 거래량 변화
        volume_change = self.sub(volume, volume_lag1)
        
        # 증가분만
        volume_increase = self.max(volume_change, 0)
        
        # 절대 변화
        abs_volume_change = self.abs(volume_change)
        
        # 6일 SMA
        increase_sma = self.sma(volume_increase, 6, 1)
        abs_sma = self.sma(abs_volume_change, 6, 1)
        
        # 비율
        alpha = self.div(increase_sma, self.mul(abs_sma, 100))
        
        return alpha.fillna(0)
