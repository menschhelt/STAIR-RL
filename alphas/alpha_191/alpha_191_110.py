import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_110(BaseAlpha):
    """
    alpha191_110: sub(sma(mul({disk:volume},div(sub(sub({disk:close},{disk:low}),sub({disk:high},{disk:close})),sub({disk:high},{disk:low}))),11,2),sma(mul({disk:volume},div(sub(sub({disk:close},{disk:low}),sub({disk:high},{disk:close})),sub({disk:high},{disk:low}))),4,2))
    
    거래량 가중 Williams %R의 11일 SMA - 4일 SMA
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_110"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_110 계산
        """
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Williams %R 계산: (close - low) - (high - close) / (high - low)
        numerator = self.sub(
            self.sub(close, low),
            self.sub(high, close)
        )
        denominator = self.sub(high, low)
        williams_r = self.div(numerator, denominator)
        
        # 거래량 가중
        volume_weighted = self.mul(volume, williams_r)
        
        # 11일과 4일 SMA
        sma_11 = self.sma(volume_weighted, 11, 2)
        sma_4 = self.sma(volume_weighted, 4, 2)
        
        # 차이
        alpha = self.sub(sma_11, sma_4)
        
        return alpha.fillna(0)
