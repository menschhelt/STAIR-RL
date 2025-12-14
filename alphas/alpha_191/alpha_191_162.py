import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_162(BaseAlpha):
    """
    alpha191_162: rank(mul(mul(mul(mul(-1,{disk:returns}),ts_mean({disk:volume},20)),{disk:vwap}),sub({disk:high},{disk:close})))
    
    음의 수익률, 거래량, VWAP, 고가-종가 차이의 곱에 대한 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_162"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_162 계산
        """
        close = data['close']
        high = data['high']
        volume = data['volume']
        vwap = data['vwap']
        
        # 수익률 계산
        returns = self.div(self.sub(close, self.delay(close, 1)), self.delay(close, 1))
        
        # 음의 수익률
        neg_returns = self.mul(-1, returns)
        
        # 20일 평균 거래량
        volume_mean = self.ts_mean(volume, 20)
        
        # 고가-종가 차이
        high_close_diff = self.sub(high, close)
        
        # 모든 요소의 곱
        product = self.mul(self.mul(self.mul(neg_returns, volume_mean), vwap), high_close_diff)
        
        # 랭킹
        alpha = self.rank(product)
        
        return alpha.fillna(0.5)
