import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_190(BaseAlpha):
    """
    alpha191_190: sub(add(ts_corr(ts_mean({disk:volume},20),{disk:low},5),div(add({disk:high},{disk:low}),2)),{disk:close})
    
    거래량-저가 상관관계 + 중간가 - 종가
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_190"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_190 계산
        """
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # 첫 번째 부분: ts_corr(ts_mean(volume, 20), low, 5)
        volume_mean = self.ts_mean(volume, 20)
        volume_low_corr = self.ts_corr(volume_mean, low, 5)
        
        # 두 번째 부분: (high + low) / 2 (중간가)
        mid_price = self.div(self.add(high, low), 2)
        
        # 첫 번째와 두 번째 부분의 합
        sum_part = self.add(volume_low_corr, mid_price)
        
        # 종가를 빼기
        alpha = self.sub(sum_part, close)
        
        return alpha.fillna(0)
