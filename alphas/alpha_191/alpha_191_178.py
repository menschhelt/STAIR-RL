import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_178(BaseAlpha):
    """
    alpha191_178: mul(rank(ts_corr({disk:vwap},{disk:volume},4)),rank(ts_corr(rank({disk:low}),rank(ts_mean({disk:volume},50)),12)))
    
    VWAP-거래량 상관관계와 저가-거래량 상관관계의 곱
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_178"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_178 계산
        """
        vwap = data['vwap']
        low = data['low']
        volume = data['volume']
        
        # 첫 번째 부분: rank(ts_corr(vwap, volume, 4))
        vwap_volume_corr = self.ts_corr(vwap, volume, 4)
        first_rank = self.rank(vwap_volume_corr)
        
        # 두 번째 부분: rank(ts_corr(rank(low), rank(ts_mean(volume, 50)), 12))
        low_rank = self.rank(low)
        volume_mean = self.ts_mean(volume, 50)
        volume_rank = self.rank(volume_mean)
        low_volume_corr = self.ts_corr(low_rank, volume_rank, 12)
        second_rank = self.rank(low_volume_corr)
        
        # 두 랭킹의 곱
        alpha = self.mul(first_rank, second_rank)
        
        return alpha.fillna(0.25)
