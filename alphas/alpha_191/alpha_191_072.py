import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_072(BaseAlpha):
    """
    alpha191_072: mul(sub(ts_rank(ts_decayed_linear(ts_decayed_linear(ts_corr({disk:close},{disk:volume},10),16),4),5),rank(ts_decayed_linear(ts_corr({disk:vwap},ts_mean({disk:volume},30),4),3))),-1)
    
    -1 * (이중 decayed linear 상관관계 ts_rank - VWAP-거래량 상관관계 decayed linear 랭킹)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_072"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_072 계산
        """
        close = data['close']
        volume = data['volume']
        vwap = data['vwap']
        
        # 첫 번째 부분: 이중 decayed linear
        corr1 = self.ts_corr(close, volume, 10)
        decay1_1 = self.ts_decayed_linear(corr1, 16)
        decay1_2 = self.ts_decayed_linear(decay1_1, 4)
        ts_rank1 = self.ts_rank(decay1_2, 5)
        
        # 두 번째 부분: VWAP와 30일 평균 거래량의 상관관계
        volume_mean_30 = self.ts_mean(volume, 30)
        corr2 = self.ts_corr(vwap, volume_mean_30, 4)
        decay2 = self.ts_decayed_linear(corr2, 3)
        rank2 = self.rank(decay2)
        
        # 차이
        diff = self.sub(ts_rank1, rank2)
        
        # -1 곱하기
        alpha = self.mul(diff, -1)
        
        return alpha.fillna(0)
