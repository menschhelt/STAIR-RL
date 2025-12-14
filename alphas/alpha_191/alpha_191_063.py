import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_063(BaseAlpha):
    """
    alpha191_063: mul(max(rank(ts_decayed_linear(ts_corr(rank({disk:vwap}),rank({disk:volume}),4),4)),rank(ts_decayed_linear(max(ts_corr(rank({disk:close}),rank(ts_mean({disk:volume},60)),4),13),14))),-1)
    
    -1 * max(두 개의 복잡한 상관관계 지표)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_063"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_063 계산
        """
        vwap = data['vwap']
        volume = data['volume']
        close = data['close']
        
        # 첫 번째 부분: VWAP와 거래량 랭킹의 상관관계
        vwap_rank = self.rank(vwap)
        volume_rank = self.rank(volume)
        corr1 = self.ts_corr(vwap_rank, volume_rank, 4)
        decay1 = self.ts_decayed_linear(corr1, 4)
        rank1 = self.rank(decay1)
        
        # 두 번째 부분: 종가와 60일 평균 거래량 랭킹의 상관관계
        volume_mean_60 = self.ts_mean(volume, 60)
        volume_mean_rank = self.rank(volume_mean_60)
        close_rank = self.rank(close)
        corr2 = self.ts_corr(close_rank, volume_mean_rank, 4)
        max_corr2 = self.ts_max(corr2, 13)
        decay2 = self.ts_decayed_linear(max_corr2, 14)
        rank2 = self.rank(decay2)
        
        # 최대값
        max_rank = self.max(rank1, rank2)
        
        # -1 곱하기
        alpha = self.mul(max_rank, -1)
        
        return alpha.fillna(0)
