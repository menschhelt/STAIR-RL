import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_060(BaseAlpha):
    """
    alpha191_060: mul(max(rank(ts_decayed_linear(ts_delta({disk:vwap},1),12)),rank(ts_decayed_linear(rank(ts_corr({disk:low},ts_mean({disk:volume},80),8)),17))),-1)
    
    -1 * max(VWAP 변화 decayed linear 랭킹, 저가-거래량 상관관계 decayed linear 랭킹)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_060"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_060 계산
        """
        vwap = data['vwap']
        low = data['low']
        volume = data['volume']
        
        # 첫 번째 부분: VWAP 1일 변화의 12일 decayed linear
        vwap_delta = self.ts_delta(vwap, 1)
        decay1 = self.ts_decayed_linear(vwap_delta, 12)
        rank1 = self.rank(decay1)
        
        # 두 번째 부분: 저가와 80일 평균 거래량의 8일 상관관계
        volume_mean_80 = self.ts_mean(volume, 80)
        corr = self.ts_corr(low, volume_mean_80, 8)
        corr_rank = self.rank(corr)
        decay2 = self.ts_decayed_linear(corr_rank, 17)
        rank2 = self.rank(decay2)
        
        # 최대값
        max_rank = self.max(rank1, rank2)
        
        # -1 곱하기
        alpha = self.mul(max_rank, -1)
        
        return alpha.fillna(0)
