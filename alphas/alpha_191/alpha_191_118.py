import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_118(BaseAlpha):
    """
    alpha191_118: sub(rank(ts_decayed_linear(ts_corr({disk:vwap},ts_sum(ts_mean({disk:volume},5),26),5),7)),rank(ts_decayed_linear(ts_rank(min(ts_corr(rank({disk:open}),rank(ts_mean({disk:volume},15)),21),9),7),8)))
    
    두 개의 복잡한 decayed linear 랭킹의 차이
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_118"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_118 계산
        """
        vwap = data['vwap']
        volume = data['volume']
        open_price = data['open']
        
        # 첫 번째 부분: VWAP와 복잡한 거래량 지표의 상관관계
        volume_mean_5 = self.ts_mean(volume, 5)
        volume_sum = self.ts_sum(volume_mean_5, 26)
        corr1 = self.ts_corr(vwap, volume_sum, 5)
        decay1 = self.ts_decayed_linear(corr1, 7)
        rank1 = self.rank(decay1)
        
        # 두 번째 부분: 복잡한 시가-거래량 상관관계
        open_rank = self.rank(open_price)
        volume_mean_15 = self.ts_mean(volume, 15)
        volume_mean_rank = self.rank(volume_mean_15)
        corr2 = self.ts_corr(open_rank, volume_mean_rank, 21)
        min_corr = self.ts_min(corr2, 9)
        ts_rank2 = self.ts_rank(min_corr, 7)
        decay2 = self.ts_decayed_linear(ts_rank2, 8)
        rank2 = self.rank(decay2)
        
        # 차이
        alpha = self.sub(rank1, rank2)
        
        return alpha.fillna(0)
