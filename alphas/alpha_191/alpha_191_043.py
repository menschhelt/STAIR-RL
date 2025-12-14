import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_043(BaseAlpha):
    """
    alpha191_043: add(ts_rank(ts_decayed_linear(ts_corr({disk:low},ts_mean({disk:volume},10),7),6),4),ts_rank(ts_decayed_linear(ts_delta({disk:vwap},3),10),15))
    
    두 개의 ts_rank 값의 합계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_043"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_043 계산
        """
        low = data['low']
        volume = data['volume']
        vwap = data['vwap']
        
        # 첫 번째 부분
        volume_mean_10 = self.ts_mean(volume, 10)
        corr = self.ts_corr(low, volume_mean_10, 7)
        decay1 = self.ts_decayed_linear(corr, 6)
        rank1 = self.ts_rank(decay1, 4)
        
        # 두 번째 부분
        vwap_delta = self.ts_delta(vwap, 3)
        decay2 = self.ts_decayed_linear(vwap_delta, 10)
        rank2 = self.ts_rank(decay2, 15)
        
        # 합계
        alpha = self.add(rank1, rank2)
        
        return alpha.fillna(0)
