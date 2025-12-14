import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_120(BaseAlpha):
    """
    alpha191_120: mul(pow(rank(sub({disk:vwap},min({disk:vwap},12))),ts_rank(ts_corr(ts_rank({disk:vwap},20),ts_rank(ts_mean({disk:volume},60),2),18),3)),-1)
    
    -1 * (VWAP 상대적 위치 랭킹^복잡한 상관관계 ts_rank)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_120"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_120 계산
        """
        vwap = data['vwap']
        volume = data['volume']
        
        # VWAP - 12일 최소 VWAP
        vwap_diff = self.sub(vwap, self.ts_min(vwap, 12))
        base_rank = self.rank(vwap_diff)
        
        # 복잡한 상관관계
        vwap_ts_rank = self.ts_rank(vwap, 20)
        volume_mean_60 = self.ts_mean(volume, 60)
        volume_ts_rank = self.ts_rank(volume_mean_60, 2)
        corr = self.ts_corr(vwap_ts_rank, volume_ts_rank, 18)
        exp_rank = self.ts_rank(corr, 3)
        
        # 거듭제곱
        power_result = self.pow(base_rank, exp_rank)
        
        # -1 곱하기
        alpha = self.mul(power_result, -1)
        
        return alpha.fillna(0)
