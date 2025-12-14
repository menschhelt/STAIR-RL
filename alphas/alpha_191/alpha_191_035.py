import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_035(BaseAlpha):
    """
    alpha191_035: rank(ts_sum(ts_corr(rank({disk:volume}),rank({disk:vwap}),6),6))
    
    거래량과 VWAP 랭킹 상관관계의 6일 합계에 대한 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_035"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_035 계산
        """
        volume = data['volume']
        vwap = data['vwap']
        
        # 랭킹
        volume_rank = self.rank(volume)
        vwap_rank = self.rank(vwap)
        
        # 6일 상관관계
        corr = self.ts_corr(volume_rank, vwap_rank, 6)
        
        # 6일 합계
        sum_corr = self.ts_sum(corr, 6)
        
        # 랭킹
        alpha = self.rank(sum_corr)
        
        return alpha.fillna(0)
