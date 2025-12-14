import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_049(BaseAlpha):
    """
    alpha101_049: mul(-1,ts_max(rank(ts_corr(rank({disk:volume}),rank({disk:vwap}),5)),5))
    
    거래량과 VWAP 랭킹 간 5기간 상관관계 랭킹의 5기간 최대값의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_049"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_049 계산
        """
        # 1. rank(volume)
        volume_rank = self.rank(data['volume'])
        
        # 2. rank(vwap)
        vwap_rank = self.rank(data['vwap'])
        
        # 3. ts_corr(volume_rank, vwap_rank, 5)
        corr_5 = self.ts_corr(volume_rank, vwap_rank, 5)
        
        # 4. rank(corr_5)
        corr_rank = self.rank(corr_5)
        
        # 5. ts_max(corr_rank, 5)
        max_5 = self.ts_max(corr_rank, 5)
        
        # 6. mul(-1, max_5)
        alpha = self.mul(max_5, -1)
        
        return alpha.fillna(0)
