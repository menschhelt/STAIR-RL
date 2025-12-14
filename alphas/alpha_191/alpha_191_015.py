import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_015(BaseAlpha):
    """
    alpha191_015: mul(-1,ts_max(rank(ts_corr(rank({disk:volume}),rank({disk:vwap}),5)),5))
    
    -1 * 거래량과 VWAP 랭킹 상관관계의 5일 최대값
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_015"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_015 계산
        """
        volume = data['volume']
        vwap = data['vwap']
        
        # 랭킹
        volume_rank = self.rank(volume)
        vwap_rank = self.rank(vwap)
        
        # 5일 상관관계
        corr = self.ts_corr(volume_rank, vwap_rank, 5)
        
        # 상관관계 랭킹
        corr_rank = self.rank(corr)
        
        # 5일 최대값
        max_corr_rank = self.ts_max(corr_rank, 5)
        
        # -1 곱하기
        alpha = self.mul(-1, max_corr_rank)
        
        return alpha.fillna(0)
