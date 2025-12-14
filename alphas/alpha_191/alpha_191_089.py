import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_089(BaseAlpha):
    """
    alpha191_089: mul(rank(ts_corr(rank({disk:vwap}),rank({disk:volume}),5)),-1)
    
    -1 * VWAP와 거래량 랭킹의 5일 상관관계 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_089"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_089 계산
        """
        vwap = data['vwap']
        volume = data['volume']
        
        # 랭킹
        vwap_rank = self.rank(vwap)
        volume_rank = self.rank(volume)
        
        # 5일 상관관계
        corr = self.ts_corr(vwap_rank, volume_rank, 5)
        
        # 상관관계 랭킹
        corr_rank = self.rank(corr)
        
        # -1 곱하기
        alpha = self.mul(corr_rank, -1)
        
        return alpha.fillna(0)
