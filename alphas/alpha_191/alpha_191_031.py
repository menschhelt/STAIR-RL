import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_031(BaseAlpha):
    """
    alpha191_031: mul(-1,ts_sum(rank(ts_corr(rank({disk:high}),rank({disk:volume}),3)),3))
    
    -1 * 고가와 거래량 랭킹 상관관계 랭킹의 3일 합계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_031"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_031 계산
        """
        high = data['high']
        volume = data['volume']
        
        # 랭킹
        high_rank = self.rank(high)
        volume_rank = self.rank(volume)
        
        # 3일 상관관계
        corr = self.ts_corr(high_rank, volume_rank, 3)
        
        # 상관관계 랭킹
        corr_rank = self.rank(corr)
        
        # 3일 합계
        sum_corr_rank = self.ts_sum(corr_rank, 3)
        
        # -1 곱하기
        alpha = self.mul(-1, sum_corr_rank)
        
        return alpha.fillna(0)
