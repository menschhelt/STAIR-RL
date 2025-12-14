import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_004(BaseAlpha):
    """
    alpha191_004: mul(-1,ts_max(ts_corr(ts_rank({disk:volume},5),ts_rank({disk:high},5),5),3))
    
    -1 * 거래량과 고가의 랭킹 상관관계의 3일 최대값
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_004"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_004 계산
        """
        # 5일 시계열 랭킹
        volume_rank = self.ts_rank(data['volume'], 5)
        high_rank = self.ts_rank(data['high'], 5)
        
        # 5일 상관관계
        corr = self.ts_corr(volume_rank, high_rank, 5)
        
        # 3일 최대값
        max_corr = self.ts_max(corr, 3)
        
        # -1 곱하기
        alpha = self.mul(-1, max_corr)
        
        return alpha.fillna(0)
