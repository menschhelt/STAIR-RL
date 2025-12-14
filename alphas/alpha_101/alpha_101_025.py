import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_025(BaseAlpha):
    """
    alpha101_025: mul(-1,ts_max(ts_corr(ts_rank({disk:volume},5),ts_rank({disk:high},5),5),3))
    
    거래량과 고가의 5기간 랭킹 간 5기간 상관관계의 3기간 최대값의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_025"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_025 계산
        """
        # 1. ts_rank(volume, 5)
        volume_rank = self.ts_rank(data['volume'], 5)
        
        # 2. ts_rank(high, 5)
        high_rank = self.ts_rank(data['high'], 5)
        
        # 3. ts_corr(..., ..., 5)
        corr_5 = self.ts_corr(volume_rank, high_rank, 5)
        
        # 4. ts_max(..., 3)
        max_3 = self.ts_max(corr_5, 3)
        
        # 5. mul(-1, ...)
        alpha = self.mul(max_3, -1)
        
        return alpha.fillna(0)
