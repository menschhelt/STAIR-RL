import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_014(BaseAlpha):
    """
    alpha101_014: mul(-1,ts_sum(rank(ts_corr(rank({disk:high}),rank({disk:volume}),3)),3))
    
    고가와 거래량의 랭킹 간 3기간 상관계수를 랭킹한 후 3기간 합계의 음수
    """
    # 기본 파라미터 정의
    default_params = {
        "corr_window": 3,
        "sum_window": 3
    }

    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_014"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_014 계산
        """
        # 1. rank(high)와 rank(volume)
        high_rank = self.rank(data['high'])
        volume_rank = self.rank(data['volume'])
        
        # 2. ts_corr(..., 3)
        corr_3 = self.ts_corr(high_rank, volume_rank, self.params["corr_window"])
        
        # 3. rank(...)
        corr_rank = self.rank(corr_3)
        
        # 4. ts_sum(..., 3)
        sum_3 = self.ts_sum(corr_rank, self.params["sum_window"])
        
        # 5. mul(-1, ...)
        alpha = self.mul(sum_3, -1)
        
        return alpha.fillna(0)
