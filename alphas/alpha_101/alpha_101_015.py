import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_015(BaseAlpha):
    """
    alpha101_015: mul(-1,rank(ts_cov(rank({disk:high}),rank({disk:volume}),5)))
    
    고가와 거래량의 랭킹 간 5기간 공분산을 랭킹한 후 음수
    """
    # 기본 파라미터 정의
    default_params = {
        "cov_window": 5
    }

    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_015"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_015 계산
        """
        # 1. rank(high)와 rank(volume)
        high_rank = self.rank(data['high'])
        volume_rank = self.rank(data['volume'])
        
        # 2. ts_cov(..., 5)
        cov_5 = self.ts_cov(high_rank, volume_rank, self.params["cov_window"])
        
        # 3. rank(...)
        cov_rank = self.rank(cov_5)
        
        # 4. mul(-1, ...)
        alpha = self.mul(cov_rank, -1)
        
        return alpha.fillna(0)
