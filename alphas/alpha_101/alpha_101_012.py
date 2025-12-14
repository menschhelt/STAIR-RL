import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

# 부모 디렉토리를 import 경로에 추가
from alphas.base.base import BaseAlpha

class alpha_101_012(BaseAlpha):
    """
    Alpha 101_012: mul(-1,rank(ts_cov(rank({disk:close}),rank({disk:volume}),5)))
    
    종가-거래량 랭킹 간 공분산의 음의 랭킹
    """
    neutralizer_type: str = "mean"  # 평균 중립화

    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_012"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_012 계산
        """
        # 1. rank(close)
        close_rank = self.rank(data['close'])
        
        # 2. rank(volume)
        volume_rank = self.rank(data['volume'])
        
        # 3. ts_cov(rank(close), rank(volume), 5)
        cov_5 = self.ts_cov(close_rank, volume_rank, 5)

        # 4. rank(ts_cov(...))
        ranked_cov = self.rank(cov_5)

        # 5. -1 * rank(ts_cov(...))
        alpha = self.mul(ranked_cov, -1)
        
        return alpha.fillna(0)
