import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_082(BaseAlpha):
    """
    alpha191_082: mul(-1,rank(ts_cov(rank({disk:high}),rank({disk:volume}),5)))
    
    -1 * 고가와 거래량 랭킹의 5일 공분산 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_082"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_082 계산
        """
        high = data['high']
        volume = data['volume']
        
        # 랭킹
        high_rank = self.rank(high)
        volume_rank = self.rank(volume)
        
        # 5일 공분산
        cov = self.ts_cov(high_rank, volume_rank, 5)
        
        # 공분산 랭킹
        cov_rank = self.rank(cov)
        
        # -1 곱하기
        alpha = self.mul(-1, cov_rank)
        
        return alpha.fillna(0)
