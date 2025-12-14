import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_043(BaseAlpha):
    """
    alpha101_043: mul(-1,ts_corr({disk:high},rank({disk:volume}),5))
    
    고가와 거래량 랭킹의 5기간 상관관계의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_043"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_043 계산
        """
        # 1. rank(volume)
        volume_rank = self.rank(data['volume'])
        
        # 2. ts_corr(high, volume_rank, 5)
        corr_5 = self.ts_corr(data['high'], volume_rank, 5)
        
        # 3. mul(-1, corr_5)
        alpha = self.mul(corr_5, -1)
        
        return alpha.fillna(0)
