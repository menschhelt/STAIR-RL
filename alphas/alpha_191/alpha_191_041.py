import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_041(BaseAlpha):
    """
    alpha191_041: mul(mul(-1,rank(ts_std({disk:high},10))),ts_corr({disk:high},{disk:volume},10))
    
    -1 * 고가 표준편차 랭킹 * 고가와 거래량의 10일 상관관계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_041"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_041 계산
        """
        high = data['high']
        volume = data['volume']
        
        # 10일 고가 표준편차
        high_std = self.ts_std(high, 10)
        high_std_rank = self.rank(high_std)
        
        # 고가와 거래량의 10일 상관관계
        corr = self.ts_corr(high, volume, 10)
        
        # 최종 계산
        alpha = self.mul(
            self.mul(-1, high_std_rank),
            corr
        )
        
        return alpha.fillna(0)
