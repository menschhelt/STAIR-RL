import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_138(BaseAlpha):
    """
    alpha191_138: mul(-1,ts_corr({disk:open},{disk:volume},10))
    
    -1 * 시가와 거래량의 10일 상관관계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_138"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_138 계산
        """
        # 시가와 거래량의 10일 상관관계
        open_volume_corr = self.ts_corr(data['open'], data['volume'], 10)
        
        # -1 곱하기
        alpha = self.mul(-1, open_volume_corr)
        
        return alpha.fillna(0)
