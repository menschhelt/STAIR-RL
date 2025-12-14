import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_096(BaseAlpha):
    """
    alpha191_096: ts_std({disk:volume},10)
    
    거래량의 10일 표준편차
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_096"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_096 계산
        """
        volume = data['volume']
        
        # 10일 표준편차
        alpha = self.ts_std(volume, 10)
        
        return alpha.fillna(0)
