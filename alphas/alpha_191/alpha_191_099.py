import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_099(BaseAlpha):
    """
    alpha191_099: ts_std({disk:volume},20)
    
    거래량의 20일 표준편차 (096과 유사하지만 기간이 다름)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_099"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_099 계산
        """
        volume = data['volume']
        
        # 20일 표준편차
        alpha = self.ts_std(volume, 20)
        
        return alpha.fillna(0)
