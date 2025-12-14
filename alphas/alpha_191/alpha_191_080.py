import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_080(BaseAlpha):
    """
    alpha191_080: sma({disk:volume},21,2)
    
    거래량의 21일 SMA (스무딩 팩터 2)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_080"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_080 계산
        """
        volume = data['volume']
        
        # 21일 SMA, 스무딩 팩터 2
        alpha = self.sma(volume, 21, 2)
        
        return alpha.fillna(0)
