import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_040(BaseAlpha):
    """
    alpha101_040: sub(pow(mul({disk:high},{disk:low}),0.5),{disk:vwap})
    
    고가와 저가의 기하평균에서 VWAP을 뺀 값
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_040"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_040 계산
        """
        # 1. mul(high, low)
        high_low_product = self.mul(data['high'], data['low'])
        
        # 2. pow(high_low_product, 0.5) - 기하평균
        geometric_mean = self.pow(high_low_product, 0.5)
        
        # 3. sub(geometric_mean, vwap)
        alpha = self.sub(geometric_mean, data['vwap'])
        
        return alpha.fillna(0)