import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_012(BaseAlpha):
    """
    alpha191_012: sub(pow(mul({disk:high},{disk:low}),0.5),{disk:vwap})
    
    sqrt(고가 * 저가) - VWAP = 기하평균 - VWAP
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_012"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_012 계산
        """
        high = data['high']
        low = data['low']
        vwap = data['vwap']
        
        # sqrt(high * low) - 기하평균
        geometric_mean = self.pow(self.mul(high, low), 0.5)
        
        # 기하평균 - VWAP
        alpha = self.sub(geometric_mean, vwap)
        
        return alpha.fillna(0)
