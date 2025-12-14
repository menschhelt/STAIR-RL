import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_131(BaseAlpha):
    """
    alpha191_131: ts_mean({disk:amount},20)
    
    20일 거래대금 평균
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_131"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_131 계산
        """
        # 거래대금 계산 (volume * close)
        close = data['close']
        volume = data['volume']
        amount = self.mul(volume, close)
        
        # 20일 평균
        alpha = self.ts_mean(amount, 20)
        
        return alpha.fillna(0)
