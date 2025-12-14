import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_032(BaseAlpha):
    """
    alpha101_032: rank(mul(-1,pow(sub(1,div({disk:open},{disk:close})),1)))
    
    시가/종가 비율을 이용한 단순 모멘텀 팩터의 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_032"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_032 계산
        """
        # 1. div(open, close)
        open_close_ratio = self.div(data['open'], data['close'])
        
        # 2. sub(1, open_close_ratio) - 스칼라가 첫 번째이므로 직접 연산
        one_minus_ratio = 1 - open_close_ratio
        
        # 3. pow(..., 1) - 1제곱이므로 그대로
        powered = self.pow(one_minus_ratio, 1)
        
        # 4. mul(-1, powered)
        neg_powered = self.mul(powered, -1)
        
        # 5. rank(...)
        alpha = self.rank(neg_powered)
        
        return alpha.fillna(0)