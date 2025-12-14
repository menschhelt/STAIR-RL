import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_184(BaseAlpha):
    """
    alpha191_184: rank(mul(-1,pow(sub(1,div({disk:open},{disk:close})),2)))
    
    -1 * (1 - 시가/종가)^2의 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_184"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_184 계산
        """
        open_price = data['open']
        close = data['close']
        
        # 시가/종가 비율
        open_close_ratio = self.div(open_price, close)
        
        # 1 - 시가/종가
        ratio_diff = (1 - open_close_ratio)
        
        # 제곱
        ratio_squared = self.pow(ratio_diff, 2)
        
        # -1 곱하기
        negative_ratio = self.mul(-1, ratio_squared)
        
        # 랭킹
        alpha = self.rank(negative_ratio)
        
        return alpha.fillna(0.5)
