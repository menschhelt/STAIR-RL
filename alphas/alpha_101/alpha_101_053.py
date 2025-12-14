import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_053(BaseAlpha):
    """
    alpha101_053: div(mul(-1,mul(sub({disk:low},{disk:close}),pow({disk:open},5))),mul(sub({disk:low},{disk:high}),pow({disk:close},5)))
    
    저가-종가 차이와 시가의 5제곱, 저가-고가 차이와 종가의 5제곱의 비율
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_053"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_053 계산
        """
        # 분자 부분
        # 1. sub(low, close)
        low_close_diff = self.sub(data['low'], data['close'])
        
        # 2. pow(open, 5)
        open_power5 = self.pow(data['open'], 5)
        
        # 3. mul(low_close_diff, open_power5)
        numerator_product = self.mul(low_close_diff, open_power5)
        
        # 4. mul(-1, numerator_product)
        numerator = self.mul(numerator_product, -1)
        
        # 분모 부분
        # 5. sub(low, high)
        low_high_diff = self.sub(data['low'], data['high'])
        
        # 6. pow(close, 5)
        close_power5 = self.pow(data['close'], 5)
        
        # 7. mul(low_high_diff, close_power5)
        denominator = self.mul(low_high_diff, close_power5)
        
        # 8. div(numerator, denominator)
        alpha = self.div(numerator, denominator)
        
        return alpha.fillna(0)
