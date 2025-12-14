import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_170(BaseAlpha):
    """
    alpha191_170: div(mul(-1,mul(sub({disk:low},{disk:close}),pow({disk:open},5))),mul(sub({disk:close},{disk:high}),pow({disk:close},5)))
    
    가격 관계의 거듭제곱 비율 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_170"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_170 계산
        """
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 분자: -1 * (low - close) * open^5
        low_close_diff = self.sub(low, close)
        open_power = self.pow(open_price, 5)
        numerator = self.mul(-1, self.mul(low_close_diff, open_power))
        
        # 분모: (close - high) * close^5
        close_high_diff = self.sub(close, high)
        close_power = self.pow(close, 5)
        denominator = self.mul(close_high_diff, close_power)
        
        # 비율 계산 (0으로 나누기 방지)
        alpha = self.div(numerator, denominator.replace(0, 1))
        
        return alpha.fillna(0)
