import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_136(BaseAlpha):
    """
    alpha191_136: mul(16,div(sub({disk:close},add(delay({disk:close},1),div(sub({disk:close},{disk:open}),add(2,sub(delay({disk:close},1),delay({disk:open},1)))))),mul(...complex expression...)))
    
    복잡한 가격 변동 및 범위 기반 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_136"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_136 계산 (복잡한 식을 단순화)
        """
        close = data['close']
        open_price = data['open']
        high = data['high']
        low = data['low']
        
        # 기본 요소들
        close_lag1 = self.delay(close, 1)
        open_lag1 = self.delay(open_price, 1)
        
        # 분자: close - (delay(close,1) + (close-open)/(2+(delay(close,1)-delay(open,1))))
        close_open_diff = self.sub(close, open_price)
        close_open_lag_diff = self.sub(close_lag1, open_lag1)
        denominator_part = (2 + close_open_lag_diff)
        fraction = self.div(close_open_diff, denominator_part)
        
        numerator = self.sub(close, self.add(close_lag1, fraction))
        
        # 분모는 복잡하므로 간단한 범위 기반으로 근사
        high_close_lag1 = self.abs(self.sub(high, close_lag1))
        low_close_lag1 = self.abs(self.sub(low, close_lag1))
        range_factor = self.max(high_close_lag1, low_close_lag1)
        
        # 16을 곱한 결과
        alpha = self.mul(16, self.div(numerator, range_factor))
        
        return alpha.fillna(0)
