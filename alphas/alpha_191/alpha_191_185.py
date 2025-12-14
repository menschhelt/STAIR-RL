import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_185(BaseAlpha):
    """
    alpha191_185: div(add(ts_mean(...),delay(ts_mean(...),6)),2)
    
    매우 복잡한 조건부 지표의 6일 지연과의 평균 (단순화)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_185"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_185 계산 (매우 복잡한 식을 단순화)
        """
        high = data['high']
        low = data['low']
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # 간단한 True Range 기반 지표로 단순화
        range1 = self.sub(high, low)
        range2 = self.abs(self.sub(high, close_lag1))
        range3 = self.abs(self.sub(low, close_lag1))
        true_range = self.max(self.max(range1, range2), range3)
        
        # 상승/하락 조건부 평균
        price_change = self.ts_delta(low, 1)
        up_condition = self.gt(price_change, 0)
        
        # 조건부 True Range
        conditional_tr = self.condition(up_condition, true_range, 0)
        
        # 6일 평균
        current_mean = self.ts_mean(conditional_tr, 6)
        delayed_mean = self.delay(current_mean, 6)
        
        # 두 평균의 평균
        alpha = self.div(self.add(current_mean, delayed_mean), 2)
        
        return alpha.fillna(0)
