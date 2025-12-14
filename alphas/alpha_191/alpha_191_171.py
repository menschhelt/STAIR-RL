import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_171(BaseAlpha):
    """
    alpha191_171: ts_mean(div(abs(mul(ts_sum(condition(...)),div(100,sub(...)))),mul(mul(ts_sum(...),div(100,add(...))),100)),6)
    
    매우 복잡한 조건부 True Range 기반 지표의 6일 평균 (단순화)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_171"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_171 계산 (매우 복잡한 식을 단순화)
        """
        high = data['high']
        low = data['low']
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # 간단한 True Range 계산
        range1 = self.sub(high, low)
        range2 = self.abs(self.sub(high, close_lag1))
        range3 = self.abs(self.sub(low, close_lag1))
        true_range = self.max(self.max(range1, range2), range3)
        
        # 가격 변화 방향 (상승/하락)
        price_delta_high = self.ts_delta(high, 1)
        price_delta_low = self.ts_delta(low, 1)
        
        # 조건부 합계 (상승/하락에 따른 가중)
        up_condition = self.gt(price_delta_low, 0)
        down_condition = self.gt(price_delta_high, 0)
        
        up_sum = self.ts_sum(self.condition(up_condition, true_range, 0), 14)
        down_sum = self.ts_sum(self.condition(down_condition, true_range, 0), 14)
        
        # 비율 계산 (단순화)
        ratio = self.div(up_sum, self.add(up_sum, down_sum))
        
        # 6일 평균
        alpha = self.ts_mean(ratio, 6)
        
        return alpha.fillna(0.5)
