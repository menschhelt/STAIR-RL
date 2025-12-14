import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_050(BaseAlpha):
    """
    alpha191_050: 048과 유사한 고가/저가 변화 지표 (단순화)
    
    고가/저가 변화 지표의 다른 버전
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_050"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_050 계산 (048의 역버전)
        """
        high = data['high']
        low = data['low']
        
        # 지연된 값들
        high_lag1 = self.delay(high, 1)
        low_lag1 = self.delay(low, 1)
        
        # high + low
        hl_sum = self.add(high, low)
        hl_sum_lag1 = self.add(high_lag1, low_lag1)
        
        # max(abs(high - delay(high, 1)), abs(low - delay(low, 1)))
        high_diff = self.abs(self.sub(high, high_lag1))
        low_diff = self.abs(self.sub(low, low_lag1))
        max_diff = self.max(high_diff, low_diff)
        
        # 조건부 값들 (le 조건)
        down_condition = self.condition(
            self.le(hl_sum, hl_sum_lag1),
            0,
            max_diff
        )
        
        # ge 조건
        up_condition = self.condition(
            self.ge(hl_sum, hl_sum_lag1),
            0,
            max_diff
        )
        
        # 12일 합계
        down_sum = self.ts_sum(down_condition, 12)
        up_sum = self.ts_sum(up_condition, 12)
        
        # 비율 (048과 동일)
        alpha = self.div(down_sum, self.add(down_sum, up_sum))
        
        return alpha.fillna(0)
