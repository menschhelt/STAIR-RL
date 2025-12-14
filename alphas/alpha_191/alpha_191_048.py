import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_048(BaseAlpha):
    """
    alpha191_048: div(ts_sum(condition(ge(add({disk:high},{disk:low}),add(delay({disk:high},1),delay({disk:low},1))),0,max(abs(sub({disk:high},delay({disk:high},1))),abs(sub({disk:low},delay({disk:low},1))))),12),add(ts_sum(condition(ge(add({disk:high},{disk:low}),add(delay({disk:high},1),delay({disk:low},1))),0,max(abs(sub({disk:high},delay({disk:high},1))),abs(sub({disk:low},delay({disk:low},1))))),12),ts_sum(condition(le(add({disk:high},{disk:low}),add(delay({disk:high},1),delay({disk:low},1))),0,max(abs(sub({disk:high},delay({disk:high},1))),abs(sub({disk:low},delay({disk:low},1))))),12)))
    
    복잡한 고가/저가 변화 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_048"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_048 계산
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
        
        # 조건부 값들
        # ge(hl_sum, hl_sum_lag1)이면 0, 그렇지 않으면 max_diff
        up_condition = self.condition(
            self.ge(hl_sum, hl_sum_lag1),
            0,
            max_diff
        )
        
        # le(hl_sum, hl_sum_lag1)이면 0, 그렇지 않으면 max_diff
        down_condition = self.condition(
            self.le(hl_sum, hl_sum_lag1),
            0,
            max_diff
        )
        
        # 12일 합계
        up_sum = self.ts_sum(up_condition, 12)
        down_sum = self.ts_sum(down_condition, 12)
        
        # 최종 비율
        alpha = self.div(up_sum, self.add(up_sum, down_sum))
        
        return alpha.fillna(0)
