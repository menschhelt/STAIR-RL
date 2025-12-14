import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_049(BaseAlpha):
    """
    alpha191_049: div(ts_sum(condition(le(add({disk:high},{disk:low}),add(delay({disk:high},1),delay({disk:low},1))),0,max(abs(sub({disk:high},delay({disk:high},1))),abs(sub({disk:low},delay({disk:low},1))))),12),sub(add(ts_sum(condition(le(add({disk:high},{disk:low}),add(delay({disk:high},1),delay({disk:low},1))),0,max(abs(sub({disk:high},delay({disk:high},1))),abs(sub({disk:low},delay({disk:low},1))))),12),ts_sum(condition(ge(add({disk:high},{disk:low}),add(delay({disk:high},1),delay({disk:low},1))),0,max(abs(sub({disk:high},delay({disk:high},1))),abs(sub({disk:low},delay({disk:low},1))))),12)),div(ts_sum(condition(ge(add({disk:high},{disk:low}),add(delay({disk:high},1),delay({disk:low},1))),0,max(abs(sub({disk:high},delay({disk:high},1))),abs(sub({disk:low},delay({disk:low},1))))),12),add(ts_sum(condition(ge(add({disk:high},{disk:low}),add(delay({disk:high},1),delay({disk:low},1))),0,max(abs(sub({disk:high},delay({disk:high},1))),abs(sub({disk:low},delay({disk:low},1))))),12),ts_sum(condition(le(add({disk:high},{disk:low}),add(delay({disk:high},1),delay({disk:low},1))),0,max(abs(sub({disk:high},delay({disk:high},1))),abs(sub({disk:low},delay({disk:low},1))))),12)))))
    
    매우 복잡한 고가/저가 변화 지표 - 048과 유사하지만 더 복잡한 분모
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_049"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_049 계산
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
        
        # 조건부 값들 (049에서는 le 조건)
        down_condition = self.condition(
            self.le(hl_sum, hl_sum_lag1),
            0,
            max_diff
        )
        
        # ge 조건 (048에서와 같음)
        up_condition = self.condition(
            self.ge(hl_sum, hl_sum_lag1),
            0,
            max_diff
        )
        
        # 12일 합계
        down_sum = self.ts_sum(down_condition, 12)
        up_sum = self.ts_sum(up_condition, 12)
        
        # 복잡한 분모 계산
        # (down_sum + up_sum) - (up_sum / (up_sum + down_sum))
        total_sum = self.add(down_sum, up_sum)
        up_ratio = self.div(up_sum, total_sum)
        denominator = self.sub(total_sum, up_ratio)
        
        # 최종 계산
        alpha = self.div(down_sum, denominator)
        
        return alpha.fillna(0)
