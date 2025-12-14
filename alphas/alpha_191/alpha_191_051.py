import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_051(BaseAlpha):
    """
    alpha191_051: div(ts_sum(max(0,sub({disk:high},delay(div(add({disk:high},add({disk:low},{disk:close})),3),1))),26),mul(ts_sum(max(0,sub(delay(div(add({disk:high},add({disk:low},{disk:close})),3),1),{disk:low})),26),100))
    
    HLC 평균 기준 고가/저가 변화 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_051"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_051 계산
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # HLC 평균: (high + low + close) / 3
        hlc_avg = self.div(
            self.add(high, self.add(low, close)),
            3
        )
        
        # 1일 지연된 HLC 평균
        hlc_avg_lag1 = self.delay(hlc_avg, 1)
        
        # max(0, high - delay(hlc_avg, 1))
        high_diff = self.max(0, self.sub(high, hlc_avg_lag1))
        
        # max(0, delay(hlc_avg, 1) - low)
        low_diff = self.max(0, self.sub(hlc_avg_lag1, low))
        
        # 26일 합계
        high_sum = self.ts_sum(high_diff, 26)
        low_sum = self.ts_sum(low_diff, 26)
        
        # 비율
        alpha = self.div(high_sum, self.mul(low_sum, 100))
        
        return alpha.fillna(0)
