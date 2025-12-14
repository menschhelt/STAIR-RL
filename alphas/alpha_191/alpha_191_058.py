import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_058(BaseAlpha):
    """
    alpha191_058: ts_sum(condition(eq({disk:close},delay({disk:close},1)),0,sub({disk:close},condition(gt({disk:close},delay({disk:close},1)),min({disk:low},delay({disk:close},1)),max({disk:high},delay({disk:close},1))))),20)
    
    20일간 조건부 가격 차이 합계 (002와 유사하지만 기간이 다름)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_058"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_058 계산
        """
        close = data['close']
        high = data['high']
        low = data['low']
        close_lag1 = self.delay(close, 1)
        
        # 내부 조건: close > delay(close, 1)이면 min(low, delay(close, 1))
        # 그렇지 않으면 max(high, delay(close, 1))
        inner_condition = self.condition(
            self.gt(close, close_lag1),
            self.min(low, close_lag1),
            self.max(high, close_lag1)
        )
        
        # 외부 조건: close == delay(close, 1)이면 0
        # 그렇지 않으면 close - inner_condition
        outer_condition = self.condition(
            self.eq(close, close_lag1),
            0,
            self.sub(close, inner_condition)
        )
        
        # 20일 합계
        alpha = self.ts_sum(outer_condition, 20)
        
        return alpha.fillna(0)
