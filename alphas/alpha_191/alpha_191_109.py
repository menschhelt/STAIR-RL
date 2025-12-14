import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_109(BaseAlpha):
    """
    alpha191_109: div(ts_sum(max(0,sub({disk:high},delay({disk:close},1))),20),mul(ts_sum(max(0,sub(delay({disk:close},1),{disk:low})),20),100))
    
    20일간 상향 갭 합계 / (하향 갭 합계 * 100) = 갭 비율
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_109"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_109 계산
        """
        high = data['high']
        low = data['low']
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # 상향 갭: max(0, high - delay(close, 1))
        up_gap = self.max(0, self.sub(high, close_lag1))
        
        # 하향 갭: max(0, delay(close, 1) - low)
        down_gap = self.max(0, self.sub(close_lag1, low))
        
        # 20일 합계
        up_gap_sum = self.ts_sum(up_gap, 20)
        down_gap_sum = self.ts_sum(down_gap, 20)
        
        # 비율
        alpha = self.div(up_gap_sum, self.mul(down_gap_sum, 100))
        
        return alpha.fillna(0)
