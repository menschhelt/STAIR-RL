import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_117(BaseAlpha):
    """
    alpha191_117: div(ts_sum(sub({disk:high},{disk:open}),20),mul(ts_sum(sub({disk:open},{disk:low}),20),100))
    
    20일간 상한 합계 / (하한 합계 * 100) = 상하한 비율
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_117"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_117 계산
        """
        high = data['high']
        open_price = data['open']
        low = data['low']
        
        # 상한: high - open
        upper_range = self.sub(high, open_price)
        
        # 하한: open - low
        lower_range = self.sub(open_price, low)
        
        # 20일 합계
        upper_sum = self.ts_sum(upper_range, 20)
        lower_sum = self.ts_sum(lower_range, 20)
        
        # 비율
        alpha = self.div(upper_sum, self.mul(lower_sum, 100))
        
        return alpha.fillna(0)
