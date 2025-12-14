import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_186(BaseAlpha):
    """
    alpha191_186: ts_sum(condition(le({disk:open},delay({disk:open},1)),0,max(sub({disk:high},{disk:open}),sub({disk:open},delay({disk:open},1)))),20)
    
    시가가 전일 시가보다 높을 때의 갭업과 고가-시가 차이의 최대값을 20일 합계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_186"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_186 계산
        """
        open_price = data['open']
        high = data['high']
        open_lag1 = self.delay(open_price, 1)
        
        # 조건: 시가 <= 전일 시가
        condition = self.le(open_price, open_lag1)
        
        # 시가가 전일 시가보다 높을 때의 값들
        gap_up = self.sub(open_price, open_lag1)  # 갭업 크기
        high_open_diff = self.sub(high, open_price)  # 고가-시가 차이
        
        # 두 값의 최대값
        max_movement = self.max(high_open_diff, gap_up)
        
        # 조건: 시가가 낮거나 같으면 0, 높으면 최대값
        conditional_value = self.condition(condition, 0, max_movement)
        
        # 20일 합계
        alpha = self.ts_sum(conditional_value, 20)
        
        return alpha.fillna(0)
