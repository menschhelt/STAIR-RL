import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_068(BaseAlpha):
    """
    alpha191_068: 극도로 복잡한 시가 기반 조건부 로직 (단순화)
    
    시가 변화 기반 조건부 지표의 단순화
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_068"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_068 계산 (단순화)
        """
        open_price = data['open']
        high = data['high']
        low = data['low']
        
        # 시가의 변화
        open_lag1 = self.delay(open_price, 1)
        open_change = self.sub(open_price, open_lag1)
        
        # 시가가 하락했을 때와 상승했을 때의 다른 지표
        # 하락: high - open vs open - open_lag1의 최대값
        down_indicator = self.max(
            self.sub(high, open_price),
            self.sub(open_price, open_lag1)
        )
        
        # 상승: open - low vs open - open_lag1의 최대값  
        up_indicator = self.max(
            self.sub(open_price, low),
            self.sub(open_price, open_lag1)
        )
        
        # 조건부 선택
        condition_value = self.condition(
            self.le(open_price, open_lag1),
            down_indicator,
            up_indicator
        )
        
        # 20일 합계
        down_sum = self.ts_sum(
            self.condition(
                self.le(open_price, open_lag1),
                condition_value,
                0
            ),
            20
        )
        
        up_sum = self.ts_sum(
            self.condition(
                self.ge(open_price, open_lag1),
                condition_value,
                0
            ),
            20
        )
        
        # 조건부 반환
        alpha = self.condition(
            self.gt(down_sum, up_sum),
            self.div(self.sub(down_sum, up_sum), down_sum),
            self.condition(
                self.eq(down_sum, up_sum),
                0,
                self.div(self.sub(down_sum, up_sum), up_sum)
            )
        )
        
        return alpha.fillna(0)
