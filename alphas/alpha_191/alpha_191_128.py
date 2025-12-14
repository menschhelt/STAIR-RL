import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_128(BaseAlpha):
    """
    alpha191_128: ts_sum(condition(lt(sub({disk:close},delay({disk:close},1)),0),abs(sub({disk:close},delay({disk:close},1))),0),12)
    
    12일간 하락일의 하락폭 누적합
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_128"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_128 계산
        """
        close = data['close']
        
        # 일일 수익률
        close_lag1 = self.delay(close, 1)
        daily_return = self.sub(close, close_lag1)
        
        # 하락일의 하락폭만 선택
        down_days = self.condition(
            self.lt(daily_return, 0),
            self.abs(daily_return),
            0
        )
        
        # 12일 합계
        alpha = self.ts_sum(down_days, 12)
        
        return alpha.fillna(0)
