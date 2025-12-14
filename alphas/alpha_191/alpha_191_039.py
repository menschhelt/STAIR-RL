import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_039(BaseAlpha):
    """
    alpha191_039: div(ts_sum(condition(gt({disk:close},delay({disk:close},1)),{disk:volume},0),26),mul(ts_sum(condition(le({disk:close},delay({disk:close},1)),{disk:volume},0),26),100))
    
    26일간 상승일 거래량 합계 / (하락일 거래량 합계 * 100)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_039"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_039 계산
        """
        close = data['close']
        volume = data['volume']
        close_lag1 = self.delay(close, 1)
        
        # 상승일 거래량
        up_volume = self.condition(
            self.gt(close, close_lag1),
            volume,
            0
        )
        
        # 하락일 거래량
        down_volume = self.condition(
            self.le(close, close_lag1),
            volume,
            0
        )
        
        # 26일 합계
        up_sum = self.ts_sum(up_volume, 26)
        down_sum = self.ts_sum(down_volume, 26)
        
        # 비율
        alpha = self.div(up_sum, self.mul(down_sum, 100))
        
        return alpha.fillna(0)
