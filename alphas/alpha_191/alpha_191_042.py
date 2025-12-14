import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_042(BaseAlpha):
    """
    alpha191_042: ts_sum(condition(gt({disk:close},delay({disk:close},1)),{disk:volume},condition(lt({disk:close},delay({disk:close},1)),neg({disk:volume}),0)),6)
    
    6일간 거래량의 방향성 합계 (상승일은 +, 하락일은 -, 횡보일은 0)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_042"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_042 계산
        """
        close = data['close']
        volume = data['volume']
        close_lag1 = self.delay(close, 1)
        
        # 내부 조건: close < delay(close, 1)이면 -volume, 그렇지 않으면 0
        inner_condition = self.condition(
            self.lt(close, close_lag1),
            self.neg(volume),  # -volume
            0
        )
        
        # 외부 조건: close > delay(close, 1)이면 volume, 그렇지 않으면 inner_condition
        directional_volume = self.condition(
            self.gt(close, close_lag1),
            volume,
            inner_condition
        )
        
        # 6일 합계
        alpha = self.ts_sum(directional_volume, 6)
        
        return alpha.fillna(0)
