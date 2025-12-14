import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_018(BaseAlpha):
    """
    alpha191_018: condition(lt({disk:close},delay({disk:close},5)),div(sub({disk:close},delay({disk:close},5)),delay({disk:close},5)),condition(eq({disk:close},delay({disk:close},5)),0,div(sub({disk:close},delay({disk:close},5)),{disk:close})))
    
    조건부 5일 수익률 계산
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_018"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_018 계산
        """
        close = data['close']
        close_lag5 = self.delay(close, 5)
        
        # 가격 변화
        price_change = self.sub(close, close_lag5)
        
        # 내부 조건: close == delay(close, 5)이면 0
        # 그렇지 않으면 price_change / close
        inner_condition = self.condition(
            self.eq(close, close_lag5),
            0,
            self.div(price_change, close)
        )
        
        # 외부 조건: close < delay(close, 5)이면 price_change / delay(close, 5)
        # 그렇지 않으면 inner_condition
        alpha = self.condition(
            self.lt(close, close_lag5),
            self.div(price_change, close_lag5),
            inner_condition
        )
        
        return alpha.fillna(0)
