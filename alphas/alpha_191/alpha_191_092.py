import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_092(BaseAlpha):
    """
    alpha191_092: ts_sum(condition(ge({disk:open},delay({disk:open},1)),0,max(sub({disk:open},{disk:low}),sub({disk:open},delay({disk:open},1)))),20)
    
    20일간 시가 하락 시 변동성 지표 합계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_092"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_092 계산
        """
        open_price = data['open']
        low = data['low']
        open_lag1 = self.delay(open_price, 1)
        
        # open >= delay(open, 1)이면 0, 그렇지 않으면 변동성 지표
        volatility_indicator = self.max(
            self.sub(open_price, low),
            self.sub(open_price, open_lag1)
        )
        
        condition_value = self.condition(
            self.ge(open_price, open_lag1),
            0,
            volatility_indicator
        )
        
        # 20일 합계
        alpha = self.ts_sum(condition_value, 20)
        
        return alpha.fillna(0)
