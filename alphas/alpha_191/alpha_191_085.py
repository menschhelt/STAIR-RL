import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_085(BaseAlpha):
    """
    alpha191_085: condition(lt(0.25,sub(div(sub(delay({disk:close},20),delay({disk:close},10)),10),div(sub(delay({disk:close},10),{disk:close}),10))),mul(-1,1),condition(lt(sub(div(sub(delay({disk:close},20),delay({disk:close},10)),10),div(sub(delay({disk:close},10),{disk:close}),10)),0),1,mul(mul(-1,1),sub({disk:close},delay({disk:close},1)))))
    
    복잡한 가격 변화율 기반 조건부 신호
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_085"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_085 계산
        """
        close = data['close']
        
        # 지연된 종가들
        close_lag10 = self.delay(close, 10)
        close_lag20 = self.delay(close, 20)
        close_lag1 = self.delay(close, 1)
        
        # 첫 번째 기울기: (delay(close, 20) - delay(close, 10)) / 10
        slope1 = self.div(self.sub(close_lag20, close_lag10), 10)
        
        # 두 번째 기울기: (delay(close, 10) - close) / 10
        slope2 = self.div(self.sub(close_lag10, close), 10)
        
        # 기울기 차이
        slope_diff = self.sub(slope1, slope2)
        
        # 내부 조건: slope_diff < 0이면 1, 그렇지 않으면 -1 * (close - delay(close, 1))
        inner_condition = self.condition(
            self.lt(slope_diff, 0),
            1,
            self.mul(-1, self.sub(close, close_lag1))
        )
        
        # 외부 조건: 0.25 < slope_diff이면 -1, 그렇지 않으면 inner_condition
        alpha = self.condition(
            self.lt(0.25, slope_diff),
            -1,
            inner_condition
        )
        
        return alpha.fillna(0)
