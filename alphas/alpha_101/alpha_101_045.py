import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_045(BaseAlpha):
    """
    alpha101_045: condition(lt(0.25,sub(div(sub(delay({disk:close},20),delay({disk:close},10)),10),div(sub(delay({disk:close},10),{disk:close}),10))),mul(-1,1),condition(lt(sub(div(sub(delay({disk:close},20),delay({disk:close},10)),10),div(sub(delay({disk:close},10),{disk:close}),10)),0),1,mul(mul(-1,1),sub({disk:close},delay({disk:close},1)))))
    
    다기간 가격 변화 기울기 비교에 따른 조건부 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_045"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_045 계산
        """
        # 1. delay(close, 20), delay(close, 10)
        close_lag20 = self.delay(data['close'], 20)
        close_lag10 = self.delay(data['close'], 10)
        close_lag1 = self.delay(data['close'], 1)
        
        # 2. 첫 번째 기울기: (close_lag20 - close_lag10) / 10
        slope1_num = self.sub(close_lag20, close_lag10)
        slope1 = self.div(slope1_num, 10)
        
        # 3. 두 번째 기울기: (close_lag10 - close) / 10
        slope2_num = self.sub(close_lag10, data['close'])
        slope2 = self.div(slope2_num, 10)
        
        # 4. 기울기 차이
        slope_diff = self.sub(slope1, slope2)
        
        # 5. 첫 번째 조건: slope_diff > 0.25
        condition1 = self.lt(0.25, slope_diff)
        
        # 6. 두 번째 조건: slope_diff < 0
        condition2 = self.lt(slope_diff, 0)
        
        # 7. close - delay(close, 1)
        daily_change = self.sub(data['close'], close_lag1)
        neg_daily_change = self.mul(daily_change, -1)
        
        # 8. 중첩된 조건문
        inner_condition = self.condition(
            condition2,
            1,
            neg_daily_change
        )
        
        alpha = self.condition(
            condition1,
            -1,
            inner_condition
        )
        
        return alpha.fillna(0)
