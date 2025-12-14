import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_023(BaseAlpha):
    """
    alpha101_023: condition(or_(lt(div(ts_delta(div(ts_sum({disk:close},100),100),100),delay({disk:close},100)),0.05),eq(div(ts_delta(div(ts_sum({disk:close},100),100),100),delay({disk:close},100)),0.05)),mul(-1,sub({disk:close},ts_min({disk:close},100))),mul(-1,ts_delta({disk:close},3)))
    
    100기간 평균의 장기 변화율에 따른 조건부 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_023"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_023 계산
        """
        # 1. div(ts_sum(close, 100), 100) - 100기간 평균
        close_mean_100 = self.div(self.ts_sum(data['close'], 100), 100)
        
        # 2. ts_delta(..., 100)
        mean_delta = self.ts_delta(close_mean_100, 100)
        
        # 3. delay(close, 100)
        close_lag100 = self.delay(data['close'], 100)
        
        # 4. div(mean_delta, close_lag100) - 변화율
        rate = self.div(mean_delta, close_lag100)
        
        # 5. 조건들
        condition1 = self.lt(rate, 0.05)
        condition2 = self.eq(rate, 0.05)
        main_condition = self.or_(condition1, condition2)
        
        # 6. ts_min(close, 100)
        close_min_100 = self.ts_min(data['close'], 100)
        
        # 7. sub(close, close_min_100)
        close_min_diff = self.sub(data['close'], close_min_100)
        
        # 8. mul(-1, close_min_diff)
        neg_close_min = self.mul(close_min_diff, -1)
        
        # 9. ts_delta(close, 3)
        close_delta3 = self.ts_delta(data['close'], 3)
        
        # 10. mul(-1, close_delta3)
        neg_delta3 = self.mul(close_delta3, -1)
        
        # 11. 최종 조건문
        alpha = self.condition(main_condition, neg_close_min, neg_delta3)
        
        return alpha.fillna(0)
