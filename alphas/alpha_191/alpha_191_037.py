import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_037(BaseAlpha):
    """
    alpha191_037: condition(lt(div(ts_sum({disk:high},20),20),{disk:high}),mul(-1,ts_delta({disk:high},2)),0)
    
    20일 평균 고가 < 현재 고가이면 -1 * 2일 고가 변화, 그렇지 않으면 0
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_037"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_037 계산
        """
        high = data['high']
        
        # 20일 평균 고가
        high_mean_20 = self.div(self.ts_sum(high, 20), 20)
        
        # 2일 고가 변화
        high_delta_2 = self.ts_delta(high, 2)
        
        # 조건부 계산
        alpha = self.condition(
            self.lt(high_mean_20, high),
            self.mul(-1, high_delta_2),
            0
        )
        
        return alpha.fillna(0)
