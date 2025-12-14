import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_022(BaseAlpha):
    """
    alpha101_022: condition(lt(div(ts_sum({disk:high},20),20),{disk:high}),mul(-1,ts_delta({disk:high},2)),0)
    
    20기간 고가 평균이 현재 고가보다 낮으면 고가 2기간 변화의 음수, 아니면 0
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_022"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_022 계산
        """
        # 1. div(ts_sum(high, 20), 20) - 20기간 고가 평균
        high_mean_20 = self.div(self.ts_sum(data['high'], 20), 20)
        
        # 2. lt(high_mean_20, high) - 조건
        condition = self.lt(high_mean_20, data['high'])
        
        # 3. ts_delta(high, 2)
        high_delta = self.ts_delta(data['high'], 2)
        
        # 4. mul(-1, high_delta)
        neg_delta = self.mul(high_delta, -1)
        
        # 5. condition(..., neg_delta, 0)
        alpha = self.condition(condition, neg_delta, 0)
        
        return alpha.fillna(0)
