import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_057(BaseAlpha):
    """
    alpha191_057: div(countcond(gt({disk:close},delay({disk:close},1)),20),mul(20,100))
    
    20일간 상승일 비율 (%)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_057"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_057 계산
        """
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # 상승일 조건
        up_condition = self.gt(close, close_lag1)
        
        # 20일간 상승일 수
        up_count = self.ts_sum(up_condition.astype(float), 20)
        
        # 비율 (%)
        alpha = self.div(up_count, self.mul(20, 100))
        
        return alpha.fillna(0)
