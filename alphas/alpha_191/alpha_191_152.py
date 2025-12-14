import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_152(BaseAlpha):
    """
    alpha191_152: div(add(ts_mean({disk:close},3),add(ts_mean({disk:close},6),add(ts_mean({disk:close},12),ts_mean({disk:close},24)))),4)
    
    다중 기간 종가 평균의 평균 (3, 6, 12, 24일)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_152"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_152 계산
        """
        close = data['close']
        
        # 각기 다른 기간의 이동평균
        mean_3 = self.ts_mean(close, 3)
        mean_6 = self.ts_mean(close, 6)
        mean_12 = self.ts_mean(close, 12)
        mean_24 = self.ts_mean(close, 24)
        
        # 모든 평균의 합
        total_sum = self.add(mean_3, self.add(mean_6, self.add(mean_12, mean_24)))
        
        # 4로 나누어 평균 계산
        alpha = self.div(total_sum, 4)
        
        return alpha.fillna(0)
