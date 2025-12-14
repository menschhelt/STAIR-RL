import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_045(BaseAlpha):
    """
    alpha191_045: div(add(ts_mean({disk:close},3),add(ts_mean({disk:close},6),add(ts_mean({disk:close},12),ts_mean({disk:close},24)))),mul(4,{disk:close}))
    
    (3일 + 6일 + 12일 + 24일 평균) / (4 * 현재 종가)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_045"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_045 계산
        """
        close = data['close']
        
        # 다양한 기간의 평균
        mean_3 = self.ts_mean(close, 3)
        mean_6 = self.ts_mean(close, 6)
        mean_12 = self.ts_mean(close, 12)
        mean_24 = self.ts_mean(close, 24)
        
        # 평균들의 합계
        sum_means = self.add(
            mean_3,
            self.add(
                mean_6,
                self.add(mean_12, mean_24)
            )
        )
        
        # 분모: 4 * close
        denominator = self.mul(4, close)
        
        alpha = self.div(sum_means, denominator)
        
        return alpha.fillna(0)
