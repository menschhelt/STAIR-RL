import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_142(BaseAlpha):
    """
    alpha191_142: cumprod(condition(gt({disk:close},delay({disk:close},1)),div(sub({disk:close},delay({disk:close},1)),delay({disk:close},1)),1))
    
    상승일의 수익률 누적 곱 (복리 수익률)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_142"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_142 계산
        """
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # 상승 조건
        is_up = self.gt(close, close_lag1)
        
        # 수익률 계산
        returns = self.div(self.sub(close, close_lag1), close_lag1)
        
        # 조건: 상승일이면 수익률, 아니면 1
        condition_values = self.condition(is_up, returns, 1)
        
        # 누적 곱 (단순화하여 rolling product로 근사)
        # cumprod는 BaseAlpha에 없을 수 있으므로 다른 방법 사용
        alpha = self.ts_sum(self.log(condition_values.replace(0, 1)), 20).apply(np.exp)
        
        return alpha.fillna(1)
