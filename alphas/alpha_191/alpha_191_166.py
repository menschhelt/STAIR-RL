import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_166(BaseAlpha):
    """
    alpha191_166: ts_sum(condition(gt(sub({disk:close},delay({disk:close},1)),0),sub({disk:close},delay({disk:close},1)),0),12)
    
    12일간 상승일의 가격 상승분 합계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_166"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_166 계산
        """
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # 가격 변화
        price_change = self.sub(close, close_lag1)
        
        # 상승 조건
        is_positive = self.gt(price_change, 0)
        
        # 조건: 상승일이면 가격 변화, 아니면 0
        positive_changes = self.condition(is_positive, price_change, 0)
        
        # 12일 합계
        alpha = self.ts_sum(positive_changes, 12)
        
        return alpha.fillna(0)
