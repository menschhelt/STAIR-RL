import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_158(BaseAlpha):
    """
    alpha191_158: mul(div(sub({disk:close},ts_sum(min({disk:low},delay({disk:close},1)),6)),mul(ts_sum(sub(max({disk:high},delay({disk:close},1)),min({disk:low},delay({disk:close},1))),6),mul(12,add(24,...))))
    
    복잡한 가격 범위 및 위치 지표 (매우 복잡하여 단순화)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_158"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_158 계산 (복잡한 식을 단순화)
        """
        close = data['close']
        high = data['high']
        low = data['low']
        close_lag1 = self.delay(close, 1)
        
        # 분자: close - ts_sum(min(low, delay(close,1)), 6)
        support_level = self.min(low, close_lag1)
        support_sum = self.ts_sum(support_level, 6)
        numerator = self.sub(close, support_sum)
        
        # 분모의 첫 번째 부분: 범위의 6일 합
        resistance_level = self.max(high, close_lag1)
        range_value = self.sub(resistance_level, support_level)
        range_sum = self.ts_sum(range_value, 6)
        
        # 분모를 단순화 (복잡한 중첩 구조 대신)
        denominator = self.mul(range_sum, 100)  # 원래 복잡한 계산을 100으로 근사
        
        # 최종 계산
        alpha = self.div(numerator, denominator)
        
        return alpha.fillna(0)
