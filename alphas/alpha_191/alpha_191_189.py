import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_189(BaseAlpha):
    """
    alpha191_189: log(mul(sub(countcond(...),1),div(sumif(...),mul(countcond(...),sumif(...)))))
    
    매우 복잡한 로그 및 거듭제곱 계산 (단순화)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_189"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_189 계산 (매우 복잡한 식을 단순화)
        """
        close = data['close']
        close_lag1 = self.delay(close, 1)
        close_lag19 = self.delay(close, 19)
        
        # 수익률 계산
        returns = self.div(close, close_lag1)
        long_returns = self.div(close, close_lag19)
        
        # 특정 조건 (원래 복잡한 조건을 단순화)
        condition1 = self.gt(returns, 1)  # 상승일
        condition2 = self.lt(long_returns, 1)  # 장기적으로는 하락
        
        # 조건 카운트 (countcond 근사)
        count1 = self.ts_sum(self.condition(condition1, 1, 0), 20)
        count2 = self.ts_sum(self.condition(condition2, 1, 0), 20)
        
        # 조건부 합계 (sumif 근사)
        sum1 = self.ts_sum(self.condition(condition1, self.pow(returns, 2), 0), 20)
        sum2 = self.ts_sum(self.condition(condition2, self.pow(long_returns, 2), 0), 20)
        
        # 복잡한 비율 계산을 단순화
        ratio1 = self.div(sum1, count1.replace(0, 1))
        ratio2 = self.div(sum2, count2.replace(0, 1))
        
        # 로그 계산
        log_input = self.mul(self.sub(count1, 1), self.div(ratio1, ratio2))
        alpha = self.log(log_input.replace(0, 1).clip(lower=0.001))  # 음수/0 방지
        
        return alpha.fillna(0)
