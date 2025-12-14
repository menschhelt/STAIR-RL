import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_165(BaseAlpha):
    """
    alpha191_165: mul(-20,mul(82.819,div(ts_sum(div({disk:close},sub(delay({disk:close},1),sub(1,ts_mean(div({disk:close},sub(delay({disk:close},1),1)),20)))),20),mul(sub(20,1),mul(sub(20,2),pow(ts_sum(div({disk:close},delay({disk:close},1)),20),1.5))))))
    
    매우 복잡한 수익률 정규화 및 변동성 조정 지표 (단순화)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_165"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_165 계산 (매우 복잡한 식을 단순화)
        """
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # 기본 수익률
        returns = self.div(close, close_lag1)
        
        # 20일 평균 수익률
        avg_returns = self.ts_mean(returns, 20)
        
        # 정규화된 수익률 (단순화)
        normalized_returns = self.div(returns, avg_returns)
        
        # 20일 합
        sum_normalized = self.ts_sum(normalized_returns, 20)
        
        # 수익률 변동성 근사
        returns_sum = self.ts_sum(returns, 20)
        volatility_factor = self.pow(returns_sum, 1.5)
        
        # 복잡한 분모를 단순화
        denominator = self.mul(19, self.mul(18, volatility_factor))
        
        # 비율 계산
        ratio = self.div(sum_normalized, denominator)
        
        # 상수 곱하기
        alpha = self.mul(-20, self.mul(82.819, ratio))
        
        return alpha.fillna(0)
