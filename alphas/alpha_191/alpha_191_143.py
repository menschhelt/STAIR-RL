import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_143(BaseAlpha):
    """
    alpha191_143: div(sumif(div(abs(div({disk:close},sub(delay({disk:close},1),1))),{disk:amount}),20,lt({disk:close},delay({disk:close},1))),countcond(lt({disk:close},delay({disk:close},1)),20))
    
    하락일의 평균 거래액 대비 변동성
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_143"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_143 계산
        """
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # amount가 없는 경우 volume으로 대체
        if 'amount' in data:
            amount = data['amount']
        else:
            amount = data['volume'] * data['close']
        
        # 하락 조건
        is_down = self.lt(close, close_lag1)
        
        # 절대 수익률
        abs_returns = self.abs(self.div(close, self.sub(close_lag1, 1)))
        
        # 거래액 대비 변동성
        volatility_ratio = self.div(abs_returns, amount)
        
        # 하락일의 volatility_ratio 합계 (sumif 근사)
        down_volatility = self.condition(is_down, volatility_ratio, 0)
        sum_down_volatility = self.ts_sum(down_volatility, 20)
        
        # 하락일 수 (countcond 근사)
        down_count = self.ts_sum(self.condition(is_down, 1, 0), 20)
        
        # 평균 계산
        alpha = self.div(sum_down_volatility, down_count)
        
        return alpha.fillna(0)
