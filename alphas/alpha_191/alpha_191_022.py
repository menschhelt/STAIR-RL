import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_022(BaseAlpha):
    """
    alpha191_022: div(sma(condition(gt({disk:close},delay({disk:close},1)),ts_std({disk:close},20),0),20,1),mul(add(sma(condition(gt({disk:close},delay({disk:close},1)),ts_std({disk:close},20),0),20,1),sma(condition(le({disk:close},delay({disk:close},1)),ts_std({disk:close},20),0),20,1)),100))
    
    조건부 변동성 지표의 비율
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_022"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_022 계산
        """
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # 20일 표준편차
        close_std_20 = self.ts_std(close, 20)
        
        # 상승일 조건
        up_condition = self.condition(
            self.gt(close, close_lag1),
            close_std_20,
            0
        )
        
        # 하락/횡보일 조건
        down_condition = self.condition(
            self.le(close, close_lag1),
            close_std_20,
            0
        )
        
        # SMA 계산
        up_sma = self.sma(up_condition, 20, 1)
        down_sma = self.sma(down_condition, 20, 1)
        
        # 분모
        denominator = self.mul(
            self.add(up_sma, down_sma),
            100
        )
        
        alpha = self.div(up_sma, denominator)
        
        return alpha.fillna(0)
