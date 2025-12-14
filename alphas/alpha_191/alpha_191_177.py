import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_177(BaseAlpha):
    """
    alpha191_177: div(sub({disk:close},delay({disk:close},1)),mul(delay({disk:close},1),{disk:volume}))
    
    가격 변화를 전일 가격과 거래량의 곱으로 나눈 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_177"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_177 계산
        """
        close = data['close']
        volume = data['volume']
        close_lag1 = self.delay(close, 1)
        
        # 분자: 가격 변화
        price_change = self.sub(close, close_lag1)
        
        # 분모: 전일 가격 * 거래량
        denominator = self.mul(close_lag1, volume)
        
        # 비율 계산
        alpha = self.div(price_change, denominator)
        
        return alpha.fillna(0)
