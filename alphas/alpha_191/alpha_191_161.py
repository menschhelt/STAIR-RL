import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_161(BaseAlpha):
    """
    alpha191_161: div(div(sma(max(sub({disk:close},delay({disk:close},1)),0),12,1),mul(sma(abs(sub({disk:close},delay({disk:close},1))),12,1),sub(100,min(...)))),sub(max(...),min(...)))
    
    RSI와 유사한 복잡한 상대강도 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_161"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_161 계산 (복잡한 RSI 변형을 단순화)
        """
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # 가격 변화
        price_change = self.sub(close, close_lag1)
        
        # 상승분과 절대값
        gains = self.max(price_change, 0)
        abs_change = self.abs(price_change)
        
        # 12일 평균
        avg_gains = self.sma(gains, 12, 1)
        avg_abs_change = self.sma(abs_change, 12, 1)
        
        # RSI와 유사한 계산 (단순화)
        rsi_like = self.div(avg_gains, avg_abs_change)
        
        # 정규화 (0-1 범위)
        alpha = self.mul(rsi_like, 100)
        
        return alpha.fillna(50)
