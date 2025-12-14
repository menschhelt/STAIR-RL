import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_145(BaseAlpha):
    """
    alpha191_145: mul(ts_mean(div(sub({disk:close},delay({disk:close},1)),sub(delay({disk:close},1),sma(div(sub({disk:close},delay({disk:close},1)),delay({disk:close},1)),61,2))),20),div(...))
    
    정규화된 수익률의 복잡한 변동성 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_145"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_145 계산 (복잡한 식을 단순화)
        """
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # 기본 수익률
        returns = self.div(self.sub(close, close_lag1), close_lag1)
        
        # 61일 SMA로 정규화된 수익률
        returns_sma = self.sma(returns, 61, 2)
        normalized_returns = self.div(self.sub(close, close_lag1), self.sub(close_lag1, returns_sma))
        
        # 20일 평균
        mean_normalized = self.ts_mean(normalized_returns, 20)
        
        # 변동성 계산 (단순화)
        volatility = self.ts_std(normalized_returns, 60)
        
        # 비율 계산
        alpha = self.div(mean_normalized, volatility)
        
        return alpha.fillna(0)
