import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_151(BaseAlpha):
    """
    alpha191_151: sma(sub(ts_mean(delay(sma(delay(div({disk:close},delay({disk:close},9)),1),9,1),1),12),ts_mean(delay(sma(delay(div({disk:close},delay({disk:close},9)),1),9,1),1),26)),9,1)
    
    복잡한 다중 지연 및 평활화 수익률 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_151"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_151 계산 (복잡한 식을 단순화)
        """
        close = data['close']
        
        # 9일 수익률
        close_lag9 = self.delay(close, 9)
        returns_9d = self.div(close, close_lag9)
        
        # 지연 및 평활화 (단순화)
        returns_lag1 = self.delay(returns_9d, 1)
        sma_9 = self.sma(returns_lag1, 9, 1)
        sma_lag1 = self.delay(sma_9, 1)
        
        # 12일과 26일 평균
        mean_12 = self.ts_mean(sma_lag1, 12)
        mean_26 = self.ts_mean(sma_lag1, 26)
        
        # 차이를 구하고 9일 평활화
        diff = self.sub(mean_12, mean_26)
        alpha = self.sma(diff, 9, 1)
        
        return alpha.fillna(0)
