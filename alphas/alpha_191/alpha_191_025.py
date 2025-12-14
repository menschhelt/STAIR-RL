import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_025(BaseAlpha):
    """
    alpha191_025: add(sub(div(ts_sum({disk:close},7),7),{disk:close}),ts_corr({disk:vwap},delay({disk:close},5),230))
    
    (7일 평균 - 현재 종가) + VWAP와 5일 지연 종가의 230일 상관관계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_025"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_025 계산
        """
        close = data['close']
        vwap = data['vwap']
        
        # 7일 평균 - 현재 종가
        close_mean_7 = self.div(self.ts_sum(close, 7), 7)
        price_diff = self.sub(close_mean_7, close)
        
        # VWAP와 5일 지연 종가의 230일 상관관계
        close_lag5 = self.delay(close, 5)
        corr = self.ts_corr(vwap, close_lag5, 230)
        
        alpha = self.add(price_diff, corr)
        
        return alpha.fillna(0)
