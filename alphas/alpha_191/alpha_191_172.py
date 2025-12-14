import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_172(BaseAlpha):
    """
    alpha191_172: mul(3,sub(sma({disk:close},13,2),mul(2,add(sma(sma({disk:close},13,2),13,2),sma(sma(sma(log({disk:close}),13,2),13,2),13,2)))))
    
    다중 평활화된 종가와 로그 종가의 복합 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_172"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_172 계산
        """
        close = data['close']
        
        # 첫 번째 SMA
        sma1 = self.sma(close, 13, 2)
        
        # 두 번째 부분: sma(sma(close, 13, 2), 13, 2)
        sma2 = self.sma(sma1, 13, 2)
        
        # 세 번째 부분: log(close)의 다중 SMA
        log_close = self.log(close)
        log_sma1 = self.sma(log_close, 13, 2)
        log_sma2 = self.sma(log_sma1, 13, 2)
        log_sma3 = self.sma(log_sma2, 13, 2)
        
        # 복합 계산
        combined = self.add(sma2, log_sma3)
        second_part = self.mul(2, combined)
        
        # 차이 계산
        diff = self.sub(sma1, second_part)
        
        # 3 곱하기
        alpha = self.mul(3, diff)
        
        return alpha.fillna(0)
