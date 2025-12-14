import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_121(BaseAlpha):
    """
    alpha191_121: div(sub(sma(sma(sma(log({disk:close}),13,2),13,2),13,2),delay(sma(sma(sma(log({disk:close}),13,2),13,2),13,2),1)),delay(sma(sma(sma(log({disk:close}),13,2),13,2),13,2),1))
    
    로그 종가의 삼중 SMA 변화율
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_121"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_121 계산
        """
        close = data['close']
        
        # 로그 종가
        log_close = self.log(close)
        
        # 첫 번째 SMA
        sma1 = self.sma(log_close, 13, 2)
        
        # 두 번째 SMA
        sma2 = self.sma(sma1, 13, 2)
        
        # 세 번째 SMA (삼중 SMA)
        triple_sma = self.sma(sma2, 13, 2)
        
        # 1일 지연된 삼중 SMA
        triple_sma_lag1 = self.delay(triple_sma, 1)
        
        # 변화율
        change = self.sub(triple_sma, triple_sma_lag1)
        alpha = self.div(change, triple_sma_lag1)
        
        return alpha.fillna(0)
