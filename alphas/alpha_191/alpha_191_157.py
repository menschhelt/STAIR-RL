import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_157(BaseAlpha):
    """
    alpha191_157: div(sub(sub({disk:high},sma({disk:close},15,2)),sub({disk:low},sma({disk:close},15,2))),{disk:close})
    
    (고가-종가SMA) - (저가-종가SMA)를 종가로 정규화
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_157"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_157 계산
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 15일 종가 SMA
        close_sma = self.sma(close, 15, 2)
        
        # 고가와 저가에서 SMA 차이
        high_diff = self.sub(high, close_sma)
        low_diff = self.sub(low, close_sma)
        
        # 고가 차이에서 저가 차이를 빼기
        range_diff = self.sub(high_diff, low_diff)
        
        # 종가로 정규화
        alpha = self.div(range_diff, close)
        
        return alpha.fillna(0)
