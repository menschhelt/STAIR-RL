import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_108(BaseAlpha):
    """
    alpha191_108: div(sma(sub({disk:high},{disk:low}),10,2),sma(sma(sub({disk:high},{disk:low}),10,2),10,2))
    
    True Range SMA / True Range 이중 SMA = 변동성 정규화 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_108"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_108 계산
        """
        high = data['high']
        low = data['low']
        
        # High - Low (True Range의 일부)
        hl_range = self.sub(high, low)
        
        # 첫 번째 SMA
        sma1 = self.sma(hl_range, 10, 2)
        
        # 이중 SMA
        sma2 = self.sma(sma1, 10, 2)
        
        # 비율
        alpha = self.div(sma1, sma2)
        
        return alpha.fillna(0)
