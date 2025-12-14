import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_125(BaseAlpha):
    """
    alpha191_125: div(add({disk:close},add({disk:high},{disk:low})),3)
    
    (종가 + 고가 + 저가) / 3 (전형적인 가격)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_125"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_125 계산
        """
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 전형적인 가격: (close + high + low) / 3
        typical_price = self.div(
            self.add(close, self.add(high, low)),
            3
        )
        
        return typical_price.fillna(0)
