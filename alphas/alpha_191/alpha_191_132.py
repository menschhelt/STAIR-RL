import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_132(BaseAlpha):
    """
    alpha191_132: mul(div(sub(20,lowday({disk:high},20)),20),sub(100,mul(div(sub(20,highday({disk:low},20)),20),100)))
    
    고가의 최저점과 저가의 최고점을 이용한 복합 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_132"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_132 계산
        """
        high = data['high']
        low = data['low']
        
        # 20일 중 고가가 최저였던 날까지의 일수
        low_day_high = self.lowday(high, 20)
        term1 = self.mul(
            self.div((20 - low_day_high), 20),
            1
        )
        
        # 20일 중 저가가 최고였던 날까지의 일수
        high_day_low = self.highday(low, 20)
        term2_inner = self.mul(
            self.div((20 - high_day_low), 20),
            100
        )
        term2 = (100 - term2_inner)
        
        # 두 항의 곱
        alpha = self.mul(term1, term2)
        
        return alpha.fillna(0)
