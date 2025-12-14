import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_100(BaseAlpha):
    """
    alpha101_100: div(sub({disk:close},{disk:open}),add(sub({disk:high},{disk:low}),.001))
    
    간단한 가격 변화를 가격 범위로 나눈 비율 (정규화된 수익률)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_100"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_100 계산
        """
        # 1. sub(close, open) - 가격 변화
        price_change = self.sub(data['close'], data['open'])
        
        # 2. sub(high, low) - 가격 범위
        price_range = self.sub(data['high'], data['low'])
        
        # 3. add(price_range, 0.001) - 0으로 나누는 것을 방지
        adjusted_range = self.add(price_range, 0.001)
        
        # 4. div(price_change, adjusted_range) - 정규화된 수익률
        alpha = self.div(price_change, adjusted_range)
        
        return alpha.fillna(0)