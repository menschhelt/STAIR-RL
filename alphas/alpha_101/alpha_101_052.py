import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_052(BaseAlpha):
    """
    alpha101_052: mul(-1,ts_delta(div(sub(sub({disk:close},{disk:low}),sub({disk:high},{disk:close})),sub({disk:close},{disk:low})),9))
    
    가격 위치 지표 (Williams %R 유사)의 9기간 변화의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_052"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_052 계산
        """
        # 1. sub(close, low)
        close_low_diff = self.sub(data['close'], data['low'])
        
        # 2. sub(high, close)
        high_close_diff = self.sub(data['high'], data['close'])
        
        # 3. sub(close_low_diff, high_close_diff)
        numerator = self.sub(close_low_diff, high_close_diff)
        
        # 4. div(numerator, close_low_diff) - Williams %R과 유사한 지표
        price_position = self.div(numerator, close_low_diff)
        
        # 5. ts_delta(price_position, 9)
        delta_9 = self.ts_delta(price_position, 9)
        
        # 6. mul(-1, delta_9)
        alpha = self.mul(delta_9, -1)
        
        return alpha.fillna(0)
