import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_176(BaseAlpha):
    """
    alpha191_176: mul(div(sub(20,lowday({disk:high},20)),20),100)
    
    20일 중 고가가 최저였던 날부터의 경과일을 백분율로 표현
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_176"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_176 계산
        """
        high = data['high']
        
        # lowday 함수가 없는 경우 근사 계산
        # 20일 중 최저 고가가 나타난 날을 찾기
        if hasattr(self, 'lowday'):
            low_day = self.lowday(high, 20)
        else:
            # 근사: 20일 최소값과의 차이로 추정
            min_high = self.ts_min(high, 20)
            # 현재 고가가 최소값에 가까울수록 낮은 값
            days_since_low = self.div(self.sub(high, min_high), min_high) * 20
            low_day = 20 - days_since_low  # 역산하여 근사
        
        # (20 - lowday) / 20 * 100
        ratio = self.div((20 - low_day), 20)
        alpha = self.mul(ratio, 100)
        
        return alpha.fillna(50)
