import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

# 부모 디렉토리를 import 경로에 추가
from alphas.base.base import BaseAlpha

class alpha_101_011(BaseAlpha):
    """
    Alpha 101_011: mul(sign(ts_delta({disk:volume},1)),mul(-1,ts_delta({disk:close},1)))
    
    거래량 변화 방향과 가격 변화의 반대 관계 활용
    """
    neutralizer_type: str = "mean"  # 평균 중립화

    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_011"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_011 계산
        """
        # 1. volume의 1기간 차분의 부호
        volume_delta = self.ts_delta(data['volume'], 1)
        volume_sign = self.sign(volume_delta)

        # 2. close의 1기간 차분 * -1
        close_delta = self.ts_delta(data['close'], 1)
        close_neg = self.mul(close_delta, -1)
        
        # 3. 두 값을 곱함
        alpha = self.mul(volume_sign, close_neg)
        
        return alpha.fillna(0)
