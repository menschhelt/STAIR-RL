import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

# 부모 디렉토리를 import 경로에 추가
from alphas.base.base import BaseAlpha

class alpha_101_005(BaseAlpha):
    """
    Alpha 101_005: mul(-1,ts_corr({disk:open},{disk:volume},10))
    
    시가와 거래량의 음의 상관관계 활용
    """
    neutralizer_type: str = "mean"  # 평균 중립화

    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_005"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_005 계산
        """
        # 1. open과 volume의 10기간 상관계수 (DSL: ts_corr(..., 10))
        corr = self.ts_corr(data['open'], data['volume'], 10)
        
        # 2. -1을 곱함
        alpha = self.mul(corr, -1)
        
        return alpha.fillna(0)
