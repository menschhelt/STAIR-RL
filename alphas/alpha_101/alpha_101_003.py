import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

# 부모 디렉토리를 import 경로에 추가
from alphas.base.base import BaseAlpha

class alpha_101_003(BaseAlpha):
    """
    Alpha 101_003: mul(-1,ts_rank(rank({disk:low}),9))
    
    저가의 시계열 랭킹 반전 활용
    """
    neutralizer_type: str = "mean"  # 평균 중립화
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_003"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_003 계산
        """
        # 1. low 가격의 랭킹
        low_rank = self.rank(data['low'])
        
        # 2. 9기간 시계열 랭킹 (DSL: ts_rank(..., 9))
        ts_rank = self.ts_rank(low_rank, 9)
        
        # 3. -1을 곱함
        alpha = self.mul(ts_rank, -1)
        
        return alpha.fillna(0)
