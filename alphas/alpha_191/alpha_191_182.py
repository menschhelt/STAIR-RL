import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_182(BaseAlpha):
    """
    alpha191_182: sub(max(sub({disk:close},ts_mean({disk:close},24)),24),div(min(sub({disk:close},ts_mean({disk:close},24)),24),ts_std({disk:close},24)))
    
    24일 평균 대비 가격 위치의 최대값에서 정규화된 최소값을 뺀 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_182"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_182 계산
        """
        close = data['close']
        
        # 24일 평균과 표준편차
        close_mean = self.ts_mean(close, 24)
        close_std = self.ts_std(close, 24)
        
        # 평균에서 현재가 차이
        deviation = self.sub(close, close_mean)
        
        # 최대값 (24와 비교하는 것은 의미가 없으므로 deviation의 최대값으로 근사)
        max_deviation = self.max(deviation, 0)  # 양수 편차의 최대값
        
        # 최소값을 표준편차로 정규화
        min_deviation = self.min(deviation, 0)  # 음수 편차의 최소값
        normalized_min = self.div(min_deviation, close_std)
        
        # 차이 계산
        alpha = self.sub(max_deviation, normalized_min)
        
        return alpha.fillna(0)
