import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_126(BaseAlpha):
    """
    alpha191_126: pow(ma(pow(div(sub({disk:close},max({disk:close},12)),max({disk:close},12)),2),12),div(1,2))
    
    최대값 대비 가격 변화율의 변동성 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_126"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_126 계산
        """
        close = data['close']
        
        # 12일 최대값
        max_close_12 = self.max(close, 12)
        
        # (close - max_close_12) / max_close_12
        relative_change = self.div(
            self.sub(close, max_close_12),
            max_close_12
        )
        
        # 제곱
        squared = self.pow(relative_change, 2)
        
        # 12일 이동평균
        ma_squared = self.ma(squared, 12)
        
        # 제곱근
        alpha = self.pow(ma_squared, 0.5)
        
        return alpha.fillna(0)
