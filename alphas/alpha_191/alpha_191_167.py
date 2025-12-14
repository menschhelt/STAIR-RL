import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_167(BaseAlpha):
    """
    alpha191_167: mul(-1,div({disk:volume},ts_mean({disk:volume},20)))
    
    -1 * 상대 거래량 (현재 거래량 / 20일 평균 거래량)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_167"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_167 계산
        """
        volume = data['volume']
        
        # 20일 평균 거래량
        volume_mean = self.ts_mean(volume, 20)
        
        # 상대 거래량
        relative_volume = self.div(volume, volume_mean)
        
        # -1 곱하기
        alpha = self.mul(-1, relative_volume)
        
        return alpha.fillna(0)
