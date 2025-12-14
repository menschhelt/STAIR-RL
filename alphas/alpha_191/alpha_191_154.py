import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_154(BaseAlpha):
    """
    alpha191_154: sub(sma({disk:volume},13,2),sub(sma({disk:volume},27,2),sma(sub(sma({disk:volume},13,2),sma({disk:volume},27,2)),10,2)))
    
    거래량의 다중 기간 SMA 차이 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_154"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_154 계산
        """
        volume = data['volume']
        
        # 기본 SMA들
        sma_13 = self.sma(volume, 13, 2)
        sma_27 = self.sma(volume, 27, 2)
        
        # SMA 차이
        sma_diff = self.sub(sma_13, sma_27)
        
        # 차이의 10일 SMA
        sma_diff_smoothed = self.sma(sma_diff, 10, 2)
        
        # 전체 계산
        second_part = self.sub(sma_27, sma_diff_smoothed)
        alpha = self.sub(sma_13, second_part)
        
        return alpha.fillna(0)
