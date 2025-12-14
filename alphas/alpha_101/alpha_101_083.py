import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_083(BaseAlpha):
    """
    alpha101_083: pow(ts_rank(sub({disk:vwap},ts_max({disk:vwap},15.3217)),20.7127),ts_delta({disk:close},4.96796))
    
    VWAP과 최대 VWAP 차이의 시계열 랭킹을 종가 변화로 거듭제곱
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_083"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_083 계산
        """
        # 밑: VWAP과 최대 VWAP 차이의 시계열 랭킹
        # 1. ts_max(vwap, 15.3217) -> 반올림하여 15
        max_window = int(round(15.3217))
        vwap_max = self.ts_max(data['vwap'], max_window)
        
        # 2. sub(vwap, vwap_max)
        vwap_diff = self.sub(data['vwap'], vwap_max)
        
        # 3. ts_rank(vwap_diff, 20.7127) -> 반올림하여 21
        rank_window = int(round(20.7127))
        base = self.ts_rank(vwap_diff, rank_window)
        
        # 지수: 종가 변화
        # 4. ts_delta(close, 4.96796) -> 반올림하여 5
        delta_window = int(round(4.96796))
        power = self.ts_delta(data['close'], delta_window)
        
        # 5. pow(base, power)
        alpha = self.pow(base, power)
        
        return alpha.fillna(0)
