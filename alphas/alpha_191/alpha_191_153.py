import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_153(BaseAlpha):
    """
    alpha191_153: lt(sub({disk:vwap},min({disk:vwap},16)),ts_corr({disk:vwap},ts_mean({disk:volume},180),18))
    
    VWAP 위치와 VWAP-거래량 상관관계의 비교
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_153"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_153 계산
        """
        vwap = data['vwap']
        volume = data['volume']
        
        # 첫 번째 부분: vwap - min(vwap, 16)
        vwap_min = self.ts_min(vwap, 16)
        vwap_position = self.sub(vwap, vwap_min)
        
        # 두 번째 부분: ts_corr(vwap, ts_mean(volume, 180), 18)
        volume_mean = self.ts_mean(volume, 180)
        vwap_volume_corr = self.ts_corr(vwap, volume_mean, 18)
        
        # 비교 (less than)
        alpha = self.lt(vwap_position, vwap_volume_corr)
        
        return alpha.fillna(0)
