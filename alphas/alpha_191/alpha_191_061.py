import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_061(BaseAlpha):
    """
    alpha191_061: mul(-1,ts_corr({disk:high},rank({disk:volume}),5))
    
    -1 * 고가와 거래량 랭킹의 5일 상관관계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_061"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_061 계산
        """
        high = data['high']
        volume = data['volume']
        
        # 거래량 랭킹
        volume_rank = self.rank(volume)
        
        # 5일 상관관계
        corr = self.ts_corr(high, volume_rank, 5)
        
        # -1 곱하기
        alpha = self.mul(-1, corr)
        
        return alpha.fillna(0)
