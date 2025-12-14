import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_104(BaseAlpha):
    """
    alpha191_104: mul(-1,ts_corr(rank({disk:open}),rank({disk:volume}),10))
    
    -1 * 시가와 거래량 랭킹의 10일 상관관계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_104"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_104 계산
        """
        open_price = data['open']
        volume = data['volume']
        
        # 랭킹
        open_rank = self.rank(open_price)
        volume_rank = self.rank(volume)
        
        # 10일 상관관계
        corr = self.ts_corr(open_rank, volume_rank, 10)
        
        # -1 곱하기
        alpha = self.mul(corr, -1)
        
        return alpha.fillna(0)
