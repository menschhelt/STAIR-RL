import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_140(BaseAlpha):
    """
    alpha191_140: mul(rank(ts_corr(rank({disk:high}),rank(ts_mean({disk:volume},15)),9)),-1)
    
    -1 * 고가 랭킹과 15일 평균 거래량 랭킹의 9일 상관관계 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_140"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_140 계산
        """
        high = data['high']
        volume = data['volume']
        
        # 고가 랭킹
        high_rank = self.rank(high)
        
        # 15일 평균 거래량 랭킹
        volume_mean = self.ts_mean(volume, 15)
        volume_rank = self.rank(volume_mean)
        
        # 9일 상관관계
        corr = self.ts_corr(high_rank, volume_rank, 9)
        
        # 상관관계 랭킹에 -1 곱하기
        alpha = self.mul(self.rank(corr), -1)
        
        return alpha.fillna(0)
