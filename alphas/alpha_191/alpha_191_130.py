import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_130(BaseAlpha):
    """
    alpha191_130: pow(rank(delay({disk:vwap},1)),ts_rank(ts_corr({disk:close},ts_mean({disk:volume},50),18),18))
    
    지연된 VWAP 랭킹의 거래량-가격 상관관계 제곱
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_130"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_130 계산
        """
        close = data['close']
        vwap = data['vwap']
        volume = data['volume']
        
        # 지연된 VWAP의 랭킹
        vwap_lag1 = self.delay(vwap, 1)
        vwap_rank = self.rank(vwap_lag1)
        
        # 종가와 50일 거래량 평균의 18일 상관관계
        volume_mean_50 = self.ts_mean(volume, 50)
        corr = self.ts_corr(close, volume_mean_50, 18)
        
        # 상관관계의 18일 시계열 랭킹
        ts_rank_corr = self.ts_rank(corr, 18)
        
        # 제곱
        alpha = self.pow(vwap_rank, ts_rank_corr)
        
        return alpha.fillna(0)
