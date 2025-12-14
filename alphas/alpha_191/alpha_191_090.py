import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_090(BaseAlpha):
    """
    alpha191_090: mul(mul(rank(sub({disk:close},max({disk:close},5))),rank(ts_corr(ts_mean({disk:volume},40),{disk:low},5))),-1)
    
    -1 * (종가 상대적 위치 랭킹 * 거래량-저가 상관관계 랭킹)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_090"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_090 계산
        """
        close = data['close']
        volume = data['volume']
        low = data['low']
        
        # close - max(close, 5) (항상 음수 또는 0)
        close_diff = self.sub(close, self.ts_max(close, 5))
        rank1 = self.rank(close_diff)
        
        # 40일 평균 거래량과 저가의 5일 상관관계
        volume_mean_40 = self.ts_mean(volume, 40)
        corr = self.ts_corr(volume_mean_40, low, 5)
        rank2 = self.rank(corr)
        
        # 곱하기 후 -1 곱하기
        alpha = self.mul(self.mul(rank1, rank2), -1)
        
        return alpha.fillna(0)
