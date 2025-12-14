import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_175(BaseAlpha):
    """
    alpha191_175: ts_corr(rank(div(sub({disk:close},ts_min({disk:low},12)),sub(ts_max({disk:high},12),ts_min({disk:low},12)))),rank({disk:volume}),6)
    
    12일 스토캐스틱 %K와 거래량의 6일 상관관계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_175"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_175 계산
        """
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # 12일 최고가와 최저가
        high_max = self.ts_max(high, 12)
        low_min = self.ts_min(low, 12)
        
        # 스토캐스틱 %K 계산
        numerator = self.sub(close, low_min)
        denominator = self.sub(high_max, low_min)
        stochastic_k = self.div(numerator, denominator)
        
        # 랭킹
        stoch_rank = self.rank(stochastic_k)
        volume_rank = self.rank(volume)
        
        # 6일 상관관계
        alpha = self.ts_corr(stoch_rank, volume_rank, 6)
        
        return alpha.fillna(0)
