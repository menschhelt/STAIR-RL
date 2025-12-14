import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_054(BaseAlpha):
    """
    alpha101_054: mul(-1,ts_corr(rank(div(sub({disk:close},ts_min({disk:low},12)),sub(ts_max({disk:high},12),ts_min({disk:low},12)))),rank({disk:volume}),6))
    
    12기간 스토캐스틱 %K 지표 랭킹과 거래량 랭킹의 6기간 상관관계의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_054"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_054 계산
        """
        # 스토캐스틱 %K 계산
        # 1. ts_min(low, 12)
        low_min_12 = self.ts_min(data['low'], 12)
        
        # 2. ts_max(high, 12)
        high_max_12 = self.ts_max(data['high'], 12)
        
        # 3. sub(close, low_min_12)
        numerator = self.sub(data['close'], low_min_12)
        
        # 4. sub(high_max_12, low_min_12)
        denominator = self.sub(high_max_12, low_min_12)
        
        # 5. div(numerator, denominator) - 스토캐스틱 %K
        stochastic_k = self.div(numerator, denominator)
        
        # 6. rank(stochastic_k)
        stoch_rank = self.rank(stochastic_k)
        
        # 7. rank(volume)
        volume_rank = self.rank(data['volume'])
        
        # 8. ts_corr(stoch_rank, volume_rank, 6)
        corr_6 = self.ts_corr(stoch_rank, volume_rank, 6)
        
        # 9. mul(-1, corr_6)
        alpha = self.mul(corr_6, -1)
        
        return alpha.fillna(0)
