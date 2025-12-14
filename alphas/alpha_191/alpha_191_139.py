import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_139(BaseAlpha):
    """
    alpha191_139: min(rank(ts_decayed_linear(sub(add(rank({disk:open}),rank({disk:low})),add(rank({disk:high}),rank({disk:close}))),8)),ts_rank(ts_decayed_linear(ts_corr(ts_rank({disk:close},8),ts_rank(ts_mean({disk:volume},60),20),8),7),3))
    
    가격 랭킹 차이와 종가-거래량 상관관계의 최소값
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_139"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_139 계산
        """
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # 첫 번째 부분: (rank(open) + rank(low)) - (rank(high) + rank(close))
        open_rank = self.rank(open_price)
        low_rank = self.rank(low)
        high_rank = self.rank(high)
        close_rank = self.rank(close)
        
        first_sum = self.add(open_rank, low_rank)
        second_sum = self.add(high_rank, close_rank)
        price_diff = self.sub(first_sum, second_sum)
        
        first_part = self.rank(self.ts_decayed_linear(price_diff, 8))
        
        # 두 번째 부분: 종가와 거래량 상관관계
        close_ts_rank = self.ts_rank(close, 8)
        volume_mean = self.ts_mean(volume, 60)
        volume_ts_rank = self.ts_rank(volume_mean, 20)
        
        corr = self.ts_corr(close_ts_rank, volume_ts_rank, 8)
        decayed = self.ts_decayed_linear(corr, 7)
        second_part = self.ts_rank(decayed, 3)
        
        # 두 부분의 최소값
        alpha = self.min(first_part, second_part)
        
        return alpha.fillna(0)
