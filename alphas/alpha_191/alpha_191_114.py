import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_114(BaseAlpha):
    """
    alpha191_114: pow(rank(ts_corr(add(mul({disk:high},0.9),mul({disk:close},0.1)),ts_mean({disk:volume},30),10)),rank(ts_corr(ts_rank(div(add({disk:high},{disk:low}),2),4),ts_rank({disk:volume},10),7)))
    
    두 개의 상관관계 랭킹의 거듭제곱
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_114"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_114 계산
        """
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # 첫 번째 부분: 가중가격과 30일 평균 거래량의 10일 상관관계
        weighted_price = self.add(
            self.mul(high, 0.9),
            self.mul(close, 0.1)
        )
        volume_mean_30 = self.ts_mean(volume, 30)
        corr1 = self.ts_corr(weighted_price, volume_mean_30, 10)
        base_rank = self.rank(corr1)
        
        # 두 번째 부분: HL 평균과 거래량의 ts_rank 상관관계
        hl_avg = self.div(self.add(high, low), 2)
        hl_ts_rank = self.ts_rank(hl_avg, 4)
        volume_ts_rank = self.ts_rank(volume, 10)
        corr2 = self.ts_corr(hl_ts_rank, volume_ts_rank, 7)
        exp_rank = self.rank(corr2)
        
        # 거듭제곱
        alpha = self.pow(base_rank, exp_rank)
        
        return alpha.fillna(0)
