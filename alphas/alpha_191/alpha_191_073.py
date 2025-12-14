import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_073(BaseAlpha):
    """
    alpha191_073: add(rank(ts_corr(ts_sum(add(mul({disk:low},0.35),mul({disk:vwap},0.65)),20),ts_sum(ts_mean({disk:volume},40),20),7)),rank(ts_corr(rank({disk:vwap}),rank({disk:volume}),6)))
    
    두 개의 상관관계 랭킹의 합계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_073"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_073 계산
        """
        low = data['low']
        vwap = data['vwap']
        volume = data['volume']
        
        # 첫 번째 부분: 가중 가격과 거래량의 상관관계
        weighted_price = self.add(
            self.mul(low, 0.35),
            self.mul(vwap, 0.65)
        )
        price_sum = self.ts_sum(weighted_price, 20)
        
        volume_mean_40 = self.ts_mean(volume, 40)
        volume_sum = self.ts_sum(volume_mean_40, 20)
        
        corr1 = self.ts_corr(price_sum, volume_sum, 7)
        rank1 = self.rank(corr1)
        
        # 두 번째 부분: VWAP와 거래량 랭킹의 상관관계
        vwap_rank = self.rank(vwap)
        volume_rank = self.rank(volume)
        corr2 = self.ts_corr(vwap_rank, volume_rank, 6)
        rank2 = self.rank(corr2)
        
        # 합계
        alpha = self.add(rank1, rank2)
        
        return alpha.fillna(0)
