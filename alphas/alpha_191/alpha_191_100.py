import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_100(BaseAlpha):
    """
    alpha191_100: mul(lt(rank(ts_corr({disk:close},ts_sum(ts_mean({disk:volume},30),37),15)),rank(ts_corr(rank(add(mul({disk:high},0.1),mul({disk:vwap},0.9))),rank({disk:volume}),11))),-1)
    
    -1 * (종가-거래량 상관관계 랭킹 < 가중가격-거래량 상관관계 랭킹)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_100"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_100 계산
        """
        close = data['close']
        high = data['high']
        vwap = data['vwap']
        volume = data['volume']
        
        # 첫 번째 부분: 종가와 복잡한 거래량 지표의 상관관계
        volume_mean_30 = self.ts_mean(volume, 30)
        volume_sum = self.ts_sum(volume_mean_30, 37)
        corr1 = self.ts_corr(close, volume_sum, 15)
        rank1 = self.rank(corr1)
        
        # 두 번째 부분: 가중가격과 거래량의 상관관계
        weighted_price = self.add(
            self.mul(high, 0.1),
            self.mul(vwap, 0.9)
        )
        price_rank = self.rank(weighted_price)
        volume_rank = self.rank(volume)
        corr2 = self.ts_corr(price_rank, volume_rank, 11)
        rank2 = self.rank(corr2)
        
        # 비교 후 -1 곱하기
        comparison = self.lt(rank1, rank2)
        alpha = self.mul(comparison, -1)
        
        return alpha.fillna(0).astype(float)
