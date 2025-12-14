import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_044(BaseAlpha):
    """
    alpha191_044: mul(rank(ts_delta(add(mul({disk:close},0.6),mul({disk:open},0.4)),1)),rank(ts_corr({disk:vwap},ts_mean({disk:volume},150),15)))
    
    가중 가격 변화 랭킹 * VWAP-거래량 상관관계 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_044"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_044 계산
        """
        close = data['close']
        open_price = data['open']
        vwap = data['vwap']
        volume = data['volume']
        
        # 가중 가격: close * 0.6 + open * 0.4
        weighted_price = self.add(
            self.mul(close, 0.6),
            self.mul(open_price, 0.4)
        )
        
        # 1일 변화
        price_delta = self.ts_delta(weighted_price, 1)
        rank1 = self.rank(price_delta)
        
        # VWAP와 150일 평균 거래량의 15일 상관관계
        volume_mean_150 = self.ts_mean(volume, 150)
        corr = self.ts_corr(vwap, volume_mean_150, 15)
        rank2 = self.rank(corr)
        
        # 곱하기
        alpha = self.mul(rank1, rank2)
        
        return alpha.fillna(0)
