import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_124(BaseAlpha):
    """
    alpha191_124: div(rank(ts_decayed_linear(ts_corr({disk:vwap},ts_mean({disk:volume},80),17),20)),rank(ts_decayed_linear(ts_delta(add(mul({disk:close},0.5),mul({disk:vwap},0.5)),3),16)))
    
    VWAP-거래량 상관관계 decayed linear 랭킹 / 가중가격 변화 decayed linear 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_124"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_124 계산
        """
        vwap = data['vwap']
        volume = data['volume']
        close = data['close']
        
        # 분자: VWAP와 80일 평균 거래량의 17일 상관관계
        volume_mean_80 = self.ts_mean(volume, 80)
        corr = self.ts_corr(vwap, volume_mean_80, 17)
        decay1 = self.ts_decayed_linear(corr, 20)
        rank1 = self.rank(decay1)
        
        # 분모: 가중가격의 3일 변화
        weighted_price = self.add(
            self.mul(close, 0.5),
            self.mul(vwap, 0.5)
        )
        price_delta = self.ts_delta(weighted_price, 3)
        decay2 = self.ts_decayed_linear(price_delta, 16)
        rank2 = self.rank(decay2)
        
        alpha = self.div(rank1, rank2)
        
        return alpha.fillna(0)
