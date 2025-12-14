import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_038(BaseAlpha):
    """
    alpha191_038: mul(sub(rank(ts_decayed_linear(ts_delta({disk:close},2),8)),rank(ts_decayed_linear(ts_corr(add(mul({disk:vwap},0.3),mul({disk:open},0.7)),ts_sum(ts_mean({disk:volume},180),37),14),12))),-1)
    
    -1 * (종가 변화 decayed linear 랭킹 - 복잡한 가격과 거래량 상관관계 decayed linear 랭킹)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_038"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_038 계산
        """
        close = data['close']
        vwap = data['vwap']
        open_price = data['open']
        volume = data['volume']
        
        # 첫 번째 부분: ts_delta(close, 2)의 8일 decayed linear
        close_delta = self.ts_delta(close, 2)
        decay1 = self.ts_decayed_linear(close_delta, 8)
        rank1 = self.rank(decay1)
        
        # 두 번째 부분: 복잡한 가격과 거래량 상관관계
        # 가중 가격: vwap * 0.3 + open * 0.7
        weighted_price = self.add(
            self.mul(vwap, 0.3),
            self.mul(open_price, 0.7)
        )
        
        # 거래량 지표: ts_sum(ts_mean(volume, 180), 37)
        volume_mean_180 = self.ts_mean(volume, 180)
        volume_indicator = self.ts_sum(volume_mean_180, 37)
        
        # 14일 상관관계
        corr = self.ts_corr(weighted_price, volume_indicator, 14)
        
        # 12일 decayed linear
        decay2 = self.ts_decayed_linear(corr, 12)
        rank2 = self.rank(decay2)
        
        # 차이
        diff = self.sub(rank1, rank2)
        
        # -1 곱하기
        alpha = self.mul(diff, -1)
        
        return alpha.fillna(0)
