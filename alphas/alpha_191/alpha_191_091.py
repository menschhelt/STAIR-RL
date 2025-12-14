import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_091(BaseAlpha):
    """
    alpha191_091: mul(max(rank(ts_decayed_linear(ts_delta(add(mul({disk:close},0.35),mul({disk:vwap},0.65)),2),3)),ts_rank(ts_decayed_linear(abs(ts_corr(ts_mean({disk:volume},180),{disk:close},13)),5),15)),-1)
    
    -1 * max(가중 가격 변화 decayed linear 랭킹, 거래량-종가 상관관계 절대값 decayed linear ts_rank)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_091"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_091 계산
        """
        close = data['close']
        vwap = data['vwap']
        volume = data['volume']
        
        # 첫 번째 부분: 가중 가격 변화
        weighted_price = self.add(
            self.mul(close, 0.35),
            self.mul(vwap, 0.65)
        )
        price_delta = self.ts_delta(weighted_price, 2)
        decay1 = self.ts_decayed_linear(price_delta, 3)
        rank1 = self.rank(decay1)
        
        # 두 번째 부분: 거래량과 종가의 상관관계
        volume_mean_180 = self.ts_mean(volume, 180)
        corr = self.ts_corr(volume_mean_180, close, 13)
        abs_corr = self.abs(corr)
        decay2 = self.ts_decayed_linear(abs_corr, 5)
        ts_rank2 = self.ts_rank(decay2, 15)
        
        # 최대값
        max_rank = self.max(rank1, ts_rank2)
        
        # -1 곱하기
        alpha = self.mul(max_rank, -1)
        
        return alpha.fillna(0)
