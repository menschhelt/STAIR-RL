import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_034(BaseAlpha):
    """
    alpha191_034: mul(min(rank(ts_decayed_linear(ts_delta({disk:open},1),15)),rank(ts_decayed_linear(ts_corr({disk:volume},add(mul({disk:open},0.65),mul({disk:open},0.35)),17),7))),-1)
    
    -1 * min(시가 변화 decayed linear 랭킹, 거래량과 시가의 상관관계 decayed linear 랭킹)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_034"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_034 계산
        """
        open_price = data['open']
        volume = data['volume']
        
        # 첫 번째 부분: ts_delta(open, 1)의 15일 decayed linear
        open_delta = self.ts_delta(open_price, 1)
        decay1 = self.ts_decayed_linear(open_delta, 15)
        rank1 = self.rank(decay1)
        
        # 두 번째 부분: 거래량과 시가의 상관관계
        # add(mul(open, 0.65), mul(open, 0.35)) = open * (0.65 + 0.35) = open
        weighted_open = self.add(
            self.mul(open_price, 0.65),
            self.mul(open_price, 0.35)
        )
        corr = self.ts_corr(volume, weighted_open, 17)
        decay2 = self.ts_decayed_linear(corr, 7)
        rank2 = self.rank(decay2)
        
        # 최소값
        min_rank = self.min(rank1, rank2)
        
        # -1 곱하기
        alpha = self.mul(min_rank, -1)
        
        return alpha.fillna(0)
