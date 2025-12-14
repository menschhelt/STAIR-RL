import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_086(BaseAlpha):
    """
    alpha191_086: mul(add(rank(ts_decayed_linear(ts_delta({disk:vwap},4),7)),ts_rank(ts_decayed_linear(div(sub(add(mul({disk:low},0.9),mul({disk:low},0.1)),{disk:vwap}),sub({disk:open},div(add({disk:high},{disk:low}),2))),11),7)),-1)
    
    -1 * (VWAP 변화 decayed linear 랭킹 + 복잡한 가격 비율 decayed linear ts_rank)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_086"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_086 계산
        """
        vwap = data['vwap']
        low = data['low']
        high = data['high']
        open_price = data['open']
        
        # 첫 번째 부분: VWAP 4일 변화의 7일 decayed linear
        vwap_delta = self.ts_delta(vwap, 4)
        decay1 = self.ts_decayed_linear(vwap_delta, 7)
        rank1 = self.rank(decay1)
        
        # 두 번째 부분: 복잡한 가격 비율
        # (low * 0.9 + low * 0.1) = low * (0.9 + 0.1) = low
        weighted_low = self.add(
            self.mul(low, 0.9),
            self.mul(low, 0.1)
        )
        
        # 분자: weighted_low - vwap
        numerator = self.sub(weighted_low, vwap)
        
        # 분모: open - (high + low) / 2
        hl_mid = self.div(self.add(high, low), 2)
        denominator = self.sub(open_price, hl_mid)
        
        # 비율
        ratio = self.div(numerator, denominator)
        decay2 = self.ts_decayed_linear(ratio, 11)
        ts_rank2 = self.ts_rank(decay2, 7)
        
        # 합계
        sum_ranks = self.add(rank1, ts_rank2)
        
        # -1 곱하기
        alpha = self.mul(sum_ranks, -1)
        
        return alpha.fillna(0)
