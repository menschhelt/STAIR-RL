import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_155(BaseAlpha):
    """
    alpha191_155: mul(max(rank(ts_decayed_linear(ts_delta({disk:vwap},5),3)),rank(ts_decayed_linear(mul(div(ts_delta(add(mul({disk:open},0.15),mul({disk:low},0.85)),2),add(mul({disk:open},0.15),mul({disk:low},0.85))),-1),3))),-1)
    
    VWAP 변화와 가격 조합 변화의 최대 랭킹에 -1 곱하기
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_155"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_155 계산
        """
        vwap = data['vwap']
        open_price = data['open']
        low = data['low']
        
        # 첫 번째 부분: rank(ts_decayed_linear(ts_delta(vwap, 5), 3))
        vwap_delta = self.ts_delta(vwap, 5)
        first_part = self.rank(self.ts_decayed_linear(vwap_delta, 3))
        
        # 두 번째 부분: 가격 조합 변화
        price_combo = self.add(self.mul(open_price, 0.15), self.mul(low, 0.85))
        price_delta = self.ts_delta(price_combo, 2)
        price_ratio = self.div(price_delta, price_combo)
        price_ratio_neg = self.mul(price_ratio, -1)
        second_part = self.rank(self.ts_decayed_linear(price_ratio_neg, 3))
        
        # 최대값에 -1 곱하기
        max_rank = self.max(first_part, second_part)
        alpha = self.mul(max_rank, -1)
        
        return alpha.fillna(0)
