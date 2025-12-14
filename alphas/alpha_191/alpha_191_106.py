import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_106(BaseAlpha):
    """
    alpha191_106: mul(mul(mul(-1,rank(sub({disk:open},delay({disk:high},1)))),rank(sub({disk:open},delay({disk:close},1)))),rank(sub({disk:open},delay({disk:low},1))))
    
    -1 * (시가 갭 랭킹들의 곱)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_106"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_106 계산
        """
        open_price = data['open']
        high = data['high']
        close = data['close']
        low = data['low']
        
        # 지연된 값들
        high_lag1 = self.delay(high, 1)
        close_lag1 = self.delay(close, 1)
        low_lag1 = self.delay(low, 1)
        
        # 시가와 전일 각 가격의 차이
        open_high_gap = self.sub(open_price, high_lag1)
        open_close_gap = self.sub(open_price, close_lag1)
        open_low_gap = self.sub(open_price, low_lag1)
        
        # 랭킹
        rank1 = self.rank(open_high_gap)
        rank2 = self.rank(open_close_gap)
        rank3 = self.rank(open_low_gap)
        
        # 곱하기
        product = self.mul(self.mul(rank1, rank2), rank3)
        
        # -1 곱하기
        alpha = self.mul(-1, product)
        
        return alpha.fillna(0)
