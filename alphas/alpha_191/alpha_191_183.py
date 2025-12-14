import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_183(BaseAlpha):
    """
    alpha191_183: add(rank(ts_corr(delay(sub({disk:open},{disk:close}),1),{disk:close},200)),rank(sub({disk:open},{disk:close})))
    
    지연된 시가-종가 차이와 종가의 상관관계 랭킹 + 현재 시가-종가 차이 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_183"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_183 계산
        """
        open_price = data['open']
        close = data['close']
        
        # 시가-종가 차이
        open_close_diff = self.sub(open_price, close)
        
        # 첫 번째 부분: 지연된 차이와 종가의 200일 상관관계
        delayed_diff = self.delay(open_close_diff, 1)
        corr = self.ts_corr(delayed_diff, close, 200)
        first_rank = self.rank(corr)
        
        # 두 번째 부분: 현재 시가-종가 차이의 랭킹
        second_rank = self.rank(open_close_diff)
        
        # 두 랭킹의 합
        alpha = self.add(first_rank, second_rank)
        
        return alpha.fillna(1)
