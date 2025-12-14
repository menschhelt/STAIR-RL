import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_036(BaseAlpha):
    """
    alpha101_036: add(rank(ts_corr(delay(sub({disk:open},{disk:close}),1),{disk:close},200)),rank(sub({disk:open},{disk:close})))
    
    전일 시가-종가 차이와 현재 종가의 장기 상관관계 + 현재 시가-종가 차이 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_036"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_036 계산
        """
        # 1. sub(open, close)
        open_close_diff = self.sub(data['open'], data['close'])
        
        # 2. delay(open_close_diff, 1)
        delayed_diff = self.delay(open_close_diff, 1)
        
        # 3. ts_corr(delayed_diff, close, 200)
        corr_200 = self.ts_corr(delayed_diff, data['close'], 200)
        
        # 4. rank(corr_200)
        first_part = self.rank(corr_200)
        
        # 5. rank(open_close_diff)
        second_part = self.rank(open_close_diff)
        
        # 6. add(first_part, second_part)
        alpha = self.add(first_part, second_part)
        
        return alpha.fillna(0)
