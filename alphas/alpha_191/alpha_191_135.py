import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_135(BaseAlpha):
    """
    alpha191_135: mul(mul(mul(-1,rank(ts_delta({disk:returns},3))),ts_corr({disk:open},{disk:volume},10))
    
    -1 * 3일 수익률 변화 랭킹과 시가-거래량 10일 상관관계의 곱
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_135"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_135 계산
        """
        # 수익률 계산
        returns = self.div(self.sub(data['close'], self.delay(data['close'], 1)), self.delay(data['close'], 1))
        
        # 3일 수익률 변화를 랭킹
        returns_delta = self.ts_delta(returns, 3)
        returns_rank = self.rank(returns_delta)
        
        # 시가와 거래량의 10일 상관관계
        open_volume_corr = self.ts_corr(data['open'], data['volume'], 10)
        
        # -1 * 랭킹 * 상관관계
        alpha = self.mul(self.mul(self.mul(-1, returns_rank), open_volume_corr), 1)
        
        return alpha.fillna(0)
