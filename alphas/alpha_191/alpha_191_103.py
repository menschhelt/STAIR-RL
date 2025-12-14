import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_103(BaseAlpha):
    """
    alpha191_103: mul(-1,mul(ts_delta(ts_corr({disk:high},{disk:volume},5),5),rank(ts_std({disk:close},20))))
    
    -1 * (고가-거래량 상관관계의 5일 변화 * 종가 표준편차 랭킹)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_103"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_103 계산
        """
        high = data['high']
        volume = data['volume']
        close = data['close']
        
        # 고가와 거래량의 5일 상관관계
        corr = self.ts_corr(high, volume, 5)
        
        # 상관관계의 5일 변화
        corr_delta = self.ts_delta(corr, 5)
        
        # 종가의 20일 표준편차
        close_std = self.ts_std(close, 20)
        close_std_rank = self.rank(close_std)
        
        # 곱하기 후 -1 곱하기
        alpha = self.mul(-1, self.mul(corr_delta, close_std_rank))
        
        return alpha.fillna(0)
