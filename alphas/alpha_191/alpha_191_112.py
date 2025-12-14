import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_112(BaseAlpha):
    """
    alpha191_112: mul(-1,mul(mul(rank(div(ts_sum(delay({disk:close},5),20),20)),ts_corr({disk:close},{disk:volume},2)),rank(ts_corr(ts_sum({disk:close},5),ts_sum({disk:close},20),2))))
    
    -1 * (지연 종가 평균 랭킹 * 종가-거래량 상관관계 * 종가 합계 상관관계 랭킹)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_112"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_112 계산
        """
        close = data['close']
        volume = data['volume']
        
        # 첫 번째 부분: 5일 지연된 종가의 20일 합계 / 20
        close_lag5 = self.delay(close, 5)
        delayed_avg = self.div(self.ts_sum(close_lag5, 20), 20)
        rank1 = self.rank(delayed_avg)
        
        # 두 번째 부분: 종가와 거래량의 2일 상관관계
        corr1 = self.ts_corr(close, volume, 2)
        
        # 세 번째 부분: 종가 합계들의 상관관계
        close_sum_5 = self.ts_sum(close, 5)
        close_sum_20 = self.ts_sum(close, 20)
        corr2 = self.ts_corr(close_sum_5, close_sum_20, 2)
        rank3 = self.rank(corr2)
        
        # 곱하기 후 -1 곱하기
        product = self.mul(self.mul(rank1, corr1), rank3)
        alpha = self.mul(-1, product)
        
        return alpha.fillna(0)
