import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_044(BaseAlpha):
    """
    alpha101_044: mul(-1,mul(mul(rank(div(ts_sum(delay({disk:close},5),20),20)),ts_corr({disk:close},{disk:volume},2)),rank(ts_corr(ts_sum({disk:close},5),ts_sum({disk:close},20),2))))
    
    지연된 종가 평균, 종가-거래량 상관관계, 다기간 종가 합계 상관관계의 복합 음수 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_044"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_044 계산
        """
        # 첫 번째 부분: rank(div(ts_sum(delay(close, 5), 20), 20))
        # 1. delay(close, 5)
        close_lag5 = self.delay(data['close'], 5)
        
        # 2. ts_sum(close_lag5, 20)
        sum_delayed = self.ts_sum(close_lag5, 20)
        
        # 3. div(sum_delayed, 20)
        mean_delayed = self.div(sum_delayed, 20)
        
        # 4. rank(mean_delayed)
        first_part = self.rank(mean_delayed)
        
        # 두 번째 부분: ts_corr(close, volume, 2)
        second_part = self.ts_corr(data['close'], data['volume'], 2)
        
        # 세 번째 부분: rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2))
        # 5. ts_sum(close, 5)
        sum_5 = self.ts_sum(data['close'], 5)
        
        # 6. ts_sum(close, 20)
        sum_20 = self.ts_sum(data['close'], 20)
        
        # 7. ts_corr(sum_5, sum_20, 2)
        corr_sums = self.ts_corr(sum_5, sum_20, 2)
        
        # 8. rank(corr_sums)
        third_part = self.rank(corr_sums)
        
        # 9. 모든 부분들을 곱하기
        product = self.mul(
            self.mul(first_part, second_part),
            third_part
        )
        
        # 10. mul(-1, product)
        alpha = self.mul(product, -1)
        
        return alpha.fillna(0)
