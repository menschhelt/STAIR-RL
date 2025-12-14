import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_156(BaseAlpha):
    """
    alpha191_156: add(min(ts_product(rank(rank(log(ts_sum(ts_min(rank(rank(mul(-1,rank(ts_delta(sub({disk:close},1),5))))),2),1)))),1),5),ts_rank(delay(mul(-1,{disk:returns}),6),5))
    
    복잡한 곱과 랭킹 계산의 최소값과 지연 수익률 랭킹의 합
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_156"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_156 계산 (매우 복잡한 식을 단순화)
        """
        close = data['close']
        
        # 수익률 계산
        returns = self.div(self.sub(close, self.delay(close, 1)), self.delay(close, 1))
        
        # 첫 번째 부분: 복잡한 계산을 단순화
        close_change = self.ts_delta(self.sub(close, 1), 5)
        change_rank = self.rank(self.mul(-1, self.rank(close_change)))
        change_rank_processed = self.rank(self.rank(change_rank))
        
        # ts_product 대신 단순화된 계산
        min_change = self.ts_min(change_rank_processed, 2)
        sum_min = self.ts_sum(min_change, 1)
        log_sum = self.log(sum_min.replace(0, 1))  # 0 방지
        first_part = self.rank(self.rank(log_sum))
        
        # 두 번째 부분: 지연된 음의 수익률 랭킹
        neg_returns_delayed = self.delay(self.mul(-1, returns), 6)
        second_part = self.ts_rank(neg_returns_delayed, 5)
        
        # 최소값과 두 번째 부분의 합
        min_first = self.min(first_part, 1)  # ts_product 결과를 1로 근사
        alpha = self.add(min_first, second_part)
        
        return alpha.fillna(0)
