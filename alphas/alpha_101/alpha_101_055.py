import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_055(BaseAlpha):
    """
    alpha101_055: sub(0,mul(1,mul(rank(div(ts_sum({disk:returns},10),ts_sum(ts_sum({disk:returns},2),3))),rank(mul({disk:returns},{disk:cap})))))
    
    수익률 합계 비율 랭킹과 수익률-시가총액 곱의 랭킹을 곱한 값의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_055"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_055 계산
        """
        # 수익률 계산
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        # cap (시가총액) 계산 - volume * close로 근사
        if 'cap' not in data:
            data['cap'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 수익률 합계 비율
        # 1. ts_sum(returns, 10)
        returns_sum_10 = self.ts_sum(data['returns'], 10)
        
        # 2. ts_sum(returns, 2)
        returns_sum_2 = self.ts_sum(data['returns'], 2)
        
        # 3. ts_sum(returns_sum_2, 3)
        returns_sum_nested = self.ts_sum(returns_sum_2, 3)
        
        # 4. div(returns_sum_10, returns_sum_nested)
        returns_ratio = self.div(returns_sum_10, returns_sum_nested)
        
        # 5. rank(returns_ratio)
        ratio_rank = self.rank(returns_ratio)
        
        # 두 번째 부분: 수익률-시가총액 곱
        # 6. mul(returns, cap)
        returns_cap = self.mul(data['returns'], data['cap'])
        
        # 7. rank(returns_cap)
        cap_rank = self.rank(returns_cap)
        
        # 8. mul(ratio_rank, cap_rank)
        product = self.mul(ratio_rank, cap_rank)
        
        # 9. sub(0, product) = -product
        alpha = self.mul(product, -1)
        
        return alpha.fillna(0)
