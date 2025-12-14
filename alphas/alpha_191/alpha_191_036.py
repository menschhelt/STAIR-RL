import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_036(BaseAlpha):
    """
    alpha191_036: mul(-1,rank(sub(mul(ts_sum({disk:open},5),ts_sum({disk:returns},5)),delay(mul(ts_sum({disk:open},5),ts_sum({disk:returns},5)),10))))
    
    -1 * (시가 합계 * 수익률 합계)의 10일 변화에 대한 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_036"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_036 계산
        """
        # returns 필드 확인 및 생성
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        open_price = data['open']
        returns = data['returns']
        
        # 5일 합계
        open_sum_5 = self.ts_sum(open_price, 5)
        returns_sum_5 = self.ts_sum(returns, 5)
        
        # 곱하기
        product = self.mul(open_sum_5, returns_sum_5)
        
        # 10일 변화
        product_change = self.sub(product, self.delay(product, 10))
        
        # 랭킹
        product_rank = self.rank(product_change)
        
        # -1 곱하기
        alpha = self.mul(-1, product_rank)
        
        return alpha.fillna(0)
