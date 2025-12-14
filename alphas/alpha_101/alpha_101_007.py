import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

# 부모 디렉토리를 import 경로에 추가
from alphas.base.base import BaseAlpha

class alpha_101_007(BaseAlpha):
    """
    Alpha 101_007: mul(-1,rank(sub(mul(ts_sum({disk:open},5),ts_sum({disk:returns},5)),delay(mul(ts_sum({disk:open},5),ts_sum({disk:returns},5)),10))))
    
    시가-수익률 조합의 지연 비교 랭킹
    """
    neutralizer_type: str = "mean"  # 평균 중립화

    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_007"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_007 계산
        """
        # 수익률 계산
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        # 1. ts_sum(open, 5)
        open_sum_5 = self.ts_sum(data['open'], 5)

        # 2. ts_sum(returns, 5)
        returns_sum_5 = self.ts_sum(data['returns'], 5)

        # 3. mul(ts_sum(open, 5), ts_sum(returns, 5))
        current_product = self.mul(open_sum_5, returns_sum_5)

        # 4. delay(..., 10)
        delayed_product = self.delay(current_product, 10)

        # 5. sub(current_product, delayed_product)
        diff = self.sub(current_product, delayed_product)

        # 6. rank(diff)
        ranked = self.rank(diff)

        # 7. -1 * ranked
        alpha = self.mul(ranked, -1)
        
        return alpha.fillna(0)
