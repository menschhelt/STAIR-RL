import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha
class alpha_101_013(BaseAlpha):
    """
    Alpha 101_013: mul(mul(-1,rank(ts_delta({disk:returns},3))),ts_corr({disk:open},{disk:volume},10))
    
    수익률 변화 랭킹과 시가-거래량 상관관계의 조합
    """
    neutralizer_type: str = "mean"  # 평균 중립화

    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_013"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_013 계산
        """
        # 수익률 계산
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        # 1. ts_delta(returns, 3)
        returns_delta_3 = self.ts_delta(data['returns'], 3)

        # 2. rank(ts_delta(returns, 3))
        rank_returns_delta = self.rank(returns_delta_3)

        # 3. -1 * rank(...)
        neg_rank_returns = self.mul(rank_returns_delta, -1)

        # 4. ts_corr(open, volume, 10)
        open_volume_corr = self.ts_corr(data['open'], data['volume'], 10)
        
        # 5. (-1 * rank(...)) * ts_corr(...)
        alpha = self.mul(neg_rank_returns, open_volume_corr)
        
        return alpha.fillna(0)
