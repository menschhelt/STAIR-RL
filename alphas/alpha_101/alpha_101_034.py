import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_034(BaseAlpha):
    """
    alpha101_034: mul(mul(ts_rank({disk:volume},32),sub(1,ts_rank(sub(add({disk:close},{disk:high}),{disk:low}),16))),sub(1,ts_rank({disk:returns},32)))
    
    거래량, 가격 위치, 수익률의 시계열 랭킹을 조합한 복합 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_034"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_034 계산
        """
        # 수익률 계산
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        # 1. ts_rank(volume, 32)
        volume_rank = self.ts_rank(data['volume'], 32)
        
        # 2. add(close, high)
        close_high_sum = self.add(data['close'], data['high'])
        
        # 3. sub(close_high_sum, low)
        price_range = self.sub(close_high_sum, data['low'])
        
        # 4. ts_rank(price_range, 16)
        price_rank = self.ts_rank(price_range, 16)
        
        # 5. sub(1, price_rank)
        one_minus_price = (1 - price_rank)
        
        # 6. ts_rank(returns, 32)
        returns_rank = self.ts_rank(data['returns'], 32)
        
        # 7. sub(1, returns_rank)
        one_minus_returns = (1 - returns_rank)
        
        # 8. 모든 것들을 곱하기
        alpha = self.mul(
            self.mul(volume_rank, one_minus_price),
            one_minus_returns
        )
        
        return alpha.fillna(0)
