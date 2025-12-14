import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_024(BaseAlpha):
    """
    alpha101_024: rank(mul(mul(mul(mul(-1,{disk:returns}),ts_mean({disk:amount},20)),{disk:vwap}),sub({disk:high},{disk:close})))
    
    수익률, 거래대금, VWAP, 고가-종가 차이의 복합 팩터 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_024"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_024 계산
        """
        # 수익률 계산
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 1. mul(-1, returns)
        neg_returns = self.mul(data['returns'], -1)
        
        # 2. ts_mean(amount, 20)
        amount_mean = self.ts_mean(data['amount'], 20)
        
        # 3. sub(high, close)
        high_close_diff = self.sub(data['high'], data['close'])
        
        # 4. 모든 것들을 곱하기
        product = self.mul(
            self.mul(
                self.mul(neg_returns, amount_mean),
                data['vwap']
            ),
            high_close_diff
        )
        
        # 5. rank(...)
        alpha = self.rank(product)
        
        return alpha.fillna(0)
