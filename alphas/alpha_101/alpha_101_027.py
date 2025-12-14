import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_027(BaseAlpha):
    """
    alpha101_027: twise_a_scale(sub(add(ts_corr(ts_mean({disk:amount},20),{disk:low},5),div(add({disk:high},{disk:low}),2)),{disk:close}),1)
    
    거래대금 평균과 저가의 상관관계, 중간가격, 종가의 복합 정규화 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_027"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_027 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 1. ts_mean(amount, 20)
        amount_mean = self.ts_mean(data['amount'], 20)
        
        # 2. ts_corr(amount_mean, low, 5)
        corr_5 = self.ts_corr(amount_mean, data['low'], 5)
        
        # 3. div(add(high, low), 2) - 중간가격
        mid_price = self.div(
            self.add(data['high'], data['low']),
            2
        )
        
        # 4. add(corr_5, mid_price)
        sum_part = self.add(corr_5, mid_price)
        
        # 5. sub(sum_part, close)
        diff = self.sub(sum_part, data['close'])
        
        # 6. twise_a_scale(..., 1)
        alpha = self.twise_a_scale(diff, 1)
        
        return alpha.fillna(0)
