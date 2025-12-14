import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_042(BaseAlpha):
    """
    alpha101_042: mul(ts_rank(div({disk:volume},ts_mean({disk:amount},20)),20),ts_rank(mul(-1,ts_delta({disk:close},7)),8))
    
    거래량 비율의 20기간 랭킹과 7일 가격 변화 음수의 8기간 랭킹의 곱
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_042"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_042 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 1. div(volume, ts_mean(amount, 20))
        amount_mean = self.ts_mean(data['amount'], 20)
        volume_ratio = self.div(data['volume'], amount_mean)
        
        # 2. ts_rank(volume_ratio, 20)
        volume_rank = self.ts_rank(volume_ratio, 20)
        
        # 3. ts_delta(close, 7)
        close_delta7 = self.ts_delta(data['close'], 7)
        
        # 4. mul(-1, close_delta7)
        neg_delta = self.mul(close_delta7, -1)
        
        # 5. ts_rank(neg_delta, 8)
        delta_rank = self.ts_rank(neg_delta, 8)
        
        # 6. mul(volume_rank, delta_rank)
        alpha = self.mul(volume_rank, delta_rank)
        
        return alpha.fillna(0)
