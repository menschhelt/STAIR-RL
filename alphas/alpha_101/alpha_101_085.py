import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_085(BaseAlpha):
    """
    alpha101_085: mul(lt(ts_rank(ts_corr({disk:close},ts_sum(ts_mean({disk:amount},20),14.7444),6.00049),20.4195),rank(sub(add({disk:open},{disk:close}),add({disk:vwap},{disk:open})))),-1)
    
    종가-거래대금 상관관계의 시계열 랭킹과 가격 합계 차이 랭킹 비교의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_085"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_085 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 종가-거래대금 상관관계의 시계열 랭킹
        # 1. ts_mean(amount, 20)
        amount_mean = self.ts_mean(data['amount'], 20)
        
        # 2. ts_sum(amount_mean, 14.7444) -> 반올림하여 15
        sum_window = int(round(14.7444))
        amount_sum = self.ts_sum(amount_mean, sum_window)
        
        # 3. ts_corr(close, amount_sum, 6.00049) -> 반올림하여 6
        corr_window = int(round(6.00049))
        close_corr = self.ts_corr(data['close'], amount_sum, corr_window)
        
        # 4. ts_rank(close_corr, 20.4195) -> 반올림하여 20
        rank_window = int(round(20.4195))
        first_part = self.ts_rank(close_corr, rank_window)
        
        # 두 번째 부분: 가격 합계 차이 랭킹
        # 5. add(open, close)
        open_close_sum = self.add(data['open'], data['close'])
        
        # 6. add(vwap, open)
        vwap_open_sum = self.add(data['vwap'], data['open'])
        
        # 7. sub(open_close_sum, vwap_open_sum)
        price_diff = self.sub(open_close_sum, vwap_open_sum)
        
        # 8. rank(price_diff)
        second_part = self.rank(price_diff)
        
        # 9. lt(first_part, second_part)
        condition = self.lt(first_part, second_part)
        
        # 10. mul(condition, -1)
        alpha = self.mul(condition.astype(float), -1)
        
        return alpha.fillna(0)
