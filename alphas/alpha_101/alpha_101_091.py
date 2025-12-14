import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_091(BaseAlpha):
    """
    alpha101_091: min(ts_rank(ts_decayed_linear(lt(add(div(add({disk:high},{disk:low}),2),{disk:close}),add({disk:low},{disk:open})),14.7221),18.8683),ts_rank(ts_decayed_linear(ts_corr(rank({disk:low}),rank(ts_mean({disk:amount},30)),7.58555),6.94024),6.80584))
    
    중간가-종가 합과 저가-시가 합의 비교 조건의 감쇠선형 시계열 랭킹과 저가-거래대금 랭킹 상관관계의 복잡한 처리의 최소값
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_091"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_091 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 가격 비교 조건의 감쇠선형 시계열 랭킹
        # 1. div(add(high, low), 2) - 중간가격
        mid_price = self.div(
            self.add(data['high'], data['low']),
            2
        )
        
        # 2. add(mid_price, close)
        mid_close_sum = self.add(mid_price, data['close'])
        
        # 3. add(low, open)
        low_open_sum = self.add(data['low'], data['open'])
        
        # 4. lt(mid_close_sum, low_open_sum)
        price_condition = self.lt(mid_close_sum, low_open_sum)
        
        # 5. ts_decayed_linear(price_condition, 14.7221) -> 반올림하여 15
        decay_window1 = int(round(14.7221))
        condition_decayed = self.ts_decayed_linear(price_condition.astype(float), decay_window1)
        
        # 6. ts_rank(condition_decayed, 18.8683) -> 반올림하여 19
        rank_window1 = int(round(18.8683))
        first_part = self.ts_rank(condition_decayed, rank_window1)
        
        # 두 번째 부분: 저가-거래대금 랭킹 상관관계의 복잡한 처리
        # 7. rank(low)
        low_rank = self.rank(data['low'])
        
        # 8. ts_mean(amount, 30)
        amount_mean = self.ts_mean(data['amount'], 30)
        
        # 9. rank(amount_mean)
        amount_rank = self.rank(amount_mean)
        
        # 10. ts_corr(low_rank, amount_rank, 7.58555) -> 반올림하여 8
        corr_window = int(round(7.58555))
        low_amount_corr = self.ts_corr(low_rank, amount_rank, corr_window)
        
        # 11. ts_decayed_linear(low_amount_corr, 6.94024) -> 반올림하여 7
        decay_window2 = int(round(6.94024))
        corr_decayed = self.ts_decayed_linear(low_amount_corr, decay_window2)
        
        # 12. ts_rank(corr_decayed, 6.80584) -> 반올림하여 7
        rank_window2 = int(round(6.80584))
        second_part = self.ts_rank(corr_decayed, rank_window2)
        
        # 13. min(first_part, second_part)
        alpha = self.min(first_part, second_part)
        
        return alpha.fillna(0)
