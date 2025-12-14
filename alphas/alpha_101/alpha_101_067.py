import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_067(BaseAlpha):
    """
    alpha101_067: mul(lt(ts_rank(ts_corr(rank({disk:high}),rank(ts_mean({disk:amount},15)),8.91644),13.9333),rank(ts_delta(add(mul({disk:close},0.518371),mul({disk:low},sub(1,0.518371))),1.06157))),-1)
    
    고가-거래대금 상관관계의 시계열 랭킹과 가중 종가-저가 변화 랭킹 비교의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_067"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_067 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 고가-거래대금 상관관계의 시계열 랭킹
        # 1. rank(high)
        high_rank = self.rank(data['high'])
        
        # 2. ts_mean(amount, 15)
        amount_mean = self.ts_mean(data['amount'], 15)
        
        # 3. rank(amount_mean)
        amount_rank = self.rank(amount_mean)
        
        # 4. ts_corr(high_rank, amount_rank, 8.91644) -> 반올림하여 9
        corr_window = int(round(8.91644))
        corr_result = self.ts_corr(high_rank, amount_rank, corr_window)
        
        # 5. ts_rank(corr_result, 13.9333) -> 반올림하여 14
        rank_window = int(round(13.9333))
        first_part = self.ts_rank(corr_result, rank_window)
        
        # 두 번째 부분: 가중 종가-저가 변화
        weight = 0.518371
        
        # 6. 가중 종가-저가: close * 0.518371 + low * (1-0.518371)
        weighted_close_low = self.add(
            self.mul(data['close'], weight),
            self.mul(data['low'], 1 - weight)
        )
        
        # 7. ts_delta(weighted_close_low, 1.06157) -> 반올림하여 1
        delta_window = int(round(1.06157))
        price_delta = self.ts_delta(weighted_close_low, delta_window)
        
        # 8. rank(price_delta)
        second_part = self.rank(price_delta)
        
        # 9. lt(first_part, second_part)
        condition = self.lt(first_part, second_part)
        
        # 10. mul(condition, -1)
        alpha = self.mul(condition.astype(float), -1)
        
        return alpha.fillna(0)
