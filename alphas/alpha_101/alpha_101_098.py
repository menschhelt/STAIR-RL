import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_098(BaseAlpha):
    """
    alpha101_098: mul(lt(rank(ts_corr(ts_sum(div(add({disk:high},{disk:low}),2),19.8975),ts_sum(ts_mean({disk:amount},60),19.8975),8.8136)),rank(ts_corr({disk:low},{disk:volume},6.28259))),-1)
    
    중간가 합계와 거래대금 합계의 상관관계 랭킹이 저가-거래량 상관관계 랭킹보다 작은지 비교한 값의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_098"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_098 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 중간가 합계와 거래대금 합계의 상관관계 랭킹
        # 1. div(add(high, low), 2) - 중간가격
        mid_price = self.div(
            self.add(data['high'], data['low']),
            2
        )
        
        # 2. ts_sum(mid_price, 19.8975) -> 반올림하여 20
        sum_window = int(round(19.8975))
        mid_sum = self.ts_sum(mid_price, sum_window)
        
        # 3. ts_mean(amount, 60)
        amount_mean = self.ts_mean(data['amount'], 60)
        
        # 4. ts_sum(amount_mean, 19.8975) -> 반올림하여 20
        amount_sum = self.ts_sum(amount_mean, sum_window)
        
        # 5. ts_corr(mid_sum, amount_sum, 8.8136) -> 반올림하여 9
        corr_window1 = int(round(8.8136))
        mid_amount_corr = self.ts_corr(mid_sum, amount_sum, corr_window1)
        
        # 6. rank(mid_amount_corr)
        first_rank = self.rank(mid_amount_corr)
        
        # 두 번째 부분: 저가-거래량 상관관계 랭킹
        # 7. ts_corr(low, volume, 6.28259) -> 반올림하여 6
        corr_window2 = int(round(6.28259))
        low_vol_corr = self.ts_corr(data['low'], data['volume'], corr_window2)
        
        # 8. rank(low_vol_corr)
        second_rank = self.rank(low_vol_corr)
        
        # 9. lt(first_rank, second_rank)
        condition = self.lt(first_rank, second_rank)
        
        # 10. mul(condition, -1)
        alpha = self.mul(condition.astype(float), -1)
        
        return alpha.fillna(0)
