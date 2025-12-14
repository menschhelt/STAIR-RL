import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_076(BaseAlpha):
    """
    alpha101_076: min(rank(ts_decayed_linear(sub(add(div(add({disk:high},{disk:low}),2),{disk:high}),add({disk:vwap},{disk:high})),20.0451)),rank(ts_decayed_linear(ts_corr(div(add({disk:high},{disk:low}),2),ts_mean({disk:amount},40),3.1614),5.64125)))
    
    중간가-고가 합과 VWAP-고가 합의 차이 감쇠선형 랭킹과 중간가-거래대금 상관관계 감쇠선형 랭킹의 최소값
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_076"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_076 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 가격 차이의 감쇠선형 랭킹
        # 1. div(add(high, low), 2) - 중간가격
        mid_price = self.div(
            self.add(data['high'], data['low']),
            2
        )
        
        # 2. add(mid_price, high)
        mid_high_sum = self.add(mid_price, data['high'])
        
        # 3. add(vwap, high)
        vwap_high_sum = self.add(data['vwap'], data['high'])
        
        # 4. sub(mid_high_sum, vwap_high_sum)
        price_diff = self.sub(mid_high_sum, vwap_high_sum)
        
        # 5. ts_decayed_linear(price_diff, 20.0451) -> 반올림하여 20
        decay_window1 = int(round(20.0451))
        price_decayed = self.ts_decayed_linear(price_diff, decay_window1)
        
        # 6. rank(price_decayed)
        first_rank = self.rank(price_decayed)
        
        # 두 번째 부분: 중간가-거래대금 상관관계 감쇠선형 랭킹
        # 7. ts_mean(amount, 40)
        amount_mean = self.ts_mean(data['amount'], 40)
        
        # 8. ts_corr(mid_price, amount_mean, 3.1614) -> 반올림하여 3
        corr_window = int(round(3.1614))
        corr_result = self.ts_corr(mid_price, amount_mean, corr_window)
        
        # 9. ts_decayed_linear(corr_result, 5.64125) -> 반올림하여 6
        decay_window2 = int(round(5.64125))
        corr_decayed = self.ts_decayed_linear(corr_result, decay_window2)
        
        # 10. rank(corr_decayed)
        second_rank = self.rank(corr_decayed)
        
        # 11. min(first_rank, second_rank)
        alpha = self.min(first_rank, second_rank)
        
        return alpha.fillna(0)
