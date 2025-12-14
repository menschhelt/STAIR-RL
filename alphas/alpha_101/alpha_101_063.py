import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_063(BaseAlpha):
    """
    alpha101_063: mul(lt(rank(ts_corr(ts_sum(add(mul({disk:open},0.178404),mul({disk:low},sub(1,0.178404))),12.7054),ts_sum(ts_mean({disk:amount},120),12.7054),16.6208)),rank(ts_delta(add(mul(div(add({disk:high},{disk:low}),2),0.178404),mul({disk:vwap},sub(1,0.178404))),3.69741))),-1)
    
    가중 시가-저가와 거래대금의 상관관계 랭킹과 가중 중간가-VWAP 변화 랭킹 비교의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_063"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_063 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        weight = 0.178404
        
        # 첫 번째 부분: 가중 시가-저가와 거래대금 상관관계
        # 1. 가중 시가-저가: open * 0.178404 + low * (1-0.178404)
        weighted_open_low = self.add(
            self.mul(data['open'], weight),
            self.mul(data['low'], 1 - weight)
        )
        
        # 2. ts_sum(weighted_open_low, 12.7054) -> 반올림하여 13
        sum_window1 = int(round(12.7054))
        weighted_sum = self.ts_sum(weighted_open_low, sum_window1)
        
        # 3. ts_mean(amount, 120)
        amount_mean = self.ts_mean(data['amount'], 120)
        
        # 4. ts_sum(amount_mean, 12.7054) -> 반올림하여 13
        amount_sum = self.ts_sum(amount_mean, sum_window1)
        
        # 5. ts_corr(weighted_sum, amount_sum, 16.6208) -> 반올림하여 17
        corr_window = int(round(16.6208))
        first_corr = self.ts_corr(weighted_sum, amount_sum, corr_window)
        
        # 6. rank(first_corr)
        first_rank = self.rank(first_corr)
        
        # 두 번째 부분: 가중 중간가-VWAP 변화
        # 7. div(add(high, low), 2) - 중간가격
        mid_price = self.div(
            self.add(data['high'], data['low']),
            2
        )
        
        # 8. 가중 중간가-VWAP: mid_price * 0.178404 + vwap * (1-0.178404)
        weighted_mid_vwap = self.add(
            self.mul(mid_price, weight),
            self.mul(data['vwap'], 1 - weight)
        )
        
        # 9. ts_delta(weighted_mid_vwap, 3.69741) -> 반올림하여 4
        delta_window = int(round(3.69741))
        price_delta = self.ts_delta(weighted_mid_vwap, delta_window)
        
        # 10. rank(price_delta)
        second_rank = self.rank(price_delta)
        
        # 11. lt(first_rank, second_rank)
        condition = self.lt(first_rank, second_rank)
        
        # 12. mul(condition, -1)
        alpha = self.mul(condition.astype(float), -1)
        
        return alpha.fillna(0)
