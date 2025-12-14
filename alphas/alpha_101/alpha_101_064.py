import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_064(BaseAlpha):
    """
    alpha101_064: mul(lt(rank(ts_corr(add(mul({disk:open},0.00817205),mul({disk:vwap},sub(1,0.00817205))),ts_sum(ts_mean({disk:amount},60),8.6911),6.40374)),rank(sub({disk:open},ts_min({disk:open},13.635)))),-1)
    
    가중 시가-VWAP와 거래대금의 상관관계 랭킹과 시가-최소시가 차이 랭킹 비교의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_064"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_064 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        weight = 0.00817205
        
        # 첫 번째 부분: 가중 시가-VWAP와 거래대금 상관관계
        # 1. 가중 시가-VWAP: open * 0.00817205 + vwap * (1-0.00817205)
        weighted_open_vwap = self.add(
            self.mul(data['open'], weight),
            self.mul(data['vwap'], 1 - weight)
        )
        
        # 2. ts_mean(amount, 60)
        amount_mean = self.ts_mean(data['amount'], 60)
        
        # 3. ts_sum(amount_mean, 8.6911) -> 반올림하여 9
        sum_window = int(round(8.6911))
        amount_sum = self.ts_sum(amount_mean, sum_window)
        
        # 4. ts_corr(weighted_open_vwap, amount_sum, 6.40374) -> 반올림하여 6
        corr_window = int(round(6.40374))
        first_corr = self.ts_corr(weighted_open_vwap, amount_sum, corr_window)
        
        # 5. rank(first_corr)
        first_rank = self.rank(first_corr)
        
        # 두 번째 부분: 시가-최소시가 차이
        # 6. ts_min(open, 13.635) -> 반올림하여 14
        min_window = int(round(13.635))
        open_min = self.ts_min(data['open'], min_window)
        
        # 7. sub(open, open_min)
        open_diff = self.sub(data['open'], open_min)
        
        # 8. rank(open_diff)
        second_rank = self.rank(open_diff)
        
        # 9. lt(first_rank, second_rank)
        condition = self.lt(first_rank, second_rank)
        
        # 10. mul(condition, -1)
        alpha = self.mul(condition.astype(float), -1)
        
        return alpha.fillna(0)
