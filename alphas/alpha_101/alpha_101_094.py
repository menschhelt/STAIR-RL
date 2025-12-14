import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_094(BaseAlpha):
    """
    alpha101_094: lt(rank(sub({disk:open},ts_min({disk:open},12.4105))),ts_rank(pow(rank(ts_corr(ts_sum(div(add({disk:high},{disk:low}),2),19.1351),ts_sum(ts_mean({disk:amount},40),19.1351),12.8742)),5),11.7584))
    
    시가와 최소 시가 차이의 랭킹이 중간가-거래대금 상관관계의 복잡한 거듭제곱 시계열 랭킹보다 작은지 비교
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_094"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_094 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 시가와 최소 시가 차이의 랭킹
        # 1. ts_min(open, 12.4105) -> 반올림하여 12
        min_window = int(round(12.4105))
        open_min = self.ts_min(data['open'], min_window)
        
        # 2. sub(open, open_min)
        open_diff = self.sub(data['open'], open_min)
        
        # 3. rank(open_diff)
        first_part = self.rank(open_diff)
        
        # 두 번째 부분: 중간가-거래대금 상관관계의 복잡한 거듭제곱 시계열 랭킹
        # 4. div(add(high, low), 2) - 중간가격
        mid_price = self.div(
            self.add(data['high'], data['low']),
            2
        )
        
        # 5. ts_sum(mid_price, 19.1351) -> 반올림하여 19
        sum_window = int(round(19.1351))
        mid_sum = self.ts_sum(mid_price, sum_window)
        
        # 6. ts_mean(amount, 40)
        amount_mean = self.ts_mean(data['amount'], 40)
        
        # 7. ts_sum(amount_mean, 19.1351) -> 반올림하여 19
        amount_sum = self.ts_sum(amount_mean, sum_window)
        
        # 8. ts_corr(mid_sum, amount_sum, 12.8742) -> 반올림하여 13
        corr_window = int(round(12.8742))
        corr_result = self.ts_corr(mid_sum, amount_sum, corr_window)
        
        # 9. rank(corr_result)
        corr_rank = self.rank(corr_result)
        
        # 10. pow(corr_rank, 5)
        corr_powered = self.pow(corr_rank, 5)
        
        # 11. ts_rank(corr_powered, 11.7584) -> 반올림하여 12
        final_rank_window = int(round(11.7584))
        second_part = self.ts_rank(corr_powered, final_rank_window)
        
        # 12. lt(first_part, second_part)
        alpha = self.lt(first_part, second_part)
        
        # 불린을 숫자로 변환
        alpha = alpha.astype(float)
        
        return alpha.fillna(0)
