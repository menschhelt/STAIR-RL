import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_061(BaseAlpha):
    """
    alpha101_061: mul(lt(rank(ts_corr({disk:vwap},ts_sum(ts_mean({disk:amount},20),22.4101),9.91009)),rank(lt(add(rank({disk:open}),rank({disk:open})),add(rank(div(add({disk:high},{disk:low}),2)),rank({disk:high}))))),-1)
    
    VWAP-거래대금 상관관계 랭킹과 가격 랭킹 비교의 복합 조건부 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_061"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_061 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: VWAP-거래대금 상관관계
        # 1. ts_mean(amount, 20)
        amount_mean = self.ts_mean(data['amount'], 20)
        
        # 2. ts_sum(amount_mean, 22.4101) -> 반올림하여 22
        sum_window = int(round(22.4101))
        amount_sum = self.ts_sum(amount_mean, sum_window)
        
        # 3. ts_corr(vwap, amount_sum, 9.91009) -> 반올림하여 10
        corr_window = int(round(9.91009))
        vwap_corr = self.ts_corr(data['vwap'], amount_sum, corr_window)
        
        # 4. rank(vwap_corr)
        first_rank = self.rank(vwap_corr)
        
        # 두 번째 부분: 가격 랭킹 비교
        # 5. rank(open)
        open_rank = self.rank(data['open'])
        
        # 6. add(open_rank, open_rank) = 2 * open_rank
        open_double = self.add(open_rank, open_rank)
        
        # 7. div(add(high, low), 2) - 중간가격
        mid_price = self.div(
            self.add(data['high'], data['low']),
            2
        )
        
        # 8. rank(mid_price)
        mid_rank = self.rank(mid_price)
        
        # 9. rank(high)
        high_rank = self.rank(data['high'])
        
        # 10. add(mid_rank, high_rank)
        price_sum = self.add(mid_rank, high_rank)
        
        # 11. lt(open_double, price_sum)
        price_condition = self.lt(open_double, price_sum)
        
        # 12. rank(price_condition)
        second_rank = self.rank(price_condition.astype(float))
        
        # 13. lt(first_rank, second_rank)
        main_condition = self.lt(first_rank, second_rank)
        
        # 14. mul(main_condition, -1)
        alpha = self.mul(main_condition.astype(float), -1)
        
        return alpha.fillna(0)
