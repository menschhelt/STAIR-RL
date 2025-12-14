import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_080(BaseAlpha):
    """
    alpha101_080: mul(lt(rank(log(ts_product(rank(pow(rank(ts_corr({disk:vwap},ts_sum(ts_mean({disk:amount},10),49.6054),8.47743)),4)),14.9655))),rank(ts_corr(rank({disk:vwap}),rank({disk:volume}),5.07914))),-1)
    
    VWAP-거래대금 상관관계의 복잡한 거듭제곱 곱의 로그 랭킹과 VWAP-거래량 랭킹 상관관계 랭킹 비교의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_080"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_080 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 복잡한 VWAP-거래대금 상관관계 처리
        # 1. ts_mean(amount, 10)
        amount_mean = self.ts_mean(data['amount'], 10)
        
        # 2. ts_sum(amount_mean, 49.6054) -> 반올림하여 50
        sum_window = int(round(49.6054))
        amount_sum = self.ts_sum(amount_mean, sum_window)
        
        # 3. ts_corr(vwap, amount_sum, 8.47743) -> 반올림하여 8
        corr_window1 = int(round(8.47743))
        vwap_corr = self.ts_corr(data['vwap'], amount_sum, corr_window1)
        
        # 4. rank(vwap_corr)
        corr_rank = self.rank(vwap_corr)
        
        # 5. pow(corr_rank, 4)
        corr_powered = self.pow(corr_rank, 4)
        
        # 6. rank(corr_powered)
        powered_rank = self.rank(corr_powered)
        
        # 7. ts_product(powered_rank, 14.9655) -> 반올림하여 15
        product_window = int(round(14.9655))
        product_result = self.ts_product(powered_rank, product_window)
        
        # 8. log(product_result)
        log_result = self.log(product_result)
        
        # 9. rank(log_result)
        first_rank = self.rank(log_result)
        
        # 두 번째 부분: VWAP-거래량 랭킹 상관관계
        # 10. rank(vwap)
        vwap_rank = self.rank(data['vwap'])
        
        # 11. rank(volume)
        volume_rank = self.rank(data['volume'])
        
        # 12. ts_corr(vwap_rank, volume_rank, 5.07914) -> 반올림하여 5
        corr_window2 = int(round(5.07914))
        second_corr = self.ts_corr(vwap_rank, volume_rank, corr_window2)
        
        # 13. rank(second_corr)
        second_rank = self.rank(second_corr)
        
        # 14. lt(first_rank, second_rank)
        condition = self.lt(first_rank, second_rank)
        
        # 15. mul(condition, -1)
        alpha = self.mul(condition.astype(float), -1)
        
        return alpha.fillna(0)
