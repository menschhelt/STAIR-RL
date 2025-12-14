import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_046(BaseAlpha):
    """
    alpha101_046: sub(mul(div(mul(rank(div(1,{disk:close})),{disk:volume}),ts_mean({disk:amount},20)),div(mul({disk:high},rank(sub({disk:high},{disk:close}))),div(ts_sum({disk:high},5),5))),rank(sub({disk:vwap},delay({disk:vwap},5))))
    
    복잡한 가격 역수, 거래량, 고가 관계와 VWAP 변화의 복합 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_046"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_046 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 복잡한 부분
        # 1. rank(div(1, close))
        inverse_close = self.div(1, data['close'])
        inverse_rank = self.rank(inverse_close)
        
        # 2. mul(inverse_rank, volume)
        rank_volume = self.mul(inverse_rank, data['volume'])
        
        # 3. ts_mean(amount, 20)
        amount_mean = self.ts_mean(data['amount'], 20)
        
        # 4. div(rank_volume, amount_mean)
        first_ratio = self.div(rank_volume, amount_mean)
        
        # 두 번째 복잡한 부분
        # 5. sub(high, close)
        high_close_diff = self.sub(data['high'], data['close'])
        
        # 6. rank(high_close_diff)
        diff_rank = self.rank(high_close_diff)
        
        # 7. mul(high, diff_rank)
        high_rank_product = self.mul(data['high'], diff_rank)
        
        # 8. div(ts_sum(high, 5), 5)
        high_mean_5 = self.div(self.ts_sum(data['high'], 5), 5)
        
        # 9. div(high_rank_product, high_mean_5)
        second_ratio = self.div(high_rank_product, high_mean_5)
        
        # 10. mul(first_ratio, second_ratio)
        product = self.mul(first_ratio, second_ratio)
        
        # 세 번째 부분: VWAP 변화
        # 11. delay(vwap, 5)
        vwap_lag5 = self.delay(data['vwap'], 5)
        
        # 12. sub(vwap, vwap_lag5)
        vwap_diff = self.sub(data['vwap'], vwap_lag5)
        
        # 13. rank(vwap_diff)
        vwap_rank = self.rank(vwap_diff)
        
        # 14. sub(product, vwap_rank)
        alpha = self.sub(product, vwap_rank)
        
        return alpha.fillna(0)
