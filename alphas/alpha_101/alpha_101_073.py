import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_073(BaseAlpha):
    """
    alpha101_073: mul(lt(rank(ts_corr({disk:close},ts_sum(ts_mean({disk:amount},30),37.4843),15.1365)),rank(ts_corr(rank(add(mul({disk:high},0.0261661),mul({disk:vwap},sub(1,0.0261661)))),rank({disk:volume}),11.4791))),-1)
    
    종가-거래대금 상관관계 랭킹과 가중 고가-VWAP 랭킹-거래량 상관관계 랭킹 비교의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_073"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_073 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 종가-거래대금 상관관계
        # 1. ts_mean(amount, 30)
        amount_mean = self.ts_mean(data['amount'], 30)
        
        # 2. ts_sum(amount_mean, 37.4843) -> 반올림하여 37
        sum_window = int(round(37.4843))
        amount_sum = self.ts_sum(amount_mean, sum_window)
        
        # 3. ts_corr(close, amount_sum, 15.1365) -> 반올림하여 15
        corr_window1 = int(round(15.1365))
        close_corr = self.ts_corr(data['close'], amount_sum, corr_window1)
        
        # 4. rank(close_corr)
        first_rank = self.rank(close_corr)
        
        # 두 번째 부분: 가중 고가-VWAP 랭킹과 거래량 상관관계
        weight = 0.0261661
        
        # 5. 가중 고가-VWAP: high * 0.0261661 + vwap * (1-0.0261661)
        weighted_high_vwap = self.add(
            self.mul(data['high'], weight),
            self.mul(data['vwap'], 1 - weight)
        )
        
        # 6. rank(weighted_high_vwap)
        weighted_rank = self.rank(weighted_high_vwap)
        
        # 7. rank(volume)
        volume_rank = self.rank(data['volume'])
        
        # 8. ts_corr(weighted_rank, volume_rank, 11.4791) -> 반올림하여 11
        corr_window2 = int(round(11.4791))
        weighted_corr = self.ts_corr(weighted_rank, volume_rank, corr_window2)
        
        # 9. rank(weighted_corr)
        second_rank = self.rank(weighted_corr)
        
        # 10. lt(first_rank, second_rank)
        condition = self.lt(first_rank, second_rank)
        
        # 11. mul(condition, -1)
        alpha = self.mul(condition.astype(float), -1)
        
        return alpha.fillna(0)
