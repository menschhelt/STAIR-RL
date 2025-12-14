import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_062(BaseAlpha):
    """
    alpha101_062: mul(sub(rank(ts_decayed_linear(ts_delta(grouped_demean({disk:close},{disk:industry_group_lv2}),2.25164),8.22237)),rank(ts_decayed_linear(ts_corr(add(mul({disk:vwap},0.318108),mul({disk:open},sub(1,0.318108))),ts_sum(ts_mean({disk:amount},180),37.2467),13.557),12.2883))),-1)
    
    업종별 표준화된 종가 변화와 가중 가격-거래대금 상관관계의 감쇠선형 랭킹 차이의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_062"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_062 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 업종별 표준화된 종가 변화
        # 1. grouped_demean(close, industry_group_lv2) -> close - mean(close)
        close_demeaned = data['close'] - data['close'].mean()
        
        # 2. ts_delta(close_demeaned, 2.25164) -> 반올림하여 2
        delta_window = int(round(2.25164))
        close_delta = self.ts_delta(close_demeaned, delta_window)
        
        # 3. ts_decayed_linear(close_delta, 8.22237) -> 반올림하여 8
        decay_window1 = int(round(8.22237))
        first_decayed = self.ts_decayed_linear(close_delta, decay_window1)
        
        # 4. rank(first_decayed)
        first_rank = self.rank(first_decayed)
        
        # 두 번째 부분: 가중 가격-거래대금 상관관계
        # 5. 가중 가격: vwap * 0.318108 + open * (1-0.318108)
        weight = 0.318108
        weighted_price = self.add(
            self.mul(data['vwap'], weight),
            self.mul(data['open'], 1 - weight)
        )
        
        # 6. ts_mean(amount, 180)
        amount_mean = self.ts_mean(data['amount'], 180)
        
        # 7. ts_sum(amount_mean, 37.2467) -> 반올림하여 37
        sum_window = int(round(37.2467))
        amount_sum = self.ts_sum(amount_mean, sum_window)
        
        # 8. ts_corr(weighted_price, amount_sum, 13.557) -> 반올림하여 14
        corr_window = int(round(13.557))
        price_corr = self.ts_corr(weighted_price, amount_sum, corr_window)
        
        # 9. ts_decayed_linear(price_corr, 12.2883) -> 반올림하여 12
        decay_window2 = int(round(12.2883))
        second_decayed = self.ts_decayed_linear(price_corr, decay_window2)
        
        # 10. rank(second_decayed)
        second_rank = self.rank(second_decayed)
        
        # 11. sub(first_rank, second_rank)
        rank_diff = self.sub(first_rank, second_rank)
        
        # 12. mul(rank_diff, -1)
        alpha = self.mul(rank_diff, -1)
        
        return alpha.fillna(0)
