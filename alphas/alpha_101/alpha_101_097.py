import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_097(BaseAlpha):
    """
    alpha101_097: sub(rank(ts_decayed_linear(ts_corr({disk:vwap},ts_sum(ts_mean({disk:amount},5),26.4719),4.58418),7.18088)),rank(ts_decayed_linear(ts_rank(ts_argmin(ts_corr(rank({disk:open}),rank(ts_mean({disk:amount},15)),20.8187),8.62571),6.95668),8.07206)))
    
    VWAP-거래대금 상관관계의 감쇠선형 랭킹에서 시가-거래대금 랭킹 상관관계의 argmin 복잡한 처리 랭킹을 뺀 값
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_097"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_097 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: VWAP-거래대금 상관관계의 감쇠선형 랭킹
        # 1. ts_mean(amount, 5)
        amount_mean_5 = self.ts_mean(data['amount'], 5)
        
        # 2. ts_sum(amount_mean_5, 26.4719) -> 반올림하여 26
        sum_window = int(round(26.4719))
        amount_sum = self.ts_sum(amount_mean_5, sum_window)
        
        # 3. ts_corr(vwap, amount_sum, 4.58418) -> 반올림하여 5
        corr_window1 = int(round(4.58418))
        vwap_corr = self.ts_corr(data['vwap'], amount_sum, corr_window1)
        
        # 4. ts_decayed_linear(vwap_corr, 7.18088) -> 반올림하여 7
        decay_window1 = int(round(7.18088))
        vwap_decayed = self.ts_decayed_linear(vwap_corr, decay_window1)
        
        # 5. rank(vwap_decayed)
        first_part = self.rank(vwap_decayed)
        
        # 두 번째 부분: 시가-거래대금 랭킹 상관관계의 argmin 복잡한 처리
        # 6. rank(open)
        open_rank = self.rank(data['open'])
        
        # 7. ts_mean(amount, 15)
        amount_mean_15 = self.ts_mean(data['amount'], 15)
        
        # 8. rank(amount_mean_15)
        amount_rank = self.rank(amount_mean_15)
        
        # 9. ts_corr(open_rank, amount_rank, 20.8187) -> 반올림하여 21
        corr_window2 = int(round(20.8187))
        open_amount_corr = self.ts_corr(open_rank, amount_rank, corr_window2)
        
        # 10. ts_argmin(open_amount_corr, 8.62571) -> 반올림하여 9
        argmin_window = int(round(8.62571))
        corr_argmin = self.ts_argmin(open_amount_corr, argmin_window)
        
        # 11. ts_rank(corr_argmin, 6.95668) -> 반올림하여 7
        argmin_rank_window = int(round(6.95668))
        argmin_ranked = self.ts_rank(corr_argmin, argmin_rank_window)
        
        # 12. ts_decayed_linear(argmin_ranked, 8.07206) -> 반올림하여 8
        decay_window2 = int(round(8.07206))
        argmin_decayed = self.ts_decayed_linear(argmin_ranked, decay_window2)
        
        # 13. rank(argmin_decayed)
        second_part = self.rank(argmin_decayed)
        
        # 14. sub(first_part, second_part)
        alpha = self.sub(first_part, second_part)
        
        return alpha.fillna(0)
