import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_095(BaseAlpha):
    """
    alpha101_095: mul(max(ts_rank(ts_decayed_linear(ts_corr(rank({disk:vwap}),rank({disk:volume}),3.83878),4.16783),8.38151),ts_rank(ts_decayed_linear(ts_argmax(ts_corr(ts_rank({disk:close},7.45404),ts_rank(ts_mean({disk:amount},60),4.13242),3.65459),12.6556),14.0365),13.4143)),-1)
    
    VWAP-거래량 랭킹 상관관계의 감쇠선형 시계열 랭킹과 종가-거래대금 시계열 랭킹 상관관계의 argmax 감쇠선형 시계열 랭킹의 최대값의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_095"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_095 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: VWAP-거래량 랭킹 상관관계의 감쇠선형 시계열 랭킹
        # 1. rank(vwap)
        vwap_rank = self.rank(data['vwap'])
        
        # 2. rank(volume)
        volume_rank = self.rank(data['volume'])
        
        # 3. ts_corr(vwap_rank, volume_rank, 3.83878) -> 반올림하여 4
        corr_window1 = int(round(3.83878))
        vwap_vol_corr = self.ts_corr(vwap_rank, volume_rank, corr_window1)
        
        # 4. ts_decayed_linear(vwap_vol_corr, 4.16783) -> 반올림하여 4
        decay_window1 = int(round(4.16783))
        first_decayed = self.ts_decayed_linear(vwap_vol_corr, decay_window1)
        
        # 5. ts_rank(first_decayed, 8.38151) -> 반올림하여 8
        rank_window1 = int(round(8.38151))
        first_part = self.ts_rank(first_decayed, rank_window1)
        
        # 두 번째 부분: 종가-거래대금 시계열 랭킹 상관관계의 argmax 감쇠선형 시계열 랭킹
        # 6. ts_rank(close, 7.45404) -> 반올림하여 7
        close_rank_window = int(round(7.45404))
        close_tsrank = self.ts_rank(data['close'], close_rank_window)
        
        # 7. ts_mean(amount, 60)
        amount_mean = self.ts_mean(data['amount'], 60)
        
        # 8. ts_rank(amount_mean, 4.13242) -> 반올림하여 4
        amount_rank_window = int(round(4.13242))
        amount_tsrank = self.ts_rank(amount_mean, amount_rank_window)
        
        # 9. ts_corr(close_tsrank, amount_tsrank, 3.65459) -> 반올림하여 4
        corr_window2 = int(round(3.65459))
        close_amount_corr = self.ts_corr(close_tsrank, amount_tsrank, corr_window2)
        
        # 10. ts_argmax(close_amount_corr, 12.6556) -> 반올림하여 13
        argmax_window = int(round(12.6556))
        corr_argmax = self.ts_argmax(close_amount_corr, argmax_window)
        
        # 11. ts_decayed_linear(corr_argmax, 14.0365) -> 반올림하여 14
        decay_window2 = int(round(14.0365))
        second_decayed = self.ts_decayed_linear(corr_argmax, decay_window2)
        
        # 12. ts_rank(second_decayed, 13.4143) -> 반올림하여 13
        rank_window2 = int(round(13.4143))
        second_part = self.ts_rank(second_decayed, rank_window2)
        
        # 13. max(first_part, second_part)
        max_result = self.max(first_part, second_part)
        
        # 14. mul(max_result, -1)
        alpha = self.mul(max_result, -1)
        
        return alpha.fillna(0)
