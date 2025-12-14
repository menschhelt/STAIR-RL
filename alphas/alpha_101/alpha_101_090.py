import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_090(BaseAlpha):
    """
    alpha101_090: mul(sub(ts_rank(ts_decayed_linear(ts_decayed_linear(ts_corr(grouped_demean({disk:close},{disk:industry_group_lv2}),{disk:volume},9.74928),16.398),3.83219),4.8667),rank(ts_decayed_linear(ts_corr({disk:vwap},ts_mean({disk:amount},30),4.01303),2.6809))),-1)
    
    업종별 표준화된 종가-거래량 상관관계의 이중 감쇠선형 시계열 랭킹에서 VWAP-거래대금 상관관계의 감쇠선형 랭킹을 뺀 값의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_090"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_090 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 업종별 표준화된 종가-거래량 상관관계의 이중 감쇠선형 시계열 랭킹
        # 1. grouped_demean(close, industry_group_lv2) -> close - mean(close)
        close_demeaned = data['close'] - data['close'].mean()
        
        # 2. ts_corr(close_demeaned, volume, 9.74928) -> 반올림하여 10
        corr_window1 = int(round(9.74928))
        close_vol_corr = self.ts_corr(close_demeaned, data['volume'], corr_window1)
        
        # 3. ts_decayed_linear(close_vol_corr, 16.398) -> 반올림하여 16
        decay_window1 = int(round(16.398))
        first_decayed = self.ts_decayed_linear(close_vol_corr, decay_window1)
        
        # 4. ts_decayed_linear(first_decayed, 3.83219) -> 반올림하여 4
        decay_window2 = int(round(3.83219))
        second_decayed = self.ts_decayed_linear(first_decayed, decay_window2)
        
        # 5. ts_rank(second_decayed, 4.8667) -> 반올림하여 5
        rank_window1 = int(round(4.8667))
        first_part = self.ts_rank(second_decayed, rank_window1)
        
        # 두 번째 부분: VWAP-거래대금 상관관계의 감쇠선형 랭킹
        # 6. ts_mean(amount, 30)
        amount_mean = self.ts_mean(data['amount'], 30)
        
        # 7. ts_corr(vwap, amount_mean, 4.01303) -> 반올림하여 4
        corr_window2 = int(round(4.01303))
        vwap_amount_corr = self.ts_corr(data['vwap'], amount_mean, corr_window2)
        
        # 8. ts_decayed_linear(vwap_amount_corr, 2.6809) -> 반올림하여 3
        decay_window3 = int(round(2.6809))
        vwap_decayed = self.ts_decayed_linear(vwap_amount_corr, decay_window3)
        
        # 9. rank(vwap_decayed)
        second_part = self.rank(vwap_decayed)
        
        # 10. sub(first_part, second_part)
        diff = self.sub(first_part, second_part)
        
        # 11. mul(diff, -1)
        alpha = self.mul(diff, -1)
        
        return alpha.fillna(0)
