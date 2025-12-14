import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_075(BaseAlpha):
    """
    alpha101_075: mul(max(rank(ts_decayed_linear(ts_delta({disk:vwap},1.24383),11.8259)),ts_rank(ts_decayed_linear(ts_rank(ts_corr(grouped_demean({disk:low},{disk:industry_group_lv1}),ts_mean({disk:amount},81),8.14941),19.569),17.1543),19.383)),-1)
    
    VWAP 변화의 감쇠선형 랭킹과 업종별 표준화된 저가-거래대금 상관관계의 복잡한 처리의 최대값의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_075"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_075 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: VWAP 변화의 감쇠선형 랭킹
        # 1. ts_delta(vwap, 1.24383) -> 반올림하여 1
        delta_window = int(round(1.24383))
        vwap_delta = self.ts_delta(data['vwap'], delta_window)
        
        # 2. ts_decayed_linear(vwap_delta, 11.8259) -> 반올림하여 12
        decay_window1 = int(round(11.8259))
        vwap_decayed = self.ts_decayed_linear(vwap_delta, decay_window1)
        
        # 3. rank(vwap_decayed)
        first_part = self.rank(vwap_decayed)
        
        # 두 번째 부분: 업종별 표준화된 저가-거래대금 상관관계의 복잡한 처리
        # 4. grouped_demean(low, industry_group_lv1) -> low - mean(low)
        low_demeaned = data['low'] - data['low'].mean()
        
        # 5. ts_mean(amount, 81)
        amount_mean = self.ts_mean(data['amount'], 81)
        
        # 6. ts_corr(low_demeaned, amount_mean, 8.14941) -> 반올림하여 8
        corr_window = int(round(8.14941))
        low_corr = self.ts_corr(low_demeaned, amount_mean, corr_window)
        
        # 7. ts_rank(low_corr, 19.569) -> 반올림하여 20
        rank_window1 = int(round(19.569))
        corr_ranked = self.ts_rank(low_corr, rank_window1)
        
        # 8. ts_decayed_linear(corr_ranked, 17.1543) -> 반올림하여 17
        decay_window2 = int(round(17.1543))
        corr_decayed = self.ts_decayed_linear(corr_ranked, decay_window2)
        
        # 9. ts_rank(corr_decayed, 19.383) -> 반올림하여 19
        rank_window2 = int(round(19.383))
        second_part = self.ts_rank(corr_decayed, rank_window2)
        
        # 10. max(first_part, second_part)
        max_result = self.max(first_part, second_part)
        
        # 11. mul(max_result, -1)
        alpha = self.mul(max_result, -1)
        
        return alpha.fillna(0)
