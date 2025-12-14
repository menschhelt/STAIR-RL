import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_086(BaseAlpha):
    """
    alpha101_086: mul(max(rank(ts_decayed_linear(ts_delta(add(mul({disk:close},0.369701),mul({disk:vwap},sub(1,0.369701))),1.91233),2.65461)),ts_rank(ts_decayed_linear(abs(ts_corr(grouped_demean(ts_mean({disk:amount},81),{disk:industry_group_lv2}),{disk:close},13.4132)),4.89768),14.4535)),-1)
    
    가중 종가-VWAP 변화의 감쇠선형 랭킹과 업종별 표준화된 거래대금-종가 상관관계 절댓값의 복잡한 처리의 최대값의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_086"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_086 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        weight = 0.369701
        
        # 첫 번째 부분: 가중 종가-VWAP 변화의 감쇠선형 랭킹
        # 1. 가중 종가-VWAP: close * 0.369701 + vwap * (1-0.369701)
        weighted_close_vwap = self.add(
            self.mul(data['close'], weight),
            self.mul(data['vwap'], 1 - weight)
        )
        
        # 2. ts_delta(weighted_close_vwap, 1.91233) -> 반올림하여 2
        delta_window = int(round(1.91233))
        weighted_delta = self.ts_delta(weighted_close_vwap, delta_window)
        
        # 3. ts_decayed_linear(weighted_delta, 2.65461) -> 반올림하여 3
        decay_window1 = int(round(2.65461))
        delta_decayed = self.ts_decayed_linear(weighted_delta, decay_window1)
        
        # 4. rank(delta_decayed)
        first_part = self.rank(delta_decayed)
        
        # 두 번째 부분: 업종별 표준화된 거래대금-종가 상관관계 절댓값의 복잡한 처리
        # 5. ts_mean(amount, 81)
        amount_mean = self.ts_mean(data['amount'], 81)
        
        # 6. grouped_demean(amount_mean, industry_group_lv2) -> amount_mean - mean(amount_mean)
        amount_demeaned = amount_mean - amount_mean.mean()
        
        # 7. ts_corr(amount_demeaned, close, 13.4132) -> 반올림하여 13
        corr_window = int(round(13.4132))
        amount_corr = self.ts_corr(amount_demeaned, data['close'], corr_window)
        
        # 8. abs(amount_corr)
        abs_corr = self.abs(amount_corr)
        
        # 9. ts_decayed_linear(abs_corr, 4.89768) -> 반올림하여 5
        decay_window2 = int(round(4.89768))
        corr_decayed = self.ts_decayed_linear(abs_corr, decay_window2)
        
        # 10. ts_rank(corr_decayed, 14.4535) -> 반올림하여 14
        rank_window = int(round(14.4535))
        second_part = self.ts_rank(corr_decayed, rank_window)
        
        # 11. max(first_part, second_part)
        max_result = self.max(first_part, second_part)
        
        # 12. mul(max_result, -1)
        alpha = self.mul(max_result, -1)
        
        return alpha.fillna(0)
