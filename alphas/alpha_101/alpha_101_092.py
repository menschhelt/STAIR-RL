import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_092(BaseAlpha):
    """
    alpha101_092: div(ts_rank(ts_decayed_linear(ts_corr(grouped_demean({disk:vwap},{disk:industry_group_lv2}),ts_mean({disk:amount},81),17.4193),19.848),7.54455),rank(ts_decayed_linear(ts_delta(add(mul({disk:close},0.524434),mul({disk:vwap},sub(1,0.524434))),2.77377),16.2664)))
    
    업종별 표준화된 VWAP-거래대금 상관관계의 복잡한 처리를 가중 종가-VWAP 변화의 감쇠선형 랭킹으로 나눈 비율
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_092"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_092 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 분자: 업종별 표준화된 VWAP-거래대금 상관관계의 복잡한 처리
        # 1. grouped_demean(vwap, industry_group_lv2) -> vwap - mean(vwap)
        vwap_demeaned = data['vwap'] - data['vwap'].mean()
        
        # 2. ts_mean(amount, 81)
        amount_mean = self.ts_mean(data['amount'], 81)
        
        # 3. ts_corr(vwap_demeaned, amount_mean, 17.4193) -> 반올림하여 17
        corr_window = int(round(17.4193))
        vwap_amount_corr = self.ts_corr(vwap_demeaned, amount_mean, corr_window)
        
        # 4. ts_decayed_linear(vwap_amount_corr, 19.848) -> 반올림하여 20
        decay_window1 = int(round(19.848))
        corr_decayed = self.ts_decayed_linear(vwap_amount_corr, decay_window1)
        
        # 5. ts_rank(corr_decayed, 7.54455) -> 반올림하여 8
        rank_window1 = int(round(7.54455))
        numerator = self.ts_rank(corr_decayed, rank_window1)
        
        # 분모: 가중 종가-VWAP 변화의 감쇠선형 랭킹
        weight = 0.524434
        
        # 6. 가중 종가-VWAP: close * 0.524434 + vwap * (1-0.524434)
        weighted_close_vwap = self.add(
            self.mul(data['close'], weight),
            self.mul(data['vwap'], 1 - weight)
        )
        
        # 7. ts_delta(weighted_close_vwap, 2.77377) -> 반올림하여 3
        delta_window = int(round(2.77377))
        weighted_delta = self.ts_delta(weighted_close_vwap, delta_window)
        
        # 8. ts_decayed_linear(weighted_delta, 16.2664) -> 반올림하여 16
        decay_window2 = int(round(16.2664))
        delta_decayed = self.ts_decayed_linear(weighted_delta, decay_window2)
        
        # 9. rank(delta_decayed)
        denominator = self.rank(delta_decayed)
        
        # 10. div(numerator, denominator)
        alpha = self.div(numerator, denominator)
        
        return alpha.fillna(0)
