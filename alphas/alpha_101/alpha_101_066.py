import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_066(BaseAlpha):
    """
    alpha101_066: mul(pow(rank(sub({disk:high},ts_min({disk:high},2.14593))),rank(ts_corr(grouped_demean({disk:vwap},{disk:industry_group_lv1}),grouped_demean(ts_mean({disk:amount},20),{disk:industry_group_lv3}),6.02936))),-1)
    
    고가-최소고가 차이 랭킹의 거듭제곱과 업종별 표준화된 VWAP-거래대금 상관관계 랭킹의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_066"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_066 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 고가-최소고가 차이
        # 1. ts_min(high, 2.14593) -> 반올림하여 2
        min_window = int(round(2.14593))
        high_min = self.ts_min(data['high'], min_window)
        
        # 2. sub(high, high_min)
        high_diff = self.sub(data['high'], high_min)
        
        # 3. rank(high_diff)
        base_rank = self.rank(high_diff)
        
        # 두 번째 부분: 업종별 표준화된 상관관계 (업종 정보가 없으므로 전체 평균으로 대체)
        # 4. grouped_demean(vwap, industry_group_lv1) -> vwap - mean(vwap)
        vwap_demeaned = data['vwap'] - data['vwap'].mean()
        
        # 5. ts_mean(amount, 20)
        amount_mean = self.ts_mean(data['amount'], 20)
        
        # 6. grouped_demean(amount_mean, industry_group_lv3) -> amount_mean - mean(amount_mean)
        amount_demeaned = amount_mean - amount_mean.mean()
        
        # 7. ts_corr(vwap_demeaned, amount_demeaned, 6.02936) -> 반올림하여 6
        corr_window = int(round(6.02936))
        corr_result = self.ts_corr(vwap_demeaned, amount_demeaned, corr_window)
        
        # 8. rank(corr_result)
        power_rank = self.rank(corr_result)
        
        # 9. pow(base_rank, power_rank)
        powered = self.pow(base_rank, power_rank)
        
        # 10. mul(powered, -1)
        alpha = self.mul(powered, -1)
        
        return alpha.fillna(0)
