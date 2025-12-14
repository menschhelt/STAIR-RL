import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_057(BaseAlpha):
    """
    alpha101_057: mul(-1,ts_rank(ts_decayed_linear(ts_corr(grouped_demean({disk:vwap},{disk:industry_group_lv1}),{disk:volume},3.92795),7.89291),5.50322))
    
    업종별 표준화된 VWAP과 거래량의 상관관계에 대한 복잡한 감쇠선형 및 시계열 랭킹의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_057"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_057 계산
        """
        # industry_group_lv1이 없으므로 전체 평균으로 대체
        # 1. grouped_demean(vwap, industry_group_lv1) -> vwap - mean(vwap)
        vwap_demeaned = data['vwap'] - data['vwap'].mean()
        
        # 2. ts_corr(vwap_demeaned, volume, 3.92795)
        # 소수점 윈도우는 정수로 반올림
        corr_window = int(round(3.92795))  # 4
        corr_result = self.ts_corr(vwap_demeaned, data['volume'], corr_window)
        
        # 3. ts_decayed_linear(corr_result, 7.89291)
        decay_window = int(round(7.89291))  # 8
        decayed = self.ts_decayed_linear(corr_result, decay_window)
        
        # 4. ts_rank(decayed, 5.50322)
        rank_window = int(round(5.50322))  # 6
        ranked = self.ts_rank(decayed, rank_window)
        
        # 5. mul(-1, ranked)
        alpha = self.mul(ranked, -1)
        
        return alpha.fillna(0)
