import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_058(BaseAlpha):
    """
    alpha101_058: mul(-1,ts_rank(ts_decayed_linear(ts_corr(grouped_demean(add(mul({disk:vwap},0.728317),mul({disk:vwap},sub(1,0.728317))),{disk:industry_group_lv2}),{disk:volume},4.25197),16.2289),8.19648))
    
    가중 VWAP의 업종별 표준화와 거래량의 상관관계에 대한 복잡한 감쇠선형 및 시계열 랭킹의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_058"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_058 계산
        """
        # 1. 가중 VWAP 계산: vwap * 0.728317 + vwap * (1-0.728317) = vwap
        # 이 계산은 실제로는 vwap과 동일
        weighted_vwap = data['vwap']  # 단순화
        
        # 2. industry_group_lv2가 없으므로 전체 평균으로 대체
        vwap_demeaned = weighted_vwap - weighted_vwap.mean()
        
        # 3. ts_corr(vwap_demeaned, volume, 4.25197)
        corr_window = int(round(4.25197))  # 4
        corr_result = self.ts_corr(vwap_demeaned, data['volume'], corr_window)
        
        # 4. ts_decayed_linear(corr_result, 16.2289)
        decay_window = int(round(16.2289))  # 16
        decayed = self.ts_decayed_linear(corr_result, decay_window)
        
        # 5. ts_rank(decayed, 8.19648)
        rank_window = int(round(8.19648))  # 8
        ranked = self.ts_rank(decayed, rank_window)
        
        # 6. mul(-1, ranked)
        alpha = self.mul(ranked, -1)
        
        return alpha.fillna(0)
