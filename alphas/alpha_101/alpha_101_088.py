import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_088(BaseAlpha):
    """
    alpha101_088: sub(ts_rank(ts_decayed_linear(ts_corr(add(mul({disk:low},0.967285),mul({disk:low},sub(1,0.967285))),ts_mean({disk:amount},10),6.94279),5.51607),3.79744),ts_rank(ts_decayed_linear(ts_delta(grouped_demean({disk:vwap},{disk:industry_group_lv2}),3.48158),10.1466),15.3012))
    
    가중 저가-거래대금 상관관계의 복잡한 처리에서 업종별 표준화된 VWAP 변화의 복잡한 처리를 뺀 값
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_088"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_088 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 가중 저가-거래대금 상관관계의 복잡한 처리
        # 1. 가중 저가: low * 0.967285 + low * (1-0.967285) = low (실제로는 저가와 동일)
        weighted_low = data['low']  # 단순화
        
        # 2. ts_mean(amount, 10)
        amount_mean = self.ts_mean(data['amount'], 10)
        
        # 3. ts_corr(weighted_low, amount_mean, 6.94279) -> 반올림하여 7
        corr_window1 = int(round(6.94279))
        low_corr = self.ts_corr(weighted_low, amount_mean, corr_window1)
        
        # 4. ts_decayed_linear(low_corr, 5.51607) -> 반올림하여 6
        decay_window1 = int(round(5.51607))
        low_decayed = self.ts_decayed_linear(low_corr, decay_window1)
        
        # 5. ts_rank(low_decayed, 3.79744) -> 반올림하여 4
        rank_window1 = int(round(3.79744))
        first_part = self.ts_rank(low_decayed, rank_window1)
        
        # 두 번째 부분: 업종별 표준화된 VWAP 변화의 복잡한 처리
        # 6. grouped_demean(vwap, industry_group_lv2) -> vwap - mean(vwap)
        vwap_demeaned = data['vwap'] - data['vwap'].mean()
        
        # 7. ts_delta(vwap_demeaned, 3.48158) -> 반올림하여 3
        delta_window = int(round(3.48158))
        vwap_delta = self.ts_delta(vwap_demeaned, delta_window)
        
        # 8. ts_decayed_linear(vwap_delta, 10.1466) -> 반올림하여 10
        decay_window2 = int(round(10.1466))
        vwap_decayed = self.ts_decayed_linear(vwap_delta, decay_window2)
        
        # 9. ts_rank(vwap_decayed, 15.3012) -> 반올림하여 15
        rank_window2 = int(round(15.3012))
        second_part = self.ts_rank(vwap_decayed, rank_window2)
        
        # 10. sub(first_part, second_part)
        alpha = self.sub(first_part, second_part)
        
        return alpha.fillna(0)
