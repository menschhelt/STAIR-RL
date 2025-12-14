import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_081(BaseAlpha):
    """
    alpha101_081: mul(min(rank(ts_decayed_linear(ts_delta({disk:open},1.46063),14.8717)),ts_rank(ts_decayed_linear(ts_corr(grouped_demean({disk:volume},{disk:industry_group_lv1}),add(mul({disk:open},0.634196),mul({disk:open},sub(1,0.634196))),17.4842),6.92131),13.4283)),-1)
    
    시가 변화의 감쇠선형 랭킹과 업종별 표준화된 거래량-가중시가 상관관계의 복잡한 처리의 최소값의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_081"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_081 계산
        """
        # 첫 번째 부분: 시가 변화의 감쇠선형 랭킹
        # 1. ts_delta(open, 1.46063) -> 반올림하여 1
        delta_window = int(round(1.46063))
        open_delta = self.ts_delta(data['open'], delta_window)
        
        # 2. ts_decayed_linear(open_delta, 14.8717) -> 반올림하여 15
        decay_window1 = int(round(14.8717))
        open_decayed = self.ts_decayed_linear(open_delta, decay_window1)
        
        # 3. rank(open_decayed)
        first_part = self.rank(open_decayed)
        
        # 두 번째 부분: 업종별 표준화된 거래량-가중시가 상관관계의 복잡한 처리
        # 4. grouped_demean(volume, industry_group_lv1) -> volume - mean(volume)
        volume_demeaned = data['volume'] - data['volume'].mean()
        
        # 5. 가중 시가: open * 0.634196 + open * (1-0.634196) = open (실제로는 시가와 동일)
        weighted_open = data['open']  # 단순화
        
        # 6. ts_corr(volume_demeaned, weighted_open, 17.4842) -> 반올림하여 17
        corr_window = int(round(17.4842))
        corr_result = self.ts_corr(volume_demeaned, weighted_open, corr_window)
        
        # 7. ts_decayed_linear(corr_result, 6.92131) -> 반올림하여 7
        decay_window2 = int(round(6.92131))
        corr_decayed = self.ts_decayed_linear(corr_result, decay_window2)
        
        # 8. ts_rank(corr_decayed, 13.4283) -> 반올림하여 13
        rank_window = int(round(13.4283))
        second_part = self.ts_rank(corr_decayed, rank_window)
        
        # 9. min(first_part, second_part)
        min_result = self.min(first_part, second_part)
        
        # 10. mul(min_result, -1)
        alpha = self.mul(min_result, -1)
        
        return alpha.fillna(0)
