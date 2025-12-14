import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_072(BaseAlpha):
    """
    alpha101_072: mul(max(rank(ts_decayed_linear(ts_delta({disk:vwap},4.72775),2.91864)),ts_rank(ts_decayed_linear(mul(div(ts_delta(add(mul({disk:open},0.147155),mul({disk:low},sub(1,0.147155))),2.03608),add(mul({disk:open},0.147155),mul({disk:low},sub(1,0.147155)))),-1),3.33829),16.7411)),-1)
    
    VWAP 변화의 감쇠선형 랭킹과 가중 시가-저가 변화율의 복잡한 처리의 최대값의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_072"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_072 계산
        """
        weight = 0.147155
        
        # 첫 번째 부분: VWAP 변화의 감쇠선형 랭킹
        # 1. ts_delta(vwap, 4.72775) -> 반올림하여 5
        delta_window1 = int(round(4.72775))
        vwap_delta = self.ts_delta(data['vwap'], delta_window1)
        
        # 2. ts_decayed_linear(vwap_delta, 2.91864) -> 반올림하여 3
        decay_window1 = int(round(2.91864))
        vwap_decayed = self.ts_decayed_linear(vwap_delta, decay_window1)
        
        # 3. rank(vwap_decayed)
        first_part = self.rank(vwap_decayed)
        
        # 두 번째 부분: 가중 시가-저가 변화율의 복잡한 처리
        # 4. 가중 시가-저가: open * 0.147155 + low * (1-0.147155)
        weighted_open_low = self.add(
            self.mul(data['open'], weight),
            self.mul(data['low'], 1 - weight)
        )
        
        # 5. ts_delta(weighted_open_low, 2.03608) -> 반올림하여 2
        delta_window2 = int(round(2.03608))
        weighted_delta = self.ts_delta(weighted_open_low, delta_window2)
        
        # 6. div(weighted_delta, weighted_open_low) - 변화율
        change_rate = self.div(weighted_delta, weighted_open_low)
        
        # 7. mul(change_rate, -1)
        neg_change_rate = self.mul(change_rate, -1)
        
        # 8. ts_decayed_linear(neg_change_rate, 3.33829) -> 반올림하여 3
        decay_window2 = int(round(3.33829))
        rate_decayed = self.ts_decayed_linear(neg_change_rate, decay_window2)
        
        # 9. ts_rank(rate_decayed, 16.7411) -> 반올림하여 17
        rank_window = int(round(16.7411))
        second_part = self.ts_rank(rate_decayed, rank_window)
        
        # 10. max(first_part, second_part)
        max_result = self.max(first_part, second_part)
        
        # 11. mul(max_result, -1)
        alpha = self.mul(max_result, -1)
        
        return alpha.fillna(0)
