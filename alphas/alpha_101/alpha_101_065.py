import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_065(BaseAlpha):
    """
    alpha101_065: mul(add(rank(ts_decayed_linear(ts_delta({disk:vwap},3.51013),7.23052)),ts_rank(ts_decayed_linear(div(sub(add(mul({disk:low},0.96633),mul({disk:low},sub(1,0.96633))),{disk:vwap}),sub({disk:open},div(add({disk:high},{disk:low}),2))),11.4157),6.72611)),-1)
    
    VWAP 변화의 감쇠선형 랭킹과 가중 저가-VWAP 비율의 복잡한 시계열 랭킹의 합의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_065"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_065 계산
        """
        # 첫 번째 부분: VWAP 변화의 감쇠선형 랭킹
        # 1. ts_delta(vwap, 3.51013) -> 반올림하여 4
        delta_window = int(round(3.51013))
        vwap_delta = self.ts_delta(data['vwap'], delta_window)
        
        # 2. ts_decayed_linear(vwap_delta, 7.23052) -> 반올림하여 7
        decay_window1 = int(round(7.23052))
        vwap_decayed = self.ts_decayed_linear(vwap_delta, decay_window1)
        
        # 3. rank(vwap_decayed)
        first_part = self.rank(vwap_decayed)
        
        # 두 번째 부분: 복잡한 가격 비율
        weight = 0.96633
        
        # 4. 가중 저가: low * 0.96633 + low * (1-0.96633) = low (실제로는 저가와 동일)
        weighted_low = data['low']  # 단순화
        
        # 5. sub(weighted_low, vwap)
        low_vwap_diff = self.sub(weighted_low, data['vwap'])
        
        # 6. div(add(high, low), 2) - 중간가격
        mid_price = self.div(
            self.add(data['high'], data['low']),
            2
        )
        
        # 7. sub(open, mid_price)
        open_mid_diff = self.sub(data['open'], mid_price)
        
        # 8. div(low_vwap_diff, open_mid_diff)
        ratio = self.div(low_vwap_diff, open_mid_diff)
        
        # 9. ts_decayed_linear(ratio, 11.4157) -> 반올림하여 11
        decay_window2 = int(round(11.4157))
        ratio_decayed = self.ts_decayed_linear(ratio, decay_window2)
        
        # 10. ts_rank(ratio_decayed, 6.72611) -> 반올림하여 7
        rank_window = int(round(6.72611))
        second_part = self.ts_rank(ratio_decayed, rank_window)
        
        # 11. add(first_part, second_part)
        sum_parts = self.add(first_part, second_part)
        
        # 12. mul(sum_parts, -1)
        alpha = self.mul(sum_parts, -1)
        
        return alpha.fillna(0)
