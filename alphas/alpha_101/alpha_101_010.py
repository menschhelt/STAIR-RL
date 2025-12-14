import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

# 부모 디렉토리를 import 경로에 추가
from alphas.base.base import BaseAlpha
class alpha_101_010(BaseAlpha):
    """
    Alpha 101_010: mul(add(rank(ts_max(sub({disk:vwap},{disk:close}),3)),rank(ts_min(sub({disk:vwap},{disk:close}),3))),rank(ts_delta({disk:volume},3)))
    
    VWAP-종가 편차의 극값과 거래량 변화 조합
    """
    neutralizer_type: str = "mean"  # 평균 중립화

    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_010"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_010 계산
        """
        # VWAP 계산 (없는 경우)
        if 'vwap' not in data:
            data['vwap'] = self.vwap_calc(
                data['high'], 
                data['low'], 
                data['close'], 
                data['volume']
            )
        
        # 1. vwap - close
        vwap_close_diff = self.sub(data['vwap'], data['close'])
        
        # 2. ts_max(vwap - close, 3)
        max_diff_3 = self.ts_max(vwap_close_diff, 3)

        # 3. ts_min(vwap - close, 3)
        min_diff_3 = self.ts_min(vwap_close_diff, 3)
        
        # 4. rank(ts_max(...))
        rank_max = self.rank(max_diff_3)
        
        # 5. rank(ts_min(...))
        rank_min = self.rank(min_diff_3)
        
        # 6. add(rank_max, rank_min)
        first_part = self.add(rank_max, rank_min)
        
        # 7. ts_delta(volume, 3)
        volume_delta_3 = self.ts_delta(data['volume'], 3)
        
        # 8. rank(ts_delta(volume, 3))
        second_part = self.rank(volume_delta_3)
        
        # 9. mul(first_part, second_part)
        alpha = self.mul(first_part, second_part)
        
        return alpha.fillna(0)
