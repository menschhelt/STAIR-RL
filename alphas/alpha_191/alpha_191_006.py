import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_006(BaseAlpha):
    """
    alpha191_006: mul(add(rank(max(sub({disk:vwap},{disk:close}),3)),rank(min(sub({disk:vwap},{disk:close}),3))),rank(ts_delta({disk:volume},3)))
    
    VWAP-종가 차이의 최대/최소값 랭킹과 거래량 변화 랭킹의 곱
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_006"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_006 계산
        """
        # VWAP - close
        vwap_close_diff = self.sub(data['vwap'], data['close'])
        
        # 3일 최대값과 최소값
        max_diff = self.ts_max(vwap_close_diff, 3)
        min_diff = self.ts_min(vwap_close_diff, 3)
        
        # 랭킹
        rank_max = self.rank(max_diff)
        rank_min = self.rank(min_diff)
        
        # 거래량 3일 변화
        volume_delta = self.ts_delta(data['volume'], 3)
        rank_volume_delta = self.rank(volume_delta)
        
        # 최종 계산
        alpha = self.mul(
            self.add(rank_max, rank_min),
            rank_volume_delta
        )
        
        return alpha.fillna(0)
