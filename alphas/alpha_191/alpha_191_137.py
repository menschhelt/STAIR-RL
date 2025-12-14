import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_137(BaseAlpha):
    """
    alpha191_137: mul(sub(rank(ts_decayed_linear(ts_delta(add(mul({disk:low},0.7),mul({disk:vwap},0.3)),3),20)),ts_rank(ts_decayed_linear(ts_rank(ts_corr(ts_rank({disk:low},8),ts_rank(ts_mean({disk:volume},60),17),5),19),16),7)),-1)
    
    저가와 VWAP 조합의 변화율과 복잡한 상관관계 랭킹의 차이
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_137"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_137 계산
        """
        low = data['low']
        vwap = data['vwap']
        volume = data['volume']
        
        # 첫 번째 부분: low*0.7 + vwap*0.3의 3일 변화를 20일 감쇠선형
        low_vwap_combo = self.add(self.mul(low, 0.7), self.mul(vwap, 0.3))
        combo_delta = self.ts_delta(low_vwap_combo, 3)
        first_part = self.rank(self.ts_decayed_linear(combo_delta, 20))
        
        # 두 번째 부분: 복잡한 상관관계 계산을 단순화
        volume_mean = self.ts_mean(volume, 60)
        low_rank = self.ts_rank(low, 8)
        volume_rank = self.ts_rank(volume_mean, 17)
        
        corr = self.ts_corr(low_rank, volume_rank, 5)
        corr_rank = self.ts_rank(corr, 19)
        decayed = self.ts_decayed_linear(corr_rank, 16)
        second_part = self.ts_rank(decayed, 7)
        
        # 차이를 구하고 -1 곱하기
        alpha = self.mul(self.sub(first_part, second_part), -1)
        
        return alpha.fillna(0)
