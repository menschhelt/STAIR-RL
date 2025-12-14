import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_107(BaseAlpha):
    """
    alpha191_107: mul(pow(rank(sub({disk:high},min({disk:high},2))),rank(ts_corr({disk:vwap},ts_mean({disk:volume},120),6))),-1)
    
    -1 * (고가 상대적 위치 랭킹^VWAP-거래량 상관관계 랭킹)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_107"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_107 계산
        """
        high = data['high']
        vwap = data['vwap']
        volume = data['volume']
        
        # 고가 - 2일 최소 고가 (항상 0 이상)
        high_diff = self.sub(high, self.ts_min(high, 2))
        base_rank = self.rank(high_diff)
        
        # VWAP와 120일 평균 거래량의 6일 상관관계
        volume_mean_120 = self.ts_mean(volume, 120)
        corr = self.ts_corr(vwap, volume_mean_120, 6)
        exp_rank = self.rank(corr)
        
        # 거듭제곱
        power_result = self.pow(base_rank, exp_rank)
        
        # -1 곱하기
        alpha = self.mul(power_result, -1)
        
        return alpha.fillna(0)
