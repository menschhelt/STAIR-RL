import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_076(BaseAlpha):
    """
    alpha191_076: min(rank(ts_decayed_linear(sub(add(div(add({disk:high},{disk:low}),2),{disk:high}),add({disk:vwap},{disk:high})),20)),rank(ts_decayed_linear(ts_corr(div(add({disk:high},{disk:low}),2),ts_mean({disk:volume},40),3),6)))
    
    두 개의 복잡한 가격 지표의 최소값
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_076"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_076 계산
        """
        high = data['high']
        low = data['low']
        vwap = data['vwap']
        volume = data['volume']
        
        # (high + low) / 2
        hl_avg = self.div(self.add(high, low), 2)
        
        # 첫 번째 부분: ((high + low) / 2 + high) - (vwap + high)
        part1 = self.sub(
            self.add(hl_avg, high),
            self.add(vwap, high)
        )
        decay1 = self.ts_decayed_linear(part1, 20)
        rank1 = self.rank(decay1)
        
        # 두 번째 부분: hl_avg와 40일 평균 거래량의 3일 상관관계
        volume_mean_40 = self.ts_mean(volume, 40)
        corr = self.ts_corr(hl_avg, volume_mean_40, 3)
        decay2 = self.ts_decayed_linear(corr, 6)
        rank2 = self.rank(decay2)
        
        # 최소값
        alpha = self.min(rank1, rank2)
        
        return alpha.fillna(0)
