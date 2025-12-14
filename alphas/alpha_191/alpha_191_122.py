import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_122(BaseAlpha):
    """
    alpha191_122: mul(lt(rank(ts_corr(ts_sum(div(add({disk:high},{disk:low}),2),20),ts_sum(ts_mean({disk:volume},60),20),9)),rank(ts_corr({disk:low},{disk:volume},6))),-1)
    
    -1 * (복잡한 가격-거래량 상관관계 랭킹 < 저가-거래량 상관관계 랭킹)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_122"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_122 계산
        """
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # 첫 번째 부분: HL 평균과 복잡한 거래량 지표의 상관관계
        hl_avg = self.div(self.add(high, low), 2)
        hl_sum = self.ts_sum(hl_avg, 20)
        
        volume_mean_60 = self.ts_mean(volume, 60)
        volume_sum = self.ts_sum(volume_mean_60, 20)
        
        corr1 = self.ts_corr(hl_sum, volume_sum, 9)
        rank1 = self.rank(corr1)
        
        # 두 번째 부분: 저가와 거래량의 6일 상관관계
        corr2 = self.ts_corr(low, volume, 6)
        rank2 = self.rank(corr2)
        
        # 비교 후 -1 곱하기
        comparison = self.lt(rank1, rank2)
        alpha = self.mul(comparison, -1)
        
        return alpha.fillna(0).astype(float)
