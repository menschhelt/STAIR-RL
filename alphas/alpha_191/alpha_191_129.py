import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_129(BaseAlpha):
    """
    alpha191_129: div(rank(ts_decayed_linear(ts_corr(div(add({disk:high},{disk:low}),2),ts_mean({disk:volume},40),9),10)),rank(ts_decayed_linear(ts_corr(rank({disk:vwap}),rank({disk:volume}),7),3)))
    
    중간가격-거래량 상관관계와 VWAP-거래량 상관관계의 비교
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_129"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_129 계산
        """
        high = data['high']
        low = data['low']
        vwap = data['vwap']
        volume = data['volume']
        
        # 중간가격: (high + low) / 2
        mid_price = self.div(self.add(high, low), 2)
        
        # 40일 거래량 평균
        volume_mean_40 = self.ts_mean(volume, 40)
        
        # 첫 번째 상관관계: 중간가격과 거래량 평균
        corr1 = self.ts_corr(mid_price, volume_mean_40, 9)
        decayed1 = self.ts_decayed_linear(corr1, 10)
        rank1 = self.rank(decayed1)
        
        # 두 번째 상관관계: VWAP 랭킹과 거래량 랭킹
        vwap_rank = self.rank(vwap)
        volume_rank = self.rank(volume)
        corr2 = self.ts_corr(vwap_rank, volume_rank, 7)
        decayed2 = self.ts_decayed_linear(corr2, 3)
        rank2 = self.rank(decayed2)
        
        # 비율
        alpha = self.div(rank1, rank2)
        
        return alpha.fillna(0)
