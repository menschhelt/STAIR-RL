import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_007(BaseAlpha):
    """
    alpha191_007: rank(mul(ts_delta(add(mul(div(add({disk:high},{disk:low}),2),0.2),mul({disk:vwap},0.8)),4),-1))
    
    -1 * 가중 평균 가격 변화의 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_007"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_007 계산
        """
        # (high + low) / 2 * 0.2
        hl_avg = self.div(self.add(data['high'], data['low']), 2)
        weighted_hl = self.mul(hl_avg, 0.2)
        
        # vwap * 0.8
        weighted_vwap = self.mul(data['vwap'], 0.8)
        
        # 가중 평균
        weighted_price = self.add(weighted_hl, weighted_vwap)
        
        # 4일 변화
        delta = self.ts_delta(weighted_price, 4)
        
        # -1 곱하기
        neg_delta = self.mul(delta, -1)
        
        # 랭킹
        alpha = self.rank(neg_delta)
        
        return alpha.fillna(0)
