import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_011(BaseAlpha):
    """
    alpha191_011: mul(rank(sub({disk:open},div(ts_sum({disk:vwap},10),10))),mul(-1,rank(abs(sub({disk:close},{disk:vwap})))))
    
    시가와 VWAP 평균 차이 랭킹 * (-1) * 종가와 VWAP 절대 차이 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_011"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_011 계산
        """
        open_price = data['open']
        close = data['close']
        vwap = data['vwap']
        
        # 10일 VWAP 평균
        vwap_mean = self.div(self.ts_sum(vwap, 10), 10)
        
        # 시가 - VWAP 평균
        open_vwap_diff = self.sub(open_price, vwap_mean)
        rank1 = self.rank(open_vwap_diff)
        
        # |종가 - VWAP|
        close_vwap_abs_diff = self.abs(self.sub(close, vwap))
        rank2 = self.rank(close_vwap_abs_diff)
        
        # 최종 계산
        alpha = self.mul(rank1, self.mul(-1, rank2))
        
        return alpha.fillna(0)
