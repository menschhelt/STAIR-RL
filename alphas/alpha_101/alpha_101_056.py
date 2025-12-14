import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_056(BaseAlpha):
    """
    alpha101_056: sub(0,mul(1,div(sub({disk:close},{disk:vwap}),ts_decayed_linear(rank(ts_argmax({disk:close},30)),2))))
    
    종가-VWAP 차이를 30기간 종가 최대값 위치의 감쇠선형 랭킹으로 나눈 값의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_056"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_056 계산
        """
        # 분자: close - vwap
        # 1. sub(close, vwap)
        close_vwap_diff = self.sub(data['close'], data['vwap'])
        
        # 분모: 복잡한 감쇠선형 랭킹
        # 2. ts_argmax(close, 30)
        argmax_30 = self.ts_argmax(data['close'], 30)
        
        # 3. rank(argmax_30)
        argmax_rank = self.rank(argmax_30)
        
        # 4. ts_decayed_linear(argmax_rank, 2)
        decayed_rank = self.ts_decayed_linear(argmax_rank, 2)
        
        # 5. div(close_vwap_diff, decayed_rank)
        ratio = self.div(close_vwap_diff, decayed_rank)
        
        # 6. sub(0, ratio) = -ratio
        alpha = self.mul(ratio, -1)
        
        return alpha.fillna(0)
