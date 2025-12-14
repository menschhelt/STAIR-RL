import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_005(BaseAlpha):
    """
    alpha191_005: mul(rank(sign(ts_delta(add(mul({disk:open},0.85),mul({disk:high},0.15)),4))),-1)
    
    -1 * 가중 평균 가격 변화의 부호에 대한 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_005"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_005 계산
        """
        # 가중 평균: 시가 * 0.85 + 고가 * 0.15
        weighted_price = self.add(
            self.mul(data['open'], 0.85),
            self.mul(data['high'], 0.15)
        )
        
        # 4일 변화
        delta = self.ts_delta(weighted_price, 4)
        
        # 부호
        sign_delta = self.sign(delta)
        
        # 랭킹
        rank_sign = self.rank(sign_delta)
        
        # -1 곱하기
        alpha = self.mul(rank_sign, -1)
        
        return alpha.fillna(0)
