import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_141(BaseAlpha):
    """
    alpha191_141: mul(mul(mul(-1,rank(ts_rank({disk:close},10))),rank(ts_delta(ts_delta({disk:close},1),1))),rank(ts_rank(div({disk:volume},ts_mean({disk:volume},20)),5)))
    
    -1 * 종가 랭킹, 종가 2차 변화 랭킹, 상대 거래량 랭킹의 곱
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_141"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_141 계산
        """
        close = data['close']
        volume = data['volume']
        
        # 첫 번째 부분: -1 * rank(ts_rank(close, 10))
        close_ts_rank = self.ts_rank(close, 10)
        first_part = self.mul(-1, self.rank(close_ts_rank))
        
        # 두 번째 부분: rank(ts_delta(ts_delta(close, 1), 1)) - 종가의 2차 변화
        close_delta1 = self.ts_delta(close, 1)
        close_delta2 = self.ts_delta(close_delta1, 1)
        second_part = self.rank(close_delta2)
        
        # 세 번째 부분: rank(ts_rank(volume/ts_mean(volume, 20), 5))
        volume_mean = self.ts_mean(volume, 20)
        relative_volume = self.div(volume, volume_mean)
        volume_ts_rank = self.ts_rank(relative_volume, 5)
        third_part = self.rank(volume_ts_rank)
        
        # 세 부분의 곱
        alpha = self.mul(self.mul(first_part, second_part), third_part)
        
        return alpha.fillna(0)
