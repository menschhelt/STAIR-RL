import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_147(BaseAlpha):
    """
    alpha191_147: mul(lt(rank(ts_corr({disk:open},ts_sum(ts_mean({disk:volume},60),9),6)),rank(sub({disk:open},ts_min({disk:open},14)))),-1)
    
    -1 * 시가-거래량 상관관계와 시가 위치의 비교
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_147"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_147 계산
        """
        open_price = data['open']
        volume = data['volume']
        
        # 첫 번째 부분: rank(ts_corr(open, ts_sum(ts_mean(volume, 60), 9), 6))
        volume_mean = self.ts_mean(volume, 60)
        volume_sum = self.ts_sum(volume_mean, 9)
        open_volume_corr = self.ts_corr(open_price, volume_sum, 6)
        first_rank = self.rank(open_volume_corr)
        
        # 두 번째 부분: rank(open - ts_min(open, 14))
        open_min = self.ts_min(open_price, 14)
        open_position = self.sub(open_price, open_min)
        second_rank = self.rank(open_position)
        
        # 비교 후 -1 곱하기
        comparison = self.lt(first_rank, second_rank)
        alpha = self.mul(comparison, -1)
        
        return alpha.fillna(0)
