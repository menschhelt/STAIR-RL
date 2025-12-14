import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_055(BaseAlpha):
    """
    alpha191_055: lt(rank(sub({disk:open},ts_min({disk:open},12))),rank(pow(rank(ts_corr(ts_sum(div(add({disk:high},{disk:low}),2),19),ts_sum(ts_mean({disk:volume},40),19),13)),5)))
    
    시가 상대적 위치 < 복잡한 가격-거래량 상관관계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_055"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_055 계산
        """
        open_price = data['open']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # 첫 번째 부분: open - ts_min(open, 12)
        open_min_12 = self.ts_min(open_price, 12)
        open_diff = self.sub(open_price, open_min_12)
        rank1 = self.rank(open_diff)
        
        # 두 번째 부분: 복잡한 가격-거래량 상관관계
        # (high + low) / 2의 19일 합계
        hl_avg = self.div(self.add(high, low), 2)
        hl_sum = self.ts_sum(hl_avg, 19)
        
        # ts_mean(volume, 40)의 19일 합계
        volume_mean_40 = self.ts_mean(volume, 40)
        volume_sum = self.ts_sum(volume_mean_40, 19)
        
        # 13일 상관관계
        corr = self.ts_corr(hl_sum, volume_sum, 13)
        corr_rank = self.rank(corr)
        
        # 5제곱
        corr_pow = self.pow(corr_rank, 5)
        rank2 = self.rank(corr_pow)
        
        # 비교
        alpha = self.lt(rank1, rank2)
        
        return alpha.fillna(0).astype(float)
