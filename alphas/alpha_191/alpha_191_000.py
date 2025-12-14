import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_000(BaseAlpha):
    """
    alpha191_000: mul(-1,ts_corr(rank(ts_delta(log({disk:volume}),1)),rank(div(sub({disk:close},{disk:open}),{disk:open})),6))
    
    -1 * 거래량 변화와 일일 수익률의 6일 상관관계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_000"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_000 계산
        """
        # log(volume)의 1일 변화를 랭킹
        log_vol = self.log(data['volume'])
        vol_delta = self.ts_delta(log_vol, 1)
        vol_rank = self.rank(vol_delta)
        
        # 일일 수익률을 랭킹
        returns = self.div(self.sub(data['close'], data['open']), data['open'])
        returns_rank = self.rank(returns)
        
        # 6일 상관관계
        corr = self.ts_corr(vol_rank, returns_rank, 6)
        
        # -1 곱하기
        alpha = self.mul(-1, corr)
        
        return alpha.fillna(0)
