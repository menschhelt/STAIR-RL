import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_039(BaseAlpha):
    """
    alpha101_039: mul(mul(-1,rank(ts_std({disk:high},10))),ts_corr({disk:high},{disk:volume},10))
    
    고가의 10기간 변동성 랭킹과 고가-거래량 상관관계의 음의 복합 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_039"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_039 계산
        """
        # 1. ts_std(high, 10)
        high_std = self.ts_std(data['high'], 10)
        
        # 2. rank(high_std)
        std_rank = self.rank(high_std)
        
        # 3. mul(-1, std_rank)
        neg_rank = self.mul(std_rank, -1)
        
        # 4. ts_corr(high, volume, 10)
        corr_10 = self.ts_corr(data['high'], data['volume'], 10)
        
        # 5. mul(neg_rank, corr_10)
        alpha = self.mul(neg_rank, corr_10)
        
        return alpha.fillna(0)
