import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_021(BaseAlpha):
    """
    alpha101_021: mul(-1,mul(ts_delta(ts_corr({disk:high},{disk:volume},5),5),rank(ts_std({disk:close},20))))
    
    고가-거래량 상관관계의 5기간 변화와 종가 변동성 랭킹의 복합 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_021"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_021 계산
        """
        # 1. ts_corr(high, volume, 5)
        corr_5 = self.ts_corr(data['high'], data['volume'], 5)
        
        # 2. ts_delta(..., 5)
        corr_delta = self.ts_delta(corr_5, 5)
        
        # 3. rank(ts_std(close, 20))
        close_std = self.ts_std(data['close'], 20)
        std_rank = self.rank(close_std)
        
        # 4. mul(delta, rank)
        product = self.mul(corr_delta, std_rank)
        
        # 5. mul(-1, ...)
        alpha = self.mul(product, -1)
        
        return alpha.fillna(0)
