import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_037(BaseAlpha):
    """
    alpha101_037: mul(mul(-1,rank(ts_rank({disk:close},10))),rank(div({disk:close},{disk:open})))
    
    종가의 10기간 시계열 랭킹과 종가/시가 비율 랭킹의 음의 복합 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_037"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_037 계산
        """
        # 1. ts_rank(close, 10)
        close_tsrank = self.ts_rank(data['close'], 10)
        
        # 2. rank(close_tsrank)
        close_rank = self.rank(close_tsrank)
        
        # 3. mul(-1, close_rank)
        neg_close_rank = self.mul(close_rank, -1)
        
        # 4. div(close, open)
        close_open_ratio = self.div(data['close'], data['open'])
        
        # 5. rank(close_open_ratio)
        ratio_rank = self.rank(close_open_ratio)
        
        # 6. mul(neg_close_rank, ratio_rank)
        alpha = self.mul(neg_close_rank, ratio_rank)
        
        return alpha.fillna(0)
