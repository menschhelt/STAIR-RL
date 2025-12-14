import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_026(BaseAlpha):
    """
    alpha101_026: condition(lt(0.5,rank(div(ts_sum(ts_corr(rank({disk:volume}),rank({disk:vwap}),6),2),2.0))),mul(-1,1),1)
    
    거래량과 VWAP 랭킹 간 상관관계의 2기간 합계 평균 랭킹에 따른 조건부 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_026"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_026 계산
        """
        # 1. rank(volume)
        volume_rank = self.rank(data['volume'])
        
        # 2. rank(vwap)
        vwap_rank = self.rank(data['vwap'])
        
        # 3. ts_corr(..., ..., 6)
        corr_6 = self.ts_corr(volume_rank, vwap_rank, 6)
        
        # 4. ts_sum(..., 2)
        sum_2 = self.ts_sum(corr_6, 2)
        
        # 5. div(..., 2.0)
        div_2 = self.div(sum_2, 2.0)
        
        # 6. rank(...)
        rank_result = self.rank(div_2)
        
        # 7. lt(0.5, rank_result)
        condition = self.lt(0.5, rank_result)
        
        # 8. condition(..., -1, 1)
        alpha = self.condition(condition, -1, 1)
        
        return alpha.fillna(0)
