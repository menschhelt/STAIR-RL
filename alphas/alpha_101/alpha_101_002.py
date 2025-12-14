import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

# 부모 디렉토리를 import 경로에 추가
from alphas.base.base import BaseAlpha

class alpha_101_002(BaseAlpha):
    """
    Alpha 101_002: mul(-1,ts_corr(rank({disk:open}),rank({disk:volume}),10))
    
    시가와 거래량의 랭킹 간 음의 상관관계 활용
    """
    
    # 알파별 처리 설정 - 윈저라이징과 랭킹 적용
    neutralizer_type: str = "mean"  # 평균 중립화
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_002"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_002 계산
        """
        # 1. open 가격의 랭킹
        open_rank = self.rank(data['open'])
        
        # 2. volume의 랭킹
        volume_rank = self.rank(data['volume'])
        
        # 3. 10기간 상관계수 * -1 (DSL: ts_corr(..., 10))
        corr = self.ts_corr(open_rank, volume_rank, 10)
        alpha = self.mul(corr, -1)
        
        return alpha.fillna(0)
