import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_031(BaseAlpha):
    """
    alpha101_031: add(twise_a_scale(sub(div(ts_sum({disk:close},7),7),{disk:close})),mul(20,twise_a_scale(ts_corr({disk:vwap},delay({disk:close},5),230))))
    
    7일 평균과 현재가의 차이 정규화와 VWAP-지연종가 상관관계의 복합 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_031"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_031 계산
        """
        # 첫 번째 부분: 7일 평균과 현재가 차이
        # 1. div(ts_sum(close, 7), 7) - 7일 평균
        close_mean_7 = self.div(self.ts_sum(data['close'], 7), 7)
        
        # 2. sub(close_mean_7, close)
        diff_7 = self.sub(close_mean_7, data['close'])
        
        # 3. twise_a_scale(diff_7)
        first_part = self.twise_a_scale(diff_7, 1)
        
        # 두 번째 부분: VWAP-지연종가 상관관계
        # 4. delay(close, 5)
        close_lag5 = self.delay(data['close'], 5)
        
        # 5. ts_corr(vwap, close_lag5, 230)
        corr_230 = self.ts_corr(data['vwap'], close_lag5, 230)
        
        # 6. twise_a_scale(corr_230)
        scaled_corr = self.twise_a_scale(corr_230, 1)
        
        # 7. mul(20, scaled_corr)
        second_part = self.mul(20, scaled_corr)
        
        # 8. add(first_part, second_part)
        alpha = self.add(first_part, second_part)
        
        return alpha.fillna(0)
