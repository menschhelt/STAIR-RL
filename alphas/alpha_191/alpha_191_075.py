import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_075(BaseAlpha):
    """
    alpha191_075: div(ts_std(div(abs(div({disk:close},sub(delay({disk:close},1),1))),{disk:volume}),20),ts_mean(div(abs(div({disk:close},sub(delay({disk:close},1),1))),{disk:volume}),20))
    
    거래량 대비 가격 변화율의 변동계수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_075"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_075 계산
        """
        close = data['close']
        volume = data['volume']
        close_lag1 = self.delay(close, 1)
        
        # close / (delay(close, 1) - 1)의 절대값
        # 수정: close / delay(close, 1)로 해석
        price_ratio = self.abs(self.div(close, close_lag1))
        
        # 거래량으로 나눈 비율
        volume_adj_ratio = self.div(price_ratio, volume)
        
        # 20일 표준편차와 평균
        ratio_std = self.ts_std(volume_adj_ratio, 20)
        ratio_mean = self.ts_mean(volume_adj_ratio, 20)
        
        # 변동계수
        alpha = self.div(ratio_std, ratio_mean)
        
        return alpha.fillna(0)
