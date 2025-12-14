import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_021(BaseAlpha):
    """
    alpha191_021: EMA(div(sub({disk:close},ts_mean({disk:close},6)),sub(ts_mean({disk:close},6),delay(div(sub({disk:close},ts_mean({disk:close},6)),ts_mean({disk:close},6)),3))),12,div(1,12))
    
    복잡한 EMA 계산 - 가격 편차의 변화율
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_021"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_021 계산
        """
        close = data['close']
        
        # 6일 평균
        close_mean_6 = self.ts_mean(close, 6)
        
        # (close - 6일 평균) / 6일 평균
        deviation_ratio = self.div(
            self.sub(close, close_mean_6),
            close_mean_6
        )
        
        # 3일 지연된 deviation_ratio
        lag_deviation_ratio = self.delay(deviation_ratio, 3)
        
        # 분모: 6일 평균 - lag_deviation_ratio
        denominator = self.sub(close_mean_6, lag_deviation_ratio)
        
        # 최종 비율
        ratio = self.div(
            self.sub(close, close_mean_6),
            denominator
        )
        
        # EMA(12, 1/12)
        alpha = self.ema(ratio, 12, self.div(1, 12))
        
        return alpha.fillna(0)
