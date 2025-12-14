import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_077(BaseAlpha):
    """
    alpha191_077: div(div(add({disk:high},add({disk:low},{disk:close})),sub(3,ma(div(add({disk:high},add({disk:low},{disk:close})),3),12))),mul(0.015,ts_mean(abs(sub({disk:close},ts_mean(div(add({disk:high},add({disk:low},{disk:close})),3),12))),12)))
    
    HLC 평균 기반 복잡한 정규화 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_077"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_077 계산
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # HLC 합계
        hlc_sum = self.add(high, self.add(low, close))
        
        # HLC 평균
        hlc_avg = self.div(hlc_sum, 3)
        
        # 12일 이동평균
        hlc_ma = self.ts_mean(hlc_avg, 12)
        
        # 분자: hlc_sum / (3 - hlc_ma)
        numerator = self.div(hlc_sum, (3 - hlc_ma))
        
        # 분모: 0.015 * ts_mean(|close - hlc_ma|, 12)
        close_diff = self.abs(self.sub(close, hlc_ma))
        denominator = self.mul(0.015, self.ts_mean(close_diff, 12))
        
        alpha = self.div(numerator, denominator)
        
        return alpha.fillna(0)
