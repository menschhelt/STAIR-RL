import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_168(BaseAlpha):
    """
    alpha191_168: sma(sub(ts_mean(delay(sma(sub({disk:close},delay({disk:close},1)),9,1),1),12),ts_mean(delay(sma(sub({disk:close},delay({disk:close},1)),9,1),1),26)),10,1)
    
    가격 변화의 복잡한 다중 기간 평활화 차이
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_168"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_168 계산
        """
        close = data['close']
        
        # 가격 변화
        price_change = self.sub(close, self.delay(close, 1))
        
        # 9일 SMA 후 1일 지연
        sma_9 = self.sma(price_change, 9, 1)
        sma_delayed = self.delay(sma_9, 1)
        
        # 12일과 26일 평균
        mean_12 = self.ts_mean(sma_delayed, 12)
        mean_26 = self.ts_mean(sma_delayed, 26)
        
        # 차이
        diff = self.sub(mean_12, mean_26)
        
        # 10일 SMA
        alpha = self.sma(diff, 10, 1)
        
        return alpha.fillna(0)
