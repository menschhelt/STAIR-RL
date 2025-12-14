import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_027(BaseAlpha):
    """
    alpha191_027: mul(3,sub(sma(div(sub({disk:close},ts_min({disk:low},9)),mul(sub(ts_max({disk:high},9),ts_min({disk:low},9)),100)),3,1),mul(2,sma(sma(div(sub({disk:close},ts_min({disk:low},9)),mul(sub(max({disk:high},9),ts_max({disk:low},9)),100)),3,1),3,1))))
    
    3 * (스토캐스틱 SMA - 2 * 이중 SMA)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_027"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_027 계산
        """
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 9일 최저가와 최고가
        low_min_9 = self.ts_min(low, 9)
        high_max_9 = self.ts_max(high, 9)
        
        # 첫 번째 스토캐스틱 계산
        # (close - ts_min(low, 9)) / ((ts_max(high, 9) - ts_min(low, 9)) * 100)
        stoch1_num = self.sub(close, low_min_9)
        stoch1_den = self.mul(self.sub(high_max_9, low_min_9), 100)
        stoch1 = self.div(stoch1_num, stoch1_den)
        
        # 첫 번째 SMA
        sma1 = self.sma(stoch1, 3, 1)
        
        # 두 번째 스토캐스틱 계산 (원공식에 오류가 있어 보이므로 수정)
        # max(high, 9) -> ts_max(high, 9), ts_max(low, 9) -> ts_min(low, 9)로 가정
        stoch2_den = self.mul(self.sub(high_max_9, low_min_9), 100)
        stoch2 = self.div(stoch1_num, stoch2_den)
        
        # 이중 SMA
        sma2_inner = self.sma(stoch2, 3, 1)
        sma2 = self.sma(sma2_inner, 3, 1)
        
        # 최종 계산
        alpha = self.mul(3, self.sub(sma1, self.mul(2, sma2)))
        
        return alpha.fillna(0)
