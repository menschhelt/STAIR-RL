import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_054(BaseAlpha):
    """
    alpha191_054: 극도로 복잡한 가격 변화율 공식 (단순화 버전)
    
    복잡한 TR(True Range) 기반 지표의 단순화
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_054"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_054 계산 (단순화)
        """
        high = data['high']
        low = data['low']
        close = data['close']
        open_price = data['open']
        
        # 기본 True Range 계산
        close_lag1 = self.delay(close, 1)
        
        tr1 = self.abs(self.sub(high, low))
        tr2 = self.abs(self.sub(high, close_lag1))
        tr3 = self.abs(self.sub(low, close_lag1))
        
        true_range = self.max(tr1, self.max(tr2, tr3))
        
        # 가격 변화
        price_change = self.sub(close, close_lag1)
        
        # 정규화된 변화율
        normalized_change = self.div(price_change, true_range)
        
        # 20일 합계 (원공식에서 20)
        alpha = self.ts_sum(self.mul(16, normalized_change), 20)
        
        return alpha.fillna(0)
