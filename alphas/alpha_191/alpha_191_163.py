import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_163(BaseAlpha):
    """
    alpha191_163: sma(div(sub(condition(gt({disk:close},delay({disk:close},1)),div(1,sub({disk:close},delay({disk:close},1))),1),min(...)),mul(sub({disk:high},{disk:low}),100)),13,2)
    
    상승일 역수익률과 가격 범위 기반 지표의 평활화
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_163"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_163 계산 (복잡한 식을 단순화)
        """
        close = data['close']
        high = data['high']
        low = data['low']
        close_lag1 = self.delay(close, 1)
        
        # 상승 조건
        is_up = self.gt(close, close_lag1)
        
        # 가격 변화
        price_change = self.sub(close, close_lag1)
        
        # 조건: 상승일이면 역수익률, 아니면 1 (단순화)
        inverse_returns = self.condition(is_up, self.div(1, price_change.replace(0, 1)), 1)
        
        # 가격 범위
        price_range = self.sub(high, low)
        range_scaled = self.mul(price_range, 100)
        
        # 정규화된 지표
        normalized_indicator = self.div(self.sub(inverse_returns, 1), range_scaled)
        
        # 13일 SMA
        alpha = self.sma(normalized_indicator, 13, 2)
        
        return alpha.fillna(0)
