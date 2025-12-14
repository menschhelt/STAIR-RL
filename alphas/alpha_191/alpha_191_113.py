import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_113(BaseAlpha):
    """
    alpha191_113: div(mul(rank(delay(div(sub({disk:high},{disk:low}),div(ts_sum({disk:close},5),5)),2)),rank(rank({disk:volume}))),div(div(sub({disk:high},{disk:low}),div(ts_sum({disk:close},5),5)),sub({disk:vwap},{disk:close})))
    
    복잡한 다층 랭킹과 정규화된 변동성 비율
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_113"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_113 계산
        """
        high = data['high']
        low = data['low']
        close = data['close']
        vwap = data['vwap']
        volume = data['volume']
        
        # 기본 지표: (high - low) / (5일 평균 종가)
        hl_range = self.sub(high, low)
        close_avg_5 = self.div(self.ts_sum(close, 5), 5)
        base_indicator = self.div(hl_range, close_avg_5)
        
        # 분자: rank(delay(base_indicator, 2)) * rank(rank(volume))
        delayed_indicator = self.delay(base_indicator, 2)
        rank1 = self.rank(delayed_indicator)
        volume_double_rank = self.rank(self.rank(volume))
        numerator = self.mul(rank1, volume_double_rank)
        
        # 분모: base_indicator / (vwap - close)
        vwap_close_diff = self.sub(vwap, close)
        denominator = self.div(base_indicator, vwap_close_diff)
        
        alpha = self.div(numerator, denominator)
        
        return alpha.fillna(0)
