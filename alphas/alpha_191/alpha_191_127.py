import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_127(BaseAlpha):
    """
    alpha191_127: sub(100,div(100,add(1,div(ts_sum(condition(gt(div(add({disk:high},add({disk:low},{disk:close})),3),delay(div(add({disk:high},add({disk:low},{disk:close})),3),1)),div(add({disk:high},add({disk:low},{disk:close})),mul(3,{disk:volume})),0),14),ts_sum(condition(lt(div(add({disk:high},add({disk:low},{disk:close})),3),delay(div(add({disk:high},add({disk:low},{disk:close})),3),1)),div(add({disk:high},add({disk:low},{disk:close})),mul(3,{disk:volume})),0),14)))))
    
    거래량 가중 전형가의 상승/하락 압력 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_127"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_127 계산
        """
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # 전형가: (high + low + close) / 3
        typical_price = self.div(
            self.add(high, self.add(low, close)),
            3
        )
        
        # 지연된 전형가
        typical_price_lag1 = self.delay(typical_price, 1)
        
        # 거래량 가중 전형가: typical_price / (3 * volume)
        volume_weighted = self.div(
            self.add(high, self.add(low, close)),
            self.mul(3, volume)
        )
        
        # 상승 조건
        up_condition = self.condition(
            self.gt(typical_price, typical_price_lag1),
            volume_weighted,
            0
        )
        
        # 하락 조건
        down_condition = self.condition(
            self.lt(typical_price, typical_price_lag1),
            volume_weighted,
            0
        )
        
        # 14일 합계
        up_sum = self.ts_sum(up_condition, 14)
        down_sum = self.ts_sum(down_condition, 14)
        
        # 비율 계산
        ratio = self.div(up_sum, down_sum)
        
        # 최종 지표 - 스칼라가 첫 번째이므로 직접 연산
        alpha = 100 - self.div(100, 1 + ratio)
        
        return alpha.fillna(0)
