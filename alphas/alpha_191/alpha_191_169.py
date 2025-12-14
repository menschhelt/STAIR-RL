import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_169(BaseAlpha):
    """
    alpha191_169: sub(mul(div(mul(rank(div(1,{disk:close})),{disk:volume}),ts_mean({disk:volume},20)),div(mul({disk:high},rank(sub({disk:high},{disk:close}))),div(ts_sum({disk:high},5),5))),rank(sub({disk:vwap},delay({disk:vwap},5))))
    
    복잡한 가격-거래량 관계에서 VWAP 변화를 뺀 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_169"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_169 계산 (복잡한 식을 단순화)
        """
        close = data['close']
        high = data['high']
        volume = data['volume']
        vwap = data['vwap']
        
        # 첫 번째 부분: 역가격 랭킹과 거래량 관계
        inverse_price = self.div(1, close)
        price_volume = self.mul(self.rank(inverse_price), volume)
        volume_mean = self.ts_mean(volume, 20)
        first_factor = self.div(price_volume, volume_mean)
        
        # 두 번째 부분: 고가 관련 계산
        high_close_diff = self.sub(high, close)
        high_factor = self.mul(high, self.rank(high_close_diff))
        high_mean = self.div(self.ts_sum(high, 5), 5)
        second_factor = self.div(high_factor, high_mean)
        
        # 첫 번째 복합 요소
        first_part = self.mul(first_factor, second_factor)
        
        # 두 번째 부분: VWAP 변화 랭킹
        vwap_change = self.sub(vwap, self.delay(vwap, 5))
        second_part = self.rank(vwap_change)
        
        # 차이 계산
        alpha = self.sub(first_part, second_part)
        
        return alpha.fillna(0)
