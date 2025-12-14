import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_179(BaseAlpha):
    """
    alpha191_179: condition(lt(ts_mean({disk:volume},20),{disk:volume}),mul(mul(-1,ts_rank(abs(ts_delta({disk:close},7)),60)),sign(ts_delta({disk:close},7))),mul(-1,{disk:volume}))
    
    거래량 조건에 따른 가격 변화와 거래량 신호
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_179"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_179 계산
        """
        close = data['close']
        volume = data['volume']
        
        # 거래량 조건: 20일 평균 < 현재 거래량
        volume_mean = self.ts_mean(volume, 20)
        volume_condition = self.lt(volume_mean, volume)
        
        # 조건이 참일 때의 값
        close_delta = self.ts_delta(close, 7)
        abs_delta = self.abs(close_delta)
        delta_rank = self.ts_rank(abs_delta, 60)
        delta_sign = self.sign(close_delta)
        true_value = self.mul(self.mul(-1, delta_rank), delta_sign)
        
        # 조건이 거짓일 때의 값
        false_value = self.mul(-1, volume)
        
        # 조건부 값
        alpha = self.condition(volume_condition, true_value, false_value)
        
        return alpha.fillna(0)
