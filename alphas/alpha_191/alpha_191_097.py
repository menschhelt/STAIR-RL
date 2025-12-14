import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_097(BaseAlpha):
    """
    alpha191_097: condition(or_(lt(div(ts_delta(div(ts_sum({disk:close},100),100),100),delay({disk:close},100)),0.05),eq(div(ts_delta(div(ts_sum({disk:close},100),100),100),delay({disk:close},100)),0.05)),mul(-1,sub({disk:close},ts_min({disk:close},100))),mul(-1,ts_delta({disk:close},3)))
    
    복잡한 장기 변화율 기반 조건부 신호
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_097"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_097 계산
        """
        close = data['close']
        
        # 100일 평균 종가
        close_mean_100 = self.div(self.ts_sum(close, 100), 100)
        
        # 100일 평균의 100일 변화
        mean_delta = self.ts_delta(close_mean_100, 100)
        
        # 100일 전 종가
        close_lag100 = self.delay(close, 100)
        
        # 변화율
        change_ratio = self.div(mean_delta, close_lag100)
        
        # 조건: change_ratio <= 0.05
        condition = self.or_(
            self.lt(change_ratio, 0.05),
            self.eq(change_ratio, 0.05)
        )
        
        # 조건이 참이면: -1 * (close - ts_min(close, 100))
        # 조건이 거짓이면: -1 * ts_delta(close, 3)
        alpha = self.condition(
            condition,
            self.mul(-1, self.sub(close, self.ts_min(close, 100))),
            self.mul(-1, self.ts_delta(close, 3))
        )
        
        return alpha.fillna(0)
