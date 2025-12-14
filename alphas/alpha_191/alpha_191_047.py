import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_047(BaseAlpha):
    """
    alpha191_047: mul(-1,div(mul(rank(add(add(sign(sub({disk:close},delay({disk:close},1))),sign(sub(delay({disk:close},1),delay({disk:close},2)))),sign(sub(delay({disk:close},2),delay({disk:close},3))))),ts_sum({disk:volume},5)),ts_sum({disk:volume},20)))
    
    -1 * (3일 연속 가격 변화 부호 합계 랭킹 * 5일 거래량 합계) / 20일 거래량 합계
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_047"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_047 계산
        """
        close = data['close']
        volume = data['volume']
        
        # 지연된 종가들
        close_lag1 = self.delay(close, 1)
        close_lag2 = self.delay(close, 2)
        close_lag3 = self.delay(close, 3)
        
        # 3일 연속 가격 변화의 부호
        sign1 = self.sign(self.sub(close, close_lag1))
        sign2 = self.sign(self.sub(close_lag1, close_lag2))
        sign3 = self.sign(self.sub(close_lag2, close_lag3))
        
        # 부호들의 합계
        sign_sum = self.add(
            self.add(sign1, sign2),
            sign3
        )
        
        # 랭킹
        sign_rank = self.rank(sign_sum)
        
        # 거래량 합계
        volume_sum_5 = self.ts_sum(volume, 5)
        volume_sum_20 = self.ts_sum(volume, 20)
        
        # 계산
        numerator = self.mul(sign_rank, volume_sum_5)
        ratio = self.div(numerator, volume_sum_20)
        
        # -1 곱하기
        alpha = self.mul(-1, ratio)
        
        return alpha.fillna(0)
