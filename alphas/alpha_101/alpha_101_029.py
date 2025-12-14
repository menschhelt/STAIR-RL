import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_029(BaseAlpha):
    """
    alpha101_029: div(mul(sub(1.0,rank(add(add(sign(sub({disk:close},delay({disk:close},1))),sign(sub(delay({disk:close},1),delay({disk:close},2)))),sign(sub(delay({disk:close},2),delay({disk:close},3)))))),ts_sum({disk:volume},5)),ts_sum({disk:volume},20))
    
    연속 3일간 가격 변화 방향의 패턴과 거래량 비율의 복합 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_029"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_029 계산
        """
        # 1. 지연된 종가들
        close_lag1 = self.delay(data['close'], 1)
        close_lag2 = self.delay(data['close'], 2)
        close_lag3 = self.delay(data['close'], 3)
        
        # 2. 각 기간의 방향성
        sign1 = self.sign(self.sub(data['close'], close_lag1))
        sign2 = self.sign(self.sub(close_lag1, close_lag2))
        sign3 = self.sign(self.sub(close_lag2, close_lag3))
        
        # 3. 모든 방향성의 합
        sign_sum = self.add(
            self.add(sign1, sign2),
            sign3
        )
        
        # 4. rank(sign_sum)
        sign_rank = self.rank(sign_sum)
        
        # 5. sub(1.0, sign_rank) - 스칼라가 첫 번째이므로 직접 연산
        one_minus_rank = 1.0 - sign_rank
        
        # 6. ts_sum(volume, 5)
        volume_sum_5 = self.ts_sum(data['volume'], 5)
        
        # 7. ts_sum(volume, 20)
        volume_sum_20 = self.ts_sum(data['volume'], 20)
        
        # 8. mul(one_minus_rank, volume_sum_5)
        numerator = self.mul(one_minus_rank, volume_sum_5)
        
        # 9. div(numerator, volume_sum_20)
        alpha = self.div(numerator, volume_sum_20)
        
        return alpha.fillna(0)
