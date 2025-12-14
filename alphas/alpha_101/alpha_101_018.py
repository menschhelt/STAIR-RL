import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_018(BaseAlpha):
    """
    alpha101_018: mul(mul(-1,sign(add(sub({disk:close},delay({disk:close},7)),ts_delta({disk:close},7)))),add(1,rank(add(1,ts_sum({disk:returns},250)))))
    
    7일 가격 변화의 방향성과 장기 수익률 합계의 복합 팩터
    """
    # 기본 파라미터 정의
    default_params = {
        "delay_window": 7,
        "delta_window": 7,
        "returns_sum_window": 250
    }

    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_018"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_018 계산
        """
        # 수익률 계산
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        # 1. close - delay(close, 7)
        close_lag7 = self.delay(data['close'], self.params["delay_window"])
        close_diff = self.sub(data['close'], close_lag7)
        
        # 2. ts_delta(close, 7)
        close_delta7 = self.ts_delta(data['close'], self.params["delta_window"])
        
        # 3. add(..., ...)
        sum_changes = self.add(close_diff, close_delta7)
        
        # 4. sign(...)
        sign_changes = self.sign(sum_changes)
        
        # 5. mul(-1, ...)
        neg_sign = self.mul(sign_changes, -1)
        
        # 6. ts_sum(returns, 250)
        returns_sum = self.ts_sum(data['returns'], self.params["returns_sum_window"])
        
        # 7. add(1, returns_sum)
        returns_plus1 = self.add(returns_sum, 1)
        
        # 8. rank(...)
        returns_rank = self.rank(returns_plus1)
        
        # 9. add(1, rank)
        rank_plus1 = self.add(returns_rank, 1)
        
        # 10. 최종 곱하기
        alpha = self.mul(neg_sign, rank_plus1)
        
        return alpha.fillna(0)
