import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_033(BaseAlpha):
    """
    alpha101_033: rank(add(sub(1,rank(div(ts_std({disk:returns},2),ts_std({disk:returns},5)))),sub(1,rank(ts_delta({disk:close},1)))))
    
    수익률 변동성 비율과 가격 변화의 역순 랭킹 복합 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_033"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_033 계산
        """
        # 수익률 계산
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        # 첫 번째 부분: 수익률 변동성 비율
        # 1. ts_std(returns, 2)
        returns_std_2 = self.ts_std(data['returns'], 2)
        
        # 2. ts_std(returns, 5)
        returns_std_5 = self.ts_std(data['returns'], 5)
        
        # 3. div(returns_std_2, returns_std_5)
        std_ratio = self.div(returns_std_2, returns_std_5)
        
        # 4. rank(std_ratio)
        std_rank = self.rank(std_ratio)
        
        # 5. sub(1, std_rank)
        first_part = (1 - std_rank)
        
        # 두 번째 부분: 가격 변화
        # 6. ts_delta(close, 1)
        close_delta = self.ts_delta(data['close'], 1)
        
        # 7. rank(close_delta)
        delta_rank = self.rank(close_delta)
        
        # 8. sub(1, delta_rank)
        second_part = (1 - delta_rank)
        
        # 9. add(first_part, second_part)
        sum_parts = self.add(first_part, second_part)
        
        # 10. rank(sum_parts)
        alpha = self.rank(sum_parts)
        
        return alpha.fillna(0)
