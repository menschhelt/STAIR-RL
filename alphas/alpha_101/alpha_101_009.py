import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_009(BaseAlpha):
    """
    Alpha 101_009: rank(condition(lt(0,ts_min(ts_delta({disk:close},1),4)),ts_delta({disk:close},1),condition(lt(ts_max(ts_delta({disk:close},1),4),0),ts_delta({disk:close},1),mul(-1,ts_delta({disk:close},1)))))
    
    가격 변화 방향 일관성에 따른 중첩 조건부 랭킹
    """
    neutralizer_type: str = "mean"  # 평균 중립화

    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_009"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_009 계산
        """
        # 1. ts_delta(close, 1)
        close_delta = self.ts_delta(data['close'], 1)

        # 2. ts_min(ts_delta(close, 1), 4)
        min_delta_4 = self.ts_min(close_delta, 4)

        # 3. ts_max(ts_delta(close, 1), 4)
        max_delta_4 = self.ts_max(close_delta, 4)
        
        # 4. 조건들
        # 조건1: 0 < ts_min(...) (최근 4일간 모두 상승)
        cond1 = min_delta_4 > 0  # 직접 비교로 변경
        
        # 조건2: ts_max(...) < 0 (최근 4일간 모두 하락)
        cond2 = max_delta_4 < 0  # 직접 비교로 변경
        
        # 5. -1 * ts_delta(close, 1)
        neg_close_delta = close_delta * -1  # 직접 스칼라 곱셈으로 변경
        
        # 6. 내부 조건문: cond2이면 close_delta, 아니면 neg_close_delta
        inner_condition = self.condition(cond2, close_delta, neg_close_delta)
        
        # 7. 외부 조건문: cond1이면 close_delta, 아니면 inner_condition
        condition_result = self.condition(cond1, close_delta, inner_condition)
        
        # 8. rank(...)
        alpha = self.rank(condition_result)
        
        return alpha.fillna(0)
