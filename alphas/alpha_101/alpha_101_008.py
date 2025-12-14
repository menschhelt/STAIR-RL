import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha
class alpha_101_008(BaseAlpha):
    """
    Alpha 101_008: condition(lt(0,ts_min(ts_delta({disk:close},1),5)),ts_delta({disk:close},1),condition(lt(ts_max(ts_delta({disk:close},1),5),0),ts_delta({disk:close},1),mul(-1,ts_delta({disk:close},1))))
    
    가격 변화 방향의 일관성에 따른 조건부 반전 전략
    """
    neutralizer_type: str = "mean"  # 평균 중립화
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_008"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_008 계산
        """
        # 1. close의 1기간 차분
        close_delta = self.ts_delta(data['close'], 1)

        # 2. close_delta의 5기간 최소값과 최대값
        min_delta_5 = self.ts_min(close_delta, 5)
        max_delta_5 = self.ts_max(close_delta, 5)
        
        # 3. 조건들
        # 조건1: 0 < min_delta_5 (최근 5일간 모두 상승)
        cond1 = min_delta_5 > 0  # 직접 비교로 변경
        
        # 조건2: max_delta_5 < 0 (최근 5일간 모두 하락)
        cond2 = max_delta_5 < 0  # 직접 비교로 변경
        
        # 4. 반전된 close_delta
        neg_close_delta = self.mul(close_delta, -1)
        
        # 5. 중첩 조건문
        # 내부 조건: cond2이면 close_delta, 아니면 -close_delta
        inner_condition = self.condition(cond2, close_delta, neg_close_delta)
        
        # 외부 조건: cond1이면 close_delta, 아니면 inner_condition
        alpha = self.condition(cond1, close_delta, inner_condition)
        
        return alpha.fillna(0)
