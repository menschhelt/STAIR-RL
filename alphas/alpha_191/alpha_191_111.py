import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_111(BaseAlpha):
    """
    alpha191_111: div(sub(ts_sum(condition(gt(sub({disk:close},delay({disk:close},1)),0),sub({disk:close},delay({disk:close},1)),0),12),ts_sum(condition(lt(sub({disk:close},delay({disk:close},1)),0),abs(sub({disk:close},delay({disk:close},1))),0),12)),mul(add(ts_sum(condition(gt(sub({disk:close},delay({disk:close},1)),0),sub({disk:close},delay({disk:close},1)),0),12),ts_sum(condition(lt(sub({disk:close},delay({disk:close},1)),0),abs(sub({disk:close},delay({disk:close},1))),0),12)),100))
    
    12일간 (상승분 합계 - 하락분 합계) / (전체 변화 합계 * 100) = 방향성 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_111"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_111 계산
        """
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # 일일 변화
        daily_change = self.sub(close, close_lag1)
        
        # 상승분: 상승일이면 변화량, 그렇지 않으면 0
        up_change = self.condition(
            self.gt(daily_change, 0),
            daily_change,
            0
        )
        
        # 하락분: 하락일이면 절대변화량, 그렇지 않으면 0
        down_change = self.condition(
            self.lt(daily_change, 0),
            self.abs(daily_change),
            0
        )
        
        # 12일 합계
        up_sum = self.ts_sum(up_change, 12)
        down_sum = self.ts_sum(down_change, 12)
        
        # 분자: 상승분 - 하락분
        numerator = self.sub(up_sum, down_sum)
        
        # 분모: (상승분 + 하락분) * 100
        denominator = self.mul(self.add(up_sum, down_sum), 100)
        
        alpha = self.div(numerator, denominator)
        
        return alpha.fillna(0)
