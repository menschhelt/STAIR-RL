import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_009(BaseAlpha):
    """
    alpha191_009: rank(pow(condition(lt({disk:returns},0),ts_std({disk:returns},20),{disk:close}),2))
    
    조건부 값의 제곱에 대한 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_009"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_009 계산
        """
        # returns 필드 확인 및 생성
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        returns = data['returns']
        close = data['close']
        
        # 20일 수익률 표준편차
        returns_std = self.ts_std(returns, 20)
        
        # 조건: returns < 0이면 std, 그렇지 않으면 close
        condition_result = self.condition(
            self.lt(returns, 0),
            returns_std,
            close
        )
        
        # 제곱
        squared = self.pow(condition_result, 2)
        
        # 랭킹
        alpha = self.rank(squared)
        
        return alpha.fillna(0)
