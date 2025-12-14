import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_017(BaseAlpha):
    """
    alpha101_017: mul(-1,rank(add(add(ts_std(abs(sub({disk:close},{disk:open})),5),sub({disk:close},{disk:open})),ts_corr({disk:close},{disk:open},10))))
    
    종가-시가 차이의 변동성, 수익률, 종가-시가 상관관계의 복합 팩터
    """
    # 기본 파라미터 정의
    default_params = {
        "std_window": 5,
        "corr_window": 10
    }

    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_017"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_017 계산
        """
        # 1. close - open
        close_open_diff = self.sub(data['close'], data['open'])
        
        # 2. ts_std(abs(close - open), 5)
        abs_diff = self.abs(close_open_diff)
        std_5 = self.ts_std(abs_diff, self.params["std_window"])
        
        # 3. ts_corr(close, open, 10)
        corr_10 = self.ts_corr(data['close'], data['open'], self.params["corr_window"])
        
        # 4. add(add(std_5, close_open_diff), corr_10)
        sum_all = self.add(
            self.add(std_5, close_open_diff),
            corr_10
        )
        
        # 5. rank(...)
        ranked = self.rank(sum_all)
        
        # 6. mul(-1, ...)
        alpha = self.mul(ranked, -1)
        
        return alpha.fillna(0)
