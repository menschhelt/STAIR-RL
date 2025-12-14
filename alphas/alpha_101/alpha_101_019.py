import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_019(BaseAlpha):
    """
    alpha101_019: mul(mul(mul(-1,rank(sub({disk:open},delay({disk:high},1)))),rank(sub({disk:open},delay({disk:close},1)))),rank(sub({disk:open},delay({disk:low},1))))
    
    시가와 전일 고가, 종가, 저가의 차이를 랭킹한 복합 팩터
    """
    # 기본 파라미터 정의
    default_params = {
        "delay_window": 1
    }

    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_019"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_019 계산
        """
        # 1. rank(sub(open, delay(high, 1)))
        high_lag1 = self.delay(data['high'], self.params["delay_window"])
        open_high_diff = self.sub(data['open'], high_lag1)
        open_high_rank = self.rank(open_high_diff)
        neg_open_high_rank = self.mul(open_high_rank, -1)
        
        # 2. rank(sub(open, delay(close, 1)))
        close_lag1 = self.delay(data['close'], self.params["delay_window"])
        open_close_diff = self.sub(data['open'], close_lag1)
        open_close_rank = self.rank(open_close_diff)
        
        # 3. rank(sub(open, delay(low, 1)))
        low_lag1 = self.delay(data['low'], self.params["delay_window"])
        open_low_diff = self.sub(data['open'], low_lag1)
        open_low_rank = self.rank(open_low_diff)
        
        # 4. 모든 것들을 곱하기
        alpha = self.mul(
            self.mul(neg_open_high_rank, open_close_rank),
            open_low_rank
        )
        
        return alpha.fillna(0)
