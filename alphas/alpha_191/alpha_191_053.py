import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_053(BaseAlpha):
    """
    alpha191_053: mul(-1,rank(add(add(std(abs(sub({disk:close},{disk:open}))),sub({disk:close},{disk:open})),ts_corr({disk:close},{disk:open},10))))
    
    -1 * (일중 변동성 표준편차 + 일중 변화 + 종가-시가 상관관계) 랭킹
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_053"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_053 계산
        """
        close = data['close']
        open_price = data['open']
        
        # |close - open|의 표준편차
        intraday_change = self.abs(self.sub(close, open_price))
        intraday_std = self.ts_std(intraday_change, 20)  # 기간 추정
        
        # close - open
        daily_change = self.sub(close, open_price)
        
        # 종가와 시가의 10일 상관관계
        corr = self.ts_corr(close, open_price, 10)
        
        # 합계
        combined = self.add(
            self.add(intraday_std, daily_change),
            corr
        )
        
        # 랭킹 후 -1 곱하기
        alpha = self.mul(-1, self.rank(combined))
        
        return alpha.fillna(0)
