import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_089(BaseAlpha):
    """
    alpha101_089: mul(pow(rank(sub({disk:close},ts_max({disk:close},4.66719))),ts_rank(ts_corr(grouped_demean(ts_mean({disk:amount},40),{disk:industry_group_lv3}),{disk:low},5.38375),3.21856)),-1)
    
    종가와 최대 종가의 차이 랭킹을 업종별 표준화된 거래대금-저가 상관관계의 시계열 랭킹으로 거듭제곱한 값의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_089"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_089 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 밑: 종가와 최대 종가의 차이 랭킹
        # 1. ts_max(close, 4.66719) -> 반올림하여 5
        max_window = int(round(4.66719))
        close_max = self.ts_max(data['close'], max_window)
        
        # 2. sub(close, close_max)
        close_diff = self.sub(data['close'], close_max)
        
        # 3. rank(close_diff)
        base = self.rank(close_diff)
        
        # 지수: 업종별 표준화된 거래대금-저가 상관관계의 시계열 랭킹
        # 4. ts_mean(amount, 40)
        amount_mean = self.ts_mean(data['amount'], 40)
        
        # 5. grouped_demean(amount_mean, industry_group_lv3) -> amount_mean - mean(amount_mean)
        amount_demeaned = amount_mean - amount_mean.mean()
        
        # 6. ts_corr(amount_demeaned, low, 5.38375) -> 반올림하여 5
        corr_window = int(round(5.38375))
        amount_low_corr = self.ts_corr(amount_demeaned, data['low'], corr_window)
        
        # 7. ts_rank(amount_low_corr, 3.21856) -> 반올림하여 3
        rank_window = int(round(3.21856))
        power = self.ts_rank(amount_low_corr, rank_window)
        
        # 8. pow(base, power)
        powered = self.pow(base, power)
        
        # 9. mul(powered, -1)
        alpha = self.mul(powered, -1)
        
        return alpha.fillna(0)
