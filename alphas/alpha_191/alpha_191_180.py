import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_180(BaseAlpha):
    """
    alpha191_180: div(ts_sum(sub(sub(div({disk:close},sub(delay({disk:close},1),1)),ts_mean(div({disk:close},sub(delay({disk:close},1),1)),20)),pow(sub({disk:benchmarkindex_close},ts_mean({disk:benchmarkindex_close},20)),2)),20),sum(pow(sub({disk:benchmarkindex_close},ts_mean({disk:benchmarkindex_close},20)),3)))
    
    벤치마크 대비 수익률 편차의 복잡한 지표 (벤치마크 없이 단순화)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_180"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_180 계산 (벤치마크가 없는 경우 단순화)
        """
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # 수익률 계산
        returns = self.div(close, self.sub(close_lag1, 1))
        returns_mean = self.ts_mean(returns, 20)
        
        # 수익률 편차
        returns_deviation = self.sub(returns, returns_mean)
        
        # 벤치마크가 없으므로 시장 전체 움직임을 자기 자신으로 근사
        market_deviation = self.pow(returns_deviation, 2)  # 제곱으로 변동성 근사
        
        # 분자: 20일 합
        numerator = self.ts_sum(self.sub(returns_deviation, market_deviation), 20)
        
        # 분모: 변동성의 3제곱 합 (단순화)
        market_vol_cubed = self.pow(returns_deviation, 3)
        denominator = self.ts_sum(market_vol_cubed, 20)
        
        # 비율 계산
        alpha = self.div(numerator, denominator.replace(0, 1))
        
        return alpha.fillna(0)
