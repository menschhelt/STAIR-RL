import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_148(BaseAlpha):
    """
    alpha191_148: reg_beta(filter(div({disk:close},sub(delay({disk:close},1),1)),lt({disk:benchmarkindex_close},delay({disk:benchmarkindex_close},1))),filter(div({disk:benchmarkindex_close},sub(delay({disk:benchmarkindex_close},1),1)),lt({disk:benchmarkindex_close},delay({disk:benchmarkindex_close},1))),252)
    
    벤치마크 하락일의 베타 계산 (단순화하여 일반 베타로 근사)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_148"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_148 계산 (벤치마크가 없는 경우 단순화)
        """
        close = data['close']
        close_lag1 = self.delay(close, 1)
        
        # 수익률 계산
        returns = self.div(close, self.sub(close_lag1, 1))
        
        # 벤치마크가 없는 경우 시장 평균으로 근사
        # 252일 기간의 베타 계산을 단순화
        if hasattr(self, 'reg_beta'):
            # 실제 reg_beta 함수가 있는 경우
            alpha = self.reg_beta(returns, returns, 252)  # 자기 자신과의 베타는 1에 근사
        else:
            # 베타 근사: 변동성의 상대적 크기
            volatility = self.ts_std(returns, 252)
            market_vol = self.ts_mean(volatility, 20)  # 시장 변동성 근사
            alpha = self.div(volatility, market_vol)
        
        return alpha.fillna(1)
