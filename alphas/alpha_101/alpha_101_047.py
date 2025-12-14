import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_047(BaseAlpha):
    """
    alpha101_047: div(grouped_demean(div(mul(ts_corr(ts_delta({disk:close},1),ts_delta(delay({disk:close},1),1),250),ts_delta({disk:close},1)),{disk:close}),{disk:industry_group_lv3}),ts_sum(pow(div(ts_delta({disk:close},1),delay({disk:close},1)),2),250))
    
    업종별 표준화된 가격 변화 상관관계와 수익률 변동성의 비율
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_047"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_047 계산
        """
        # 1. ts_delta(close, 1)
        close_delta1 = self.ts_delta(data['close'], 1)
        
        # 2. delay(close, 1)
        close_lag1 = self.delay(data['close'], 1)
        
        # 3. ts_delta(close_lag1, 1)
        lag_delta1 = self.ts_delta(close_lag1, 1)
        
        # 4. ts_corr(close_delta1, lag_delta1, 250)
        corr_250 = self.ts_corr(close_delta1, lag_delta1, 250)
        
        # 5. mul(corr_250, close_delta1)
        corr_product = self.mul(corr_250, close_delta1)
        
        # 6. div(corr_product, close)
        ratio = self.div(corr_product, data['close'])
        
        # 7. industry_group_lv3가 없으므로 간단히 처리
        # grouped_demean은 업종별 평균 제거이지만, 데이터가 없으므로 전체 평균 제거로 대체
        demeaned = ratio - ratio.mean()
        
        # 분모 부분: 수익률 변동성
        # 8. div(close_delta1, close_lag1) - 수익률
        returns = self.div(close_delta1, close_lag1)
        
        # 9. pow(returns, 2)
        returns_squared = self.pow(returns, 2)
        
        # 10. ts_sum(returns_squared, 250)
        volatility = self.ts_sum(returns_squared, 250)
        
        # 11. div(demeaned, volatility)
        alpha = self.div(demeaned, volatility)
        
        return alpha.fillna(0)
