import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_084(BaseAlpha):
    """
    alpha191_084: mul(ts_rank(div({disk:volume},ts_mean({disk:volume},20)),20),ts_rank(mul(-1,ts_delta({disk:close},7)),8))
    
    거래량 비율 ts_rank * 가격 변화 ts_rank
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_084"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_084 계산
        """
        volume = data['volume']
        close = data['close']
        
        # 거래량 / 20일 평균 거래량
        volume_ratio = self.div(volume, self.ts_mean(volume, 20))
        ts_rank1 = self.ts_rank(volume_ratio, 20)
        
        # -1 * 7일 종가 변화
        close_delta = self.mul(-1, self.ts_delta(close, 7))
        ts_rank2 = self.ts_rank(close_delta, 8)
        
        # 곱하기
        alpha = self.mul(ts_rank1, ts_rank2)
        
        return alpha.fillna(0)
