import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_116(BaseAlpha):
    """
    alpha191_116: mul(mul(ts_rank({disk:volume},32),sub(1,ts_rank(sub(add({disk:close},{disk:high}),{disk:low}),16))),sub(1,ts_rank({disk:returns},32)))
    
    거래량 ts_rank * (1 - 가격 위치 ts_rank) * (1 - 수익률 ts_rank)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_116"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_116 계산
        """
        # returns 필드 확인 및 생성
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        volume = data['volume']
        close = data['close']
        high = data['high']
        low = data['low']
        returns = data['returns']
        
        # 거래량 32일 ts_rank
        volume_rank = self.ts_rank(volume, 32)
        
        # (close + high) - low = 가격 위치
        price_position = self.sub(self.add(close, high), low)
        price_rank = self.ts_rank(price_position, 16)
        inv_price_rank = (1 - price_rank)
        
        # 수익률 32일 ts_rank
        returns_rank = self.ts_rank(returns, 32)
        inv_returns_rank = (1 - returns_rank)
        
        # 곱하기
        alpha = self.mul(
            self.mul(volume_rank, inv_price_rank),
            inv_returns_rank
        )
        
        return alpha.fillna(0)
