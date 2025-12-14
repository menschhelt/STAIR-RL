import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_032(BaseAlpha):
    """
    alpha191_032: mul(mul(add(mul(-1,ts_min({disk:low},5)),delay(ts_min({disk:low},5),5)),rank(div(sub(ts_sum({disk:returns},240),ts_sum({disk:returns},20)),220))),ts_rank({disk:volume},5))
    
    복잡한 저가 변화와 수익률, 거래량의 조합
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_032"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_032 계산
        """
        # returns 필드 확인 및 생성
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        low = data['low']
        volume = data['volume']
        returns = data['returns']
        
        # 5일 최저가
        low_min_5 = self.ts_min(low, 5)
        
        # (-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)
        part1 = self.add(
            self.mul(-1, low_min_5),
            self.delay(low_min_5, 5)
        )
        
        # (ts_sum(returns, 240) - ts_sum(returns, 20)) / 220
        returns_240 = self.ts_sum(returns, 240)
        returns_20 = self.ts_sum(returns, 20)
        returns_ratio = self.div(
            self.sub(returns_240, returns_20),
            220
        )
        part2 = self.rank(returns_ratio)
        
        # 거래량 5일 랭킹
        volume_rank = self.ts_rank(volume, 5)
        
        # 최종 계산
        alpha = self.mul(self.mul(part1, part2), volume_rank)
        
        return alpha.fillna(0)
