import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_051(BaseAlpha):
    """
    alpha101_051: mul(mul(add(mul(-1,ts_min({disk:low},5)),delay(ts_min({disk:low},5),5)),rank(div(sub(ts_sum({disk:returns},240),ts_sum({disk:returns},20)),220))),ts_rank({disk:volume},5))
    
    저가의 5기간 최소값 변화, 장단기 수익률 차이, 거래량 랭킹의 복합 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_051"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_051 계산
        """
        # 수익률 계산
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        # 첫 번째 부분: 저가 최소값 변화
        # 1. ts_min(low, 5)
        low_min_5 = self.ts_min(data['low'], 5)
        
        # 2. mul(-1, low_min_5)
        neg_low_min = self.mul(low_min_5, -1)
        
        # 3. delay(low_min_5, 5)
        low_min_delayed = self.delay(low_min_5, 5)
        
        # 4. add(neg_low_min, low_min_delayed)
        low_change = self.add(neg_low_min, low_min_delayed)
        
        # 두 번째 부분: 장단기 수익률 차이
        # 5. ts_sum(returns, 240)
        returns_long = self.ts_sum(data['returns'], 240)
        
        # 6. ts_sum(returns, 20)
        returns_short = self.ts_sum(data['returns'], 20)
        
        # 7. sub(returns_long, returns_short)
        returns_diff = self.sub(returns_long, returns_short)
        
        # 8. div(returns_diff, 220)
        returns_avg_diff = self.div(returns_diff, 220)
        
        # 9. rank(returns_avg_diff)
        returns_rank = self.rank(returns_avg_diff)
        
        # 세 번째 부분: 거래량 랭킹
        # 10. ts_rank(volume, 5)
        volume_rank = self.ts_rank(data['volume'], 5)
        
        # 11. 모든 부분들을 곱하기
        alpha = self.mul(
            self.mul(low_change, returns_rank),
            volume_rank
        )
        
        return alpha.fillna(0)
