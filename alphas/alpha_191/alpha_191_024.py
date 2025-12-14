import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_024(BaseAlpha):
    """
    alpha191_024: mul(mul(-1,rank(mul(ts_delta({disk:close},7),sub(1,rank(ts_decayed_linear(div({disk:volume},ts_mean({disk:volume},20)),9)))))),add(1,rank(ts_sum({disk:returns},250))))
    
    복잡한 팩터 조합 - 가격 변화, 거래량, 수익률
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_024"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_024 계산
        """
        # returns 필드 확인 및 생성
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        close = data['close']
        volume = data['volume']
        returns = data['returns']
        
        # 7일 종가 변화
        close_delta = self.ts_delta(close, 7)
        
        # 거래량 비율
        volume_ratio = self.div(volume, self.ts_mean(volume, 20))
        
        # 9일 decayed linear
        volume_decay = self.ts_decayed_linear(volume_ratio, 9)
        volume_decay_rank = self.rank(volume_decay)
        
        # 1 - rank(decay)
        one_minus_rank = (1 - volume_decay_rank)
        
        # 첫 번째 부분
        first_part = self.mul(close_delta, one_minus_rank)
        first_part_rank = self.rank(first_part)
        
        # 250일 수익률 합계
        returns_sum = self.ts_sum(returns, 250)
        returns_rank = self.rank(returns_sum)
        
        # 최종 계산
        alpha = self.mul(
            self.mul(-1, first_part_rank),
            (1 + returns_rank)
        )
        
        return alpha.fillna(0)
