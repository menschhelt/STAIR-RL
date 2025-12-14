import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_059(BaseAlpha):
    """
    alpha101_059: sub(0,mul(1,sub(mul(2,twise_a_scale(rank(mul(div(sub(sub({disk:close},{disk:low}),sub({disk:high},{disk:close})),sub({disk:high},{disk:low})),{disk:volume})))),twise_a_scale(rank(ts_argmax({disk:close},10))))))
    
    정규화된 가격 위치-거래량 곱의 랭킹과 10기간 종가 최대값 위치 랭킹의 차이의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_059"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_059 계산
        """
        # 첫 번째 부분: 복잡한 가격 위치 지표
        # 1. sub(close, low)
        close_low = self.sub(data['close'], data['low'])
        
        # 2. sub(high, close)
        high_close = self.sub(data['high'], data['close'])
        
        # 3. sub(close_low, high_close)
        price_position_num = self.sub(close_low, high_close)
        
        # 4. sub(high, low)
        high_low = self.sub(data['high'], data['low'])
        
        # 5. div(price_position_num, high_low)
        price_position = self.div(price_position_num, high_low)
        
        # 6. mul(price_position, volume)
        position_volume = self.mul(price_position, data['volume'])
        
        # 7. rank(position_volume)
        position_rank = self.rank(position_volume)
        
        # 8. twise_a_scale(position_rank)
        scaled_position = self.twise_a_scale(position_rank, 1)
        
        # 9. mul(2, scaled_position)
        first_part = self.mul(2, scaled_position)
        
        # 두 번째 부분: 종가 최대값 위치
        # 10. ts_argmax(close, 10)
        argmax_10 = self.ts_argmax(data['close'], 10)
        
        # 11. rank(argmax_10)
        argmax_rank = self.rank(argmax_10)
        
        # 12. twise_a_scale(argmax_rank)
        second_part = self.twise_a_scale(argmax_rank, 1)
        
        # 13. sub(first_part, second_part)
        diff = self.sub(first_part, second_part)
        
        # 14. sub(0, diff) = -diff
        alpha = self.mul(diff, -1)
        
        return alpha.fillna(0)
