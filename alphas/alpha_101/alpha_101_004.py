import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

# 부모 디렉토리를 import 경로에 추가
from alphas.base.base import BaseAlpha

class alpha_101_004(BaseAlpha):
    """
    Alpha 101_004: mul(rank(sub({disk:open},div(ts_sum({disk:vwap},10),10))),mul(-1,abs(rank(sub({disk:close},{disk:vwap})))))
    
    시가-VWAP 편차와 종가-VWAP 편차의 조합
    """
    
    # 알파별 처리 설정 - 모든 처리 단계 포함
    neutralizer_type: str = "mean"  # 평균 중립화
    decay_period: int = 3
    
    
    @property
    def name(self) -> str:
        return "alpha_101_004"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_004 계산
        """
        # VWAP 계산 (없는 경우)
        if 'vwap' not in data:
            data['vwap'] = self.vwap_calc(
                data['high'], 
                data['low'], 
                data['close'], 
                data['volume']
            )
        
        # 1. ts_sum(vwap, 10) / 10 (10기간 VWAP 평균)
        vwap_avg_10 = self.div(self.ts_sum(data['vwap'], 10), 10)

        # 2. open - vwap_avg_10
        open_vwap_diff = self.sub(data['open'], vwap_avg_10)

        # 3. rank(open_vwap_diff)
        first_part = self.rank(open_vwap_diff)

        # 4. close - vwap
        close_vwap_diff = self.sub(data['close'], data['vwap'])

        # 5. rank(close_vwap_diff)
        close_vwap_rank = self.rank(close_vwap_diff)

        # 6. abs(close_vwap_rank)
        abs_close_rank = self.abs(close_vwap_rank)

        # 7. -1 * abs_close_rank
        second_part = self.mul(abs_close_rank, -1)
        
        # 8. first_part * second_part
        alpha = self.mul(first_part, second_part)
        
        return alpha.fillna(0)
