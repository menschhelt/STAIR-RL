import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_082(BaseAlpha):
    """
    alpha101_082: div(mul(rank(delay(div(sub({disk:high},{disk:low}),div(ts_sum({disk:close},5),5)),2)),rank(rank({disk:volume}))),div(div(sub({disk:high},{disk:low}),div(ts_sum({disk:close},5),5)),sub({disk:vwap},{disk:close})))
    
    지연된 가격 범위 대비 평균 비율과 거래량 이중 랭킹의 곱을 현재 비율의 VWAP-종가 차이 대비 비율로 나눈 값
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_082"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_082 계산
        """
        # 공통 계산: 가격 범위 대비 평균 비율
        # 1. sub(high, low)
        high_low_range = self.sub(data['high'], data['low'])
        
        # 2. div(ts_sum(close, 5), 5) - 5일 평균
        close_mean_5 = self.div(self.ts_sum(data['close'], 5), 5)
        
        # 3. div(high_low_range, close_mean_5)
        range_ratio = self.div(high_low_range, close_mean_5)
        
        # 분자 부분
        # 4. delay(range_ratio, 2)
        delayed_ratio = self.delay(range_ratio, 2)
        
        # 5. rank(delayed_ratio)
        delayed_rank = self.rank(delayed_ratio)
        
        # 6. rank(volume)
        volume_rank = self.rank(data['volume'])
        
        # 7. rank(volume_rank) - 이중 랭킹
        double_volume_rank = self.rank(volume_rank)
        
        # 8. mul(delayed_rank, double_volume_rank)
        numerator = self.mul(delayed_rank, double_volume_rank)
        
        # 분모 부분
        # 9. sub(vwap, close)
        vwap_close_diff = self.sub(data['vwap'], data['close'])
        
        # 10. div(range_ratio, vwap_close_diff)
        denominator = self.div(range_ratio, vwap_close_diff)
        
        # 11. div(numerator, denominator)
        alpha = self.div(numerator, denominator)
        
        return alpha.fillna(0)
