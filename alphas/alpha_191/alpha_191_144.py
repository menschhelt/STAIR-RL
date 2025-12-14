import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_144(BaseAlpha):
    """
    alpha191_144: div(sub(ts_mean({disk:volume},9),ts_mean({disk:volume},26)),mul(ts_mean({disk:volume},12),100))
    
    단기-장기 거래량 평균 차이를 중기 거래량으로 정규화
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_144"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_144 계산
        """
        volume = data['volume']
        
        # 거래량 이동평균들
        volume_mean_9 = self.ts_mean(volume, 9)
        volume_mean_26 = self.ts_mean(volume, 26)
        volume_mean_12 = self.ts_mean(volume, 12)
        
        # 분자: 9일 평균 - 26일 평균
        numerator = self.sub(volume_mean_9, volume_mean_26)
        
        # 분모: 12일 평균 * 100
        denominator = self.mul(volume_mean_12, 100)
        
        # 비율 계산
        alpha = self.div(numerator, denominator)
        
        return alpha.fillna(0)
