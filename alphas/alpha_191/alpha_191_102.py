import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_102(BaseAlpha):
    """
    alpha191_102: mul(div(sub(20,highday({disk:low},20)),20),100)
    
    (20 - 20일간 저가 최고점 위치) / 20 * 100 = 저가 고점으로부터의 시간 비율
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_102"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_102 계산
        """
        low = data['low']
        
        # highday(low, 20) = 20일간 저가가 최고였던 날의 위치
        # ts_argmax를 사용해서 구현
        low_highday = self.ts_argmax(low, 20)
        
        # (20 - highday) / 20 * 100
        numerator = (20 - low_highday)
        ratio = self.div(numerator, 20)
        alpha = self.mul(ratio, 100)
        
        return alpha.fillna(0)
