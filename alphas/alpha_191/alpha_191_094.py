import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_094(BaseAlpha):
    """
    alpha191_094: ts_std({disk:amount},20)
    
    거래대금의 20일 표준편차 (069와 유사하지만 기간이 다름)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_094"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_094 계산
        """
        # amount 필드 확인 및 생성
        if 'amount' not in data:
            # amount = price * volume으로 근사
            data['amount'] = data['close'] * data['volume']
        
        amount = data['amount']
        
        # 20일 표준편차
        alpha = self.ts_std(amount, 20)
        
        return alpha.fillna(0)
