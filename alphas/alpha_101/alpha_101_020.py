import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_020(BaseAlpha):
    """
    alpha101_020: condition(lt(add(div(ts_sum({disk:close},8),8),ts_std({disk:close},8)),div(ts_sum({disk:close},2),2)),mul(-1,1),condition(lt(div(ts_sum({disk:close},2),2),sub(div(ts_sum({disk:close},8),8),ts_std({disk:close},8))),1,condition(or_(lt(1,div({disk:volume},ts_mean({disk:amount},20))),eq(div({disk:volume},ts_mean({disk:amount},20)),1)),1,mul(-1,1))))
    
    종가의 8기간 평균과 표준편차, 2기간 평균, 거래량 비율을 이용한 조건부 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_020"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_020 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 1. div(ts_sum(close, 8), 8) - 8기간 평균
        close_mean_8 = self.div(self.ts_sum(data['close'], 8), 8)
        
        # 2. ts_std(close, 8) - 8기간 표준편차
        close_std_8 = self.ts_std(data['close'], 8)
        
        # 3. div(ts_sum(close, 2), 2) - 2기간 평균
        close_mean_2 = self.div(self.ts_sum(data['close'], 2), 2)
        
        # 4. div(volume, ts_mean(amount, 20))
        amount_mean_20 = self.ts_mean(data['amount'], 20)
        volume_ratio = self.div(data['volume'], amount_mean_20)
        
        # 첫 번째 조건: (close_mean_8 + close_std_8) < close_mean_2
        condition1 = self.lt(
            self.add(close_mean_8, close_std_8),
            close_mean_2
        )
        
        # 두 번째 조건: close_mean_2 < (close_mean_8 - close_std_8)
        condition2 = self.lt(
            close_mean_2,
            self.sub(close_mean_8, close_std_8)
        )
        
        # 세 번째 조건: volume_ratio > 1 or volume_ratio == 1
        condition3a = self.lt(1, volume_ratio)
        condition3b = self.eq(volume_ratio, 1)
        condition3 = self.or_(condition3a, condition3b)
        
        # 중첩된 조건문 구현
        inner_condition = self.condition(
            condition3,
            1,
            -1
        )
        
        middle_condition = self.condition(
            condition2,
            1,
            inner_condition
        )
        
        alpha = self.condition(
            condition1,
            -1,
            middle_condition
        )
        
        return alpha.fillna(0)
