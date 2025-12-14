import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

# 부모 디렉토리를 import 경로에 추가
from alphas.base.base import BaseAlpha

class alpha_101_006(BaseAlpha):
    """
    Alpha 101_006: condition(lt(ts_mean({disk:amount},20),{disk:volume}),mul(mul(-1,ts_rank(abs(ts_delta({disk:close},7)),60)),sign(ts_delta({disk:close},7))),mul(-1,1))
    
    거래량 기준 조건부 가격 변화 신호
    """
    neutralizer_type: str = "mean"  # 평균 중립화

    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_006"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_006 계산
        """
        # amount가 없으면 volume * close로 근사
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 1. ts_mean(amount, 20) < volume 조건
        amount_mean_20 = self.ts_mean(data['amount'], 20)
        condition_check = self.lt(amount_mean_20, data['volume'])

        # 2. 조건이 참일 때의 값
        # ts_delta(close, 7)
        close_delta_7 = self.ts_delta(data['close'], 7)

        # abs(ts_delta(close, 7))
        abs_close_delta = self.abs(close_delta_7)

        # ts_rank(abs_close_delta, 60)
        ts_rank_60 = self.ts_rank(abs_close_delta, 60)
        
        # -1 * ts_rank_60
        neg_ts_rank = ts_rank_60 * -1  # 직접 스칼라 곱셈
        
        # sign(ts_delta(close, 7))
        sign_delta = self.sign(close_delta_7)
        
        # (-1 * ts_rank_60) * sign_delta
        true_value = self.mul(neg_ts_rank, sign_delta)
        
        # 3. 조건이 거짓일 때의 값: -1
        false_value = -1  # 스칼라로 변경
        
        # 4. 최종 조건문
        alpha = self.condition(condition_check, true_value, false_value)
        
        return alpha.fillna(0)
