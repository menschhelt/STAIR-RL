import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_016(BaseAlpha):
    """
    alpha101_016: mul(mul(mul(-1,rank(ts_rank({disk:close},10))),rank(ts_delta(ts_delta({disk:close},1),1))),rank(ts_rank(div({disk:volume},ts_mean({disk:amount},20)),5)))
    
    종가 10기간 랭킹, 종가 2차 차분, 거래량 대비 거래대금 비율의 복합 팩터
    """
    # 기본 파라미터 정의
    default_params = {
        "ts_rank_window1": 10,
        "ts_rank_window2": 5,
        "amount_mean_window": 20
    }

    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_016"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_016 계산
        """
        # 1. rank(ts_rank(close, 10))
        close_ts_rank = self.ts_rank(data['close'], self.params["ts_rank_window1"])
        close_rank = self.rank(close_ts_rank)
        neg_close_rank = self.mul(close_rank, -1)
        
        # 2. rank(ts_delta(ts_delta(close, 1), 1))
        close_delta1 = self.ts_delta(data['close'], 1)
        close_delta2 = self.ts_delta(close_delta1, 1)
        delta_rank = self.rank(close_delta2)
        
        # 3. rank(ts_rank(div(volume, ts_mean(amount, 20)), 5))
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        amount_mean = self.ts_mean(data['amount'], self.params["amount_mean_window"])
        volume_ratio = self.div(data['volume'], amount_mean)
        volume_ts_rank = self.ts_rank(volume_ratio, self.params["ts_rank_window2"])
        volume_rank = self.rank(volume_ts_rank)
        
        # 4. 모든 것들을 곱하기
        alpha = self.mul(
            self.mul(neg_close_rank, delta_rank),
            volume_rank
        )
        
        return alpha.fillna(0)
