import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_035(BaseAlpha):
    """
    alpha101_035: add(add(add(add(mul(2.21,rank(ts_corr(sub({disk:close},{disk:open}),delay({disk:volume},1),15))),mul(0.7,rank(sub({disk:open},{disk:close})))),mul(0.73,rank(ts_rank(delay(mul(-1,{disk:returns}),6),5)))),rank(abs(ts_corr({disk:vwap},ts_mean({disk:amount},20),6)))),mul(0.6,rank(mul(sub(div(ts_sum({disk:close},200),200),{disk:open}),sub({disk:close},{disk:open})))))
    
    여러 가격 관계와 거래량, 수익률의 가중 복합 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_035"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_035 계산
        """
        # 수익률 계산
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 2.21 * rank(ts_corr(close-open, delay(volume,1), 15))
        close_open_diff = self.sub(data['close'], data['open'])
        volume_lag1 = self.delay(data['volume'], 1)
        corr_15 = self.ts_corr(close_open_diff, volume_lag1, 15)
        part1 = self.mul(2.21, self.rank(corr_15))
        
        # 두 번째 부분: 0.7 * rank(open - close)
        open_close_diff = self.sub(data['open'], data['close'])
        part2 = self.mul(0.7, self.rank(open_close_diff))
        
        # 세 번째 부분: 0.73 * rank(ts_rank(delay(-returns, 6), 5))
        neg_returns = self.mul(data['returns'], -1)
        returns_lag6 = self.delay(neg_returns, 6)
        returns_tsrank = self.ts_rank(returns_lag6, 5)
        part3 = self.mul(0.73, self.rank(returns_tsrank))
        
        # 네 번째 부분: rank(abs(ts_corr(vwap, ts_mean(amount, 20), 6)))
        amount_mean = self.ts_mean(data['amount'], 20)
        vwap_amount_corr = self.ts_corr(data['vwap'], amount_mean, 6)
        part4 = self.rank(self.abs(vwap_amount_corr))
        
        # 다섯 번째 부분: 0.6 * rank((close_mean_200 - open) * (close - open))
        close_mean_200 = self.div(self.ts_sum(data['close'], 200), 200)
        mean_open_diff = self.sub(close_mean_200, data['open'])
        close_open_product = self.mul(mean_open_diff, close_open_diff)
        part5 = self.mul(0.6, self.rank(close_open_product))
        
        # 모든 부분들을 더하기
        alpha = self.add(
            self.add(
                self.add(
                    self.add(part1, part2),
                    part3
                ),
                part4
            ),
            part5
        )
        
        return alpha.fillna(0)
