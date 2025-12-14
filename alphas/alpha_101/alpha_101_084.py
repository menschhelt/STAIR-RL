import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_084(BaseAlpha):
    """
    alpha101_084: pow(rank(ts_corr(add(mul({disk:high},0.876703),mul({disk:close},sub(1,0.876703))),ts_mean({disk:amount},30),9.61331)),rank(ts_corr(ts_rank(div(add({disk:high},{disk:low}),2),3.70596),ts_rank({disk:volume},10.1595),7.11408)))
    
    가중 고가-종가와 거래대금 상관관계 랭킹을 중간가-거래량 시계열 랭킹 상관관계 랭킹으로 거듭제곱
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_084"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_084 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        weight = 0.876703
        
        # 밑: 가중 고가-종가와 거래대금 상관관계 랭킹
        # 1. 가중 고가-종가: high * 0.876703 + close * (1-0.876703)
        weighted_high_close = self.add(
            self.mul(data['high'], weight),
            self.mul(data['close'], 1 - weight)
        )
        
        # 2. ts_mean(amount, 30)
        amount_mean = self.ts_mean(data['amount'], 30)
        
        # 3. ts_corr(weighted_high_close, amount_mean, 9.61331) -> 반올림하여 10
        corr_window1 = int(round(9.61331))
        first_corr = self.ts_corr(weighted_high_close, amount_mean, corr_window1)
        
        # 4. rank(first_corr)
        base = self.rank(first_corr)
        
        # 지수: 중간가-거래량 시계열 랭킹 상관관계 랭킹
        # 5. div(add(high, low), 2) - 중간가격
        mid_price = self.div(
            self.add(data['high'], data['low']),
            2
        )
        
        # 6. ts_rank(mid_price, 3.70596) -> 반올림하여 4
        mid_rank_window = int(round(3.70596))
        mid_tsrank = self.ts_rank(mid_price, mid_rank_window)
        
        # 7. ts_rank(volume, 10.1595) -> 반올림하여 10
        vol_rank_window = int(round(10.1595))
        vol_tsrank = self.ts_rank(data['volume'], vol_rank_window)
        
        # 8. ts_corr(mid_tsrank, vol_tsrank, 7.11408) -> 반올림하여 7
        corr_window2 = int(round(7.11408))
        second_corr = self.ts_corr(mid_tsrank, vol_tsrank, corr_window2)
        
        # 9. rank(second_corr)
        power = self.rank(second_corr)
        
        # 10. pow(base, power)
        alpha = self.pow(base, power)
        
        return alpha.fillna(0)
