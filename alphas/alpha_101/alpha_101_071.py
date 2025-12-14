import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_071(BaseAlpha):
    """
    alpha101_071: div(rank(ts_decayed_linear(ts_corr(div(add({disk:high},{disk:low}),2),ts_mean({disk:amount},40),8.93345),10.1519)),rank(ts_decayed_linear(ts_corr(ts_rank({disk:vwap},3.72469),ts_rank({disk:volume},18.5188),6.86671),2.95011)))
    
    중간가-거래대금 상관관계의 감쇠선형 랭킹을 VWAP-거래량 시계열 랭킹 상관관계의 감쇠선형 랭킹으로 나눈 비율
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_071"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_071 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분 (분자): 중간가-거래대금 상관관계의 감쇠선형 랭킹
        # 1. div(add(high, low), 2) - 중간가격
        mid_price = self.div(
            self.add(data['high'], data['low']),
            2
        )
        
        # 2. ts_mean(amount, 40)
        amount_mean = self.ts_mean(data['amount'], 40)
        
        # 3. ts_corr(mid_price, amount_mean, 8.93345) -> 반올림하여 9
        corr_window1 = int(round(8.93345))
        corr1 = self.ts_corr(mid_price, amount_mean, corr_window1)
        
        # 4. ts_decayed_linear(corr1, 10.1519) -> 반올림하여 10
        decay_window1 = int(round(10.1519))
        decayed1 = self.ts_decayed_linear(corr1, decay_window1)
        
        # 5. rank(decayed1)
        numerator = self.rank(decayed1)
        
        # 두 번째 부분 (분모): VWAP-거래량 시계열 랭킹 상관관계의 감쇠선형 랭킹
        # 6. ts_rank(vwap, 3.72469) -> 반올림하여 4
        vwap_rank_window = int(round(3.72469))
        vwap_tsrank = self.ts_rank(data['vwap'], vwap_rank_window)
        
        # 7. ts_rank(volume, 18.5188) -> 반올림하여 19
        volume_rank_window = int(round(18.5188))
        volume_tsrank = self.ts_rank(data['volume'], volume_rank_window)
        
        # 8. ts_corr(vwap_tsrank, volume_tsrank, 6.86671) -> 반올림하여 7
        corr_window2 = int(round(6.86671))
        corr2 = self.ts_corr(vwap_tsrank, volume_tsrank, corr_window2)
        
        # 9. ts_decayed_linear(corr2, 2.95011) -> 반올림하여 3
        decay_window2 = int(round(2.95011))
        decayed2 = self.ts_decayed_linear(corr2, decay_window2)
        
        # 10. rank(decayed2)
        denominator = self.rank(decayed2)
        
        # 11. div(numerator, denominator)
        alpha = self.div(numerator, denominator)
        
        return alpha.fillna(0)
