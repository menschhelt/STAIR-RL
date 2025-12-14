import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_070(BaseAlpha):
    """
    alpha101_070: max(ts_rank(ts_decayed_linear(ts_corr(ts_rank({disk:close},3.43976),ts_rank(ts_mean({disk:amount},180),12.0647),18.0175),4.20501),15.6948),ts_rank(ts_decayed_linear(pow(rank(sub(add({disk:low},{disk:open}),add({disk:vwap},{disk:vwap}))),2),16.4662),4.4388))
    
    종가-거래대금 상관관계의 복잡한 감쇠선형 랭킹과 가격 차이의 거듭제곱 감쇠선형 랭킹의 최대값
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_070"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_070 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 종가-거래대금 상관관계의 복잡한 감쇠선형 랭킹
        # 1. ts_rank(close, 3.43976) -> 반올림하여 3
        close_rank_window = int(round(3.43976))
        close_tsrank = self.ts_rank(data['close'], close_rank_window)
        
        # 2. ts_mean(amount, 180)
        amount_mean = self.ts_mean(data['amount'], 180)
        
        # 3. ts_rank(amount_mean, 12.0647) -> 반올림하여 12
        amount_rank_window = int(round(12.0647))
        amount_tsrank = self.ts_rank(amount_mean, amount_rank_window)
        
        # 4. ts_corr(close_tsrank, amount_tsrank, 18.0175) -> 반올림하여 18
        corr_window = int(round(18.0175))
        corr_result = self.ts_corr(close_tsrank, amount_tsrank, corr_window)
        
        # 5. ts_decayed_linear(corr_result, 4.20501) -> 반올림하여 4
        decay_window1 = int(round(4.20501))
        corr_decayed = self.ts_decayed_linear(corr_result, decay_window1)
        
        # 6. ts_rank(corr_decayed, 15.6948) -> 반올림하여 16
        corr_final_window = int(round(15.6948))
        first_part = self.ts_rank(corr_decayed, corr_final_window)
        
        # 두 번째 부분: 가격 차이의 거듭제곱 감쇠선형 랭킹
        # 7. add(low, open)
        low_open_sum = self.add(data['low'], data['open'])
        
        # 8. add(vwap, vwap) = 2 * vwap
        vwap_double = self.add(data['vwap'], data['vwap'])
        
        # 9. sub(low_open_sum, vwap_double)
        price_diff = self.sub(low_open_sum, vwap_double)
        
        # 10. rank(price_diff)
        price_rank = self.rank(price_diff)
        
        # 11. pow(price_rank, 2)
        price_squared = self.pow(price_rank, 2)
        
        # 12. ts_decayed_linear(price_squared, 16.4662) -> 반올림하여 16
        decay_window2 = int(round(16.4662))
        price_decayed = self.ts_decayed_linear(price_squared, decay_window2)
        
        # 13. ts_rank(price_decayed, 4.4388) -> 반올림하여 4
        price_final_window = int(round(4.4388))
        second_part = self.ts_rank(price_decayed, price_final_window)
        
        # 14. max(first_part, second_part)
        alpha = self.max(first_part, second_part)
        
        return alpha.fillna(0)
