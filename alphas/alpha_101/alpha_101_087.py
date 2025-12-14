import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_087(BaseAlpha):
    """
    alpha101_087: min(rank(ts_decayed_linear(sub(add(rank({disk:open}),rank({disk:low})),add(rank({disk:high}),rank({disk:close}))),8.06882)),ts_rank(ts_decayed_linear(ts_corr(ts_rank({disk:close},8.44728),ts_rank(ts_mean({disk:amount},60),20.6966),8.01266),6.65053),2.61957))
    
    가격 랭킹 합계 차이의 감쇠선형 랭킹과 종가-거래대금 시계열 랭킹 상관관계의 복잡한 처리의 최소값
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_087"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_087 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 가격 랭킹 합계 차이의 감쇠선형 랭킹
        # 1. rank(open)
        open_rank = self.rank(data['open'])
        
        # 2. rank(low)
        low_rank = self.rank(data['low'])
        
        # 3. rank(high)
        high_rank = self.rank(data['high'])
        
        # 4. rank(close)
        close_rank = self.rank(data['close'])
        
        # 5. add(open_rank, low_rank)
        open_low_sum = self.add(open_rank, low_rank)
        
        # 6. add(high_rank, close_rank)
        high_close_sum = self.add(high_rank, close_rank)
        
        # 7. sub(open_low_sum, high_close_sum)
        rank_diff = self.sub(open_low_sum, high_close_sum)
        
        # 8. ts_decayed_linear(rank_diff, 8.06882) -> 반올림하여 8
        decay_window1 = int(round(8.06882))
        diff_decayed = self.ts_decayed_linear(rank_diff, decay_window1)
        
        # 9. rank(diff_decayed)
        first_part = self.rank(diff_decayed)
        
        # 두 번째 부분: 종가-거래대금 시계열 랭킹 상관관계의 복잡한 처리
        # 10. ts_rank(close, 8.44728) -> 반올림하여 8
        close_rank_window = int(round(8.44728))
        close_tsrank = self.ts_rank(data['close'], close_rank_window)
        
        # 11. ts_mean(amount, 60)
        amount_mean = self.ts_mean(data['amount'], 60)
        
        # 12. ts_rank(amount_mean, 20.6966) -> 반올림하여 21
        amount_rank_window = int(round(20.6966))
        amount_tsrank = self.ts_rank(amount_mean, amount_rank_window)
        
        # 13. ts_corr(close_tsrank, amount_tsrank, 8.01266) -> 반올림하여 8
        corr_window = int(round(8.01266))
        corr_result = self.ts_corr(close_tsrank, amount_tsrank, corr_window)
        
        # 14. ts_decayed_linear(corr_result, 6.65053) -> 반올림하여 7
        decay_window2 = int(round(6.65053))
        corr_decayed = self.ts_decayed_linear(corr_result, decay_window2)
        
        # 15. ts_rank(corr_decayed, 2.61957) -> 반올림하여 3
        final_rank_window = int(round(2.61957))
        second_part = self.ts_rank(corr_decayed, final_rank_window)
        
        # 16. min(first_part, second_part)
        alpha = self.min(first_part, second_part)
        
        return alpha.fillna(0)
