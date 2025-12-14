import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_078(BaseAlpha):
    """
    alpha101_078: lt(rank(ts_delta(grouped_demean(add(mul({disk:close},0.60733),mul({disk:open},sub(1,0.60733))),{disk:industry_group_lv1}),1.23438)),rank(ts_corr(ts_rank({disk:vwap},3.60973),ts_rank(ts_mean({disk:amount},150),9.18637),14.6644)))
    
    업종별 표준화된 가중 종가-시가 변화 랭킹과 VWAP-거래대금 시계열 랭킹 상관관계 랭킹 비교
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_078"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_078 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        weight = 0.60733
        
        # 첫 번째 부분: 업종별 표준화된 가중 종가-시가 변화
        # 1. 가중 종가-시가: close * 0.60733 + open * (1-0.60733)
        weighted_close_open = self.add(
            self.mul(data['close'], weight),
            self.mul(data['open'], 1 - weight)
        )
        
        # 2. grouped_demean(weighted_close_open, industry_group_lv1) -> 전체 평균으로 대체
        weighted_demeaned = weighted_close_open - weighted_close_open.mean()
        
        # 3. ts_delta(weighted_demeaned, 1.23438) -> 반올림하여 1
        delta_window = int(round(1.23438))
        weighted_delta = self.ts_delta(weighted_demeaned, delta_window)
        
        # 4. rank(weighted_delta)
        first_rank = self.rank(weighted_delta)
        
        # 두 번째 부분: VWAP-거래대금 시계열 랭킹 상관관계
        # 5. ts_rank(vwap, 3.60973) -> 반올림하여 4
        vwap_rank_window = int(round(3.60973))
        vwap_tsrank = self.ts_rank(data['vwap'], vwap_rank_window)
        
        # 6. ts_mean(amount, 150)
        amount_mean = self.ts_mean(data['amount'], 150)
        
        # 7. ts_rank(amount_mean, 9.18637) -> 반올림하여 9
        amount_rank_window = int(round(9.18637))
        amount_tsrank = self.ts_rank(amount_mean, amount_rank_window)
        
        # 8. ts_corr(vwap_tsrank, amount_tsrank, 14.6644) -> 반올림하여 15
        corr_window = int(round(14.6644))
        corr_result = self.ts_corr(vwap_tsrank, amount_tsrank, corr_window)
        
        # 9. rank(corr_result)
        second_rank = self.rank(corr_result)
        
        # 10. lt(first_rank, second_rank)
        alpha = self.lt(first_rank, second_rank)
        
        # 불린을 숫자로 변환
        alpha = alpha.astype(float)
        
        return alpha.fillna(0)
