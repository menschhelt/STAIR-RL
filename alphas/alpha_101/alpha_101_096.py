import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_096(BaseAlpha):
    """
    alpha101_096: mul(sub(rank(ts_decayed_linear(ts_delta(grouped_demean(add(mul({disk:low},0.721001),mul({disk:vwap},sub(1,0.721001))),{disk:industry_group_lv2}),3.3705),20.4523)),ts_rank(ts_decayed_linear(ts_rank(ts_corr(ts_rank({disk:low},7.87871),ts_rank(ts_mean({disk:amount},60),17.255),4.97547),18.5925),15.7152),6.71659)),-1)
    
    업종별 표준화된 가중 저가-VWAP 변화의 감쇠선형 랭킹에서 저가-거래대금 시계열 랭킹 상관관계의 복잡한 처리를 뺀 값의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_096"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_096 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        weight = 0.721001
        
        # 첫 번째 부분: 업종별 표준화된 가중 저가-VWAP 변화의 감쇠선형 랭킹
        # 1. 가중 저가-VWAP: low * 0.721001 + vwap * (1-0.721001)
        weighted_low_vwap = self.add(
            self.mul(data['low'], weight),
            self.mul(data['vwap'], 1 - weight)
        )
        
        # 2. grouped_demean(weighted_low_vwap, industry_group_lv2) -> 전체 평균으로 대체
        weighted_demeaned = weighted_low_vwap - weighted_low_vwap.mean()
        
        # 3. ts_delta(weighted_demeaned, 3.3705) -> 반올림하여 3
        delta_window = int(round(3.3705))
        weighted_delta = self.ts_delta(weighted_demeaned, delta_window)
        
        # 4. ts_decayed_linear(weighted_delta, 20.4523) -> 반올림하여 20
        decay_window1 = int(round(20.4523))
        delta_decayed = self.ts_decayed_linear(weighted_delta, decay_window1)
        
        # 5. rank(delta_decayed)
        first_part = self.rank(delta_decayed)
        
        # 두 번째 부분: 저가-거래대금 시계열 랭킹 상관관계의 복잡한 처리
        # 6. ts_rank(low, 7.87871) -> 반올림하여 8
        low_rank_window = int(round(7.87871))
        low_tsrank = self.ts_rank(data['low'], low_rank_window)
        
        # 7. ts_mean(amount, 60)
        amount_mean = self.ts_mean(data['amount'], 60)
        
        # 8. ts_rank(amount_mean, 17.255) -> 반올림하여 17
        amount_rank_window = int(round(17.255))
        amount_tsrank = self.ts_rank(amount_mean, amount_rank_window)
        
        # 9. ts_corr(low_tsrank, amount_tsrank, 4.97547) -> 반올림하여 5
        corr_window = int(round(4.97547))
        low_amount_corr = self.ts_corr(low_tsrank, amount_tsrank, corr_window)
        
        # 10. ts_rank(low_amount_corr, 18.5925) -> 반올림하여 19
        corr_rank_window = int(round(18.5925))
        corr_ranked = self.ts_rank(low_amount_corr, corr_rank_window)
        
        # 11. ts_decayed_linear(corr_ranked, 15.7152) -> 반올림하여 16
        decay_window2 = int(round(15.7152))
        corr_decayed = self.ts_decayed_linear(corr_ranked, decay_window2)
        
        # 12. ts_rank(corr_decayed, 6.71659) -> 반올림하여 7
        final_rank_window = int(round(6.71659))
        second_part = self.ts_rank(corr_decayed, final_rank_window)
        
        # 13. sub(first_part, second_part)
        diff = self.sub(first_part, second_part)
        
        # 14. mul(diff, -1)
        alpha = self.mul(diff, -1)
        
        return alpha.fillna(0)
