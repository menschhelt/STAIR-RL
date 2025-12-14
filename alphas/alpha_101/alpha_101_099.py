import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_099(BaseAlpha):
    """
    alpha101_099: sub(0,mul(1,mul(sub(mul(1.5,twise_a_scale(grouped_demean(grouped_demean(rank(mul(div(sub(sub({disk:close},{disk:low}),sub({disk:high},{disk:close})),sub({disk:high},{disk:low})),{disk:volume})),{disk:industry_group_lv3}),{disk:industry_group_lv3}))),twise_a_scale(grouped_demean(sub(ts_corr({disk:close},rank(ts_mean({disk:amount},20)),5),rank(ts_argmin({disk:close},30))),{disk:industry_group_lv3}))),div({disk:volume},ts_mean({disk:amount},20)))))
    
    매우 복잡한 다중 업종별 표준화와 정규화가 포함된 가격 위치-거래량 및 상관관계 복합 팩터의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_099"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_099 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 복잡한 부분: 다중 업종별 표준화된 가격 위치-거래량
        # 1. Williams %R 계산
        close_low = self.sub(data['close'], data['low'])
        high_close = self.sub(data['high'], data['close'])
        high_low = self.sub(data['high'], data['low'])
        
        williams_r_num = self.sub(close_low, high_close)
        williams_r = self.div(williams_r_num, high_low)
        
        # 2. mul(williams_r, volume)
        wr_volume = self.mul(williams_r, data['volume'])
        
        # 3. rank(wr_volume)
        wr_rank = self.rank(wr_volume)
        
        # 4. 이중 grouped_demean (업종 정보가 없으므로 전체 평균으로 대체)
        first_demean = wr_rank - wr_rank.mean()
        second_demean = first_demean - first_demean.mean()
        
        # 5. twise_a_scale(second_demean) and mul(1.5, ...)
        first_scaled = self.twise_a_scale(second_demean, 1)
        first_part = self.mul(1.5, first_scaled)
        
        # 두 번째 복잡한 부분: 상관관계와 argmin의 차이
        # 6. ts_mean(amount, 20)
        amount_mean = self.ts_mean(data['amount'], 20)
        
        # 7. rank(amount_mean)
        amount_rank = self.rank(amount_mean)
        
        # 8. ts_corr(close, amount_rank, 5)
        close_amount_corr = self.ts_corr(data['close'], amount_rank, 5)
        
        # 9. ts_argmin(close, 30)
        close_argmin = self.ts_argmin(data['close'], 30)
        
        # 10. rank(close_argmin)
        argmin_rank = self.rank(close_argmin)
        
        # 11. sub(close_amount_corr, argmin_rank)
        corr_diff = self.sub(close_amount_corr, argmin_rank)
        
        # 12. grouped_demean(corr_diff, industry_group_lv3) -> 전체 평균으로 대체
        corr_demeaned = corr_diff - corr_diff.mean()
        
        # 13. twise_a_scale(corr_demeaned)
        second_part = self.twise_a_scale(corr_demeaned, 1)
        
        # 세 번째 부분: 거래량 비율
        # 14. div(volume, amount_mean)
        volume_ratio = self.div(data['volume'], amount_mean)
        
        # 최종 계산
        # 15. sub(first_part, second_part)
        main_diff = self.sub(first_part, second_part)
        
        # 16. mul(main_diff, volume_ratio)
        product = self.mul(main_diff, volume_ratio)
        
        # 17. sub(0, product) = -product
        alpha = self.mul(product, -1)
        
        return alpha.fillna(0)
