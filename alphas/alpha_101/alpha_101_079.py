import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_079(BaseAlpha):
    """
    alpha101_079: mul(pow(rank(sign(ts_delta(grouped_demean(add(mul({disk:open},0.868128),mul({disk:high},sub(1,0.868128))),{disk:industry_group_lv2}),4.04545))),ts_rank(ts_corr({disk:high},ts_mean({disk:amount},10),5.11456),5.53756)),-1)
    
    업종별 표준화된 가중 시가-고가 변화 방향성 랭킹과 고가-거래대금 상관관계 시계열 랭킹의 거듭제곱의 음수
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_079"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_079 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        weight = 0.868128
        
        # 첫 번째 부분: 업종별 표준화된 가중 시가-고가 변화 방향성
        # 1. 가중 시가-고가: open * 0.868128 + high * (1-0.868128)
        weighted_open_high = self.add(
            self.mul(data['open'], weight),
            self.mul(data['high'], 1 - weight)
        )
        
        # 2. grouped_demean(weighted_open_high, industry_group_lv2) -> 전체 평균으로 대체
        weighted_demeaned = weighted_open_high - weighted_open_high.mean()
        
        # 3. ts_delta(weighted_demeaned, 4.04545) -> 반올림하여 4
        delta_window = int(round(4.04545))
        weighted_delta = self.ts_delta(weighted_demeaned, delta_window)
        
        # 4. sign(weighted_delta)
        delta_sign = self.sign(weighted_delta)
        
        # 5. rank(delta_sign)
        base_rank = self.rank(delta_sign)
        
        # 두 번째 부분: 고가-거래대금 상관관계 시계열 랭킹
        # 6. ts_mean(amount, 10)
        amount_mean = self.ts_mean(data['amount'], 10)
        
        # 7. ts_corr(high, amount_mean, 5.11456) -> 반올림하여 5
        corr_window = int(round(5.11456))
        corr_result = self.ts_corr(data['high'], amount_mean, corr_window)
        
        # 8. ts_rank(corr_result, 5.53756) -> 반올림하여 6
        rank_window = int(round(5.53756))
        power_rank = self.ts_rank(corr_result, rank_window)
        
        # 9. pow(base_rank, power_rank)
        powered = self.pow(base_rank, power_rank)
        
        # 10. mul(powered, -1)
        alpha = self.mul(powered, -1)
        
        return alpha.fillna(0)
