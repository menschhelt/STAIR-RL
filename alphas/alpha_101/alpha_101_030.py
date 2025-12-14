import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_030(BaseAlpha):
    """
    alpha101_030: add(add(rank(rank(rank(ts_decayed_linear(mul(-1,rank(rank(ts_delta({disk:close},10)))),10)))),rank(mul(-1,ts_delta({disk:close},3)))),sign(twise_a_scale(ts_corr(ts_mean({disk:amount},20),{disk:low},12))))
    
    종가 변화의 다중 랭킹 감쇠, 단기 변화, 거래대금-저가 상관관계의 복합 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_030"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_030 계산
        """
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 복잡한 랭킹 체인
        # 1. ts_delta(close, 10)
        close_delta10 = self.ts_delta(data['close'], 10)
        
        # 2. rank(close_delta10)
        rank1 = self.rank(close_delta10)
        
        # 3. rank(rank1)
        rank2 = self.rank(rank1)
        
        # 4. mul(-1, rank2)
        neg_rank = self.mul(rank2, -1)
        
        # 5. ts_decayed_linear(..., 10)
        decayed = self.ts_decayed_linear(neg_rank, 10)
        
        # 6. rank(decayed)
        rank3 = self.rank(decayed)
        
        # 7. rank(rank3)
        rank4 = self.rank(rank3)
        
        # 8. rank(rank4)
        first_part = self.rank(rank4)
        
        # 두 번째 부분: 단기 변화
        # 9. ts_delta(close, 3)
        close_delta3 = self.ts_delta(data['close'], 3)
        
        # 10. mul(-1, close_delta3)
        neg_delta3 = self.mul(close_delta3, -1)
        
        # 11. rank(neg_delta3)
        second_part = self.rank(neg_delta3)
        
        # 세 번째 부분: 거래대금-저가 상관관계
        # 12. ts_mean(amount, 20)
        amount_mean = self.ts_mean(data['amount'], 20)
        
        # 13. ts_corr(amount_mean, low, 12)
        corr_12 = self.ts_corr(amount_mean, data['low'], 12)
        
        # 14. twise_a_scale(corr_12, ?)
        scaled_corr = self.twise_a_scale(corr_12, 1)  # scale 파라미터 추정
        
        # 15. sign(scaled_corr)
        third_part = self.sign(scaled_corr)
        
        # 16. 모든 부분 합하기
        alpha = self.add(
            self.add(first_part, second_part),
            third_part
        )
        
        return alpha.fillna(0)
