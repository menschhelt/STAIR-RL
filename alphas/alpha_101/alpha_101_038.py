import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_038(BaseAlpha):
    """
    alpha101_038: mul(mul(-1,rank(mul(ts_delta({disk:close},7),sub(1,rank(ts_decayed_linear(div({disk:volume},ts_mean({disk:amount},20)),9)))))),add(1,rank(ts_sum({disk:returns},250))))
    
    7일 가격 변화와 거래량 비율의 감쇠선형 랭킹, 장기 수익률 합계의 복합 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_038"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_038 계산
        """
        # 수익률 계산
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        # amount 계산
        if 'amount' not in data:
            data['amount'] = data['volume'] * data['close']
        
        # 첫 번째 부분: 복잡한 랭킹 조합
        # 1. ts_delta(close, 7)
        close_delta7 = self.ts_delta(data['close'], 7)
        
        # 2. div(volume, ts_mean(amount, 20))
        amount_mean = self.ts_mean(data['amount'], 20)
        volume_ratio = self.div(data['volume'], amount_mean)
        
        # 3. ts_decayed_linear(volume_ratio, 9)
        decayed_ratio = self.ts_decayed_linear(volume_ratio, 9)
        
        # 4. rank(decayed_ratio)
        ratio_rank = self.rank(decayed_ratio)
        
        # 5. sub(1, ratio_rank)
        one_minus_rank = (1 - ratio_rank)
        
        # 6. mul(close_delta7, one_minus_rank)
        product = self.mul(close_delta7, one_minus_rank)
        
        # 7. rank(product)
        product_rank = self.rank(product)
        
        # 8. mul(-1, product_rank)
        neg_rank = self.mul(product_rank, -1)
        
        # 두 번째 부분: 장기 수익률
        # 9. ts_sum(returns, 250)
        returns_sum = self.ts_sum(data['returns'], 250)
        
        # 10. rank(returns_sum)
        returns_rank = self.rank(returns_sum)
        
        # 11. add(1, returns_rank)
        returns_plus1 = (1 + returns_rank)
        
        # 12. 최종 곱하기
        alpha = self.mul(neg_rank, returns_plus1)
        
        return alpha.fillna(0)
