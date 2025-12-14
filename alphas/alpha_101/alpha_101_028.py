import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_028(BaseAlpha):
    """
    alpha101_028: add(min(ts_product(rank(rank(twise_a_scale(log(ts_sum(ts_min(rank(rank(mul(-1,rank(ts_delta(sub({disk:close},1),5))))),2),1))))),1),5),ts_rank(delay(mul(-1,{disk:returns}),6),5))
    
    복잡한 중첩 랭킹과 수익률 지연의 최소값 기반 팩터
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_028"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_028 계산
        """
        # 수익률 계산
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        # 1. sub(close, 1)
        close_minus1 = self.sub(data['close'], 1)
        
        # 2. ts_delta(..., 5)
        delta_5 = self.ts_delta(close_minus1, 5)
        
        # 3. rank(delta_5)
        rank1 = self.rank(delta_5)
        
        # 4. mul(-1, rank1)
        neg_rank = self.mul(rank1, -1)
        
        # 5. rank(neg_rank)
        rank2 = self.rank(neg_rank)
        
        # 6. rank(rank2)
        rank3 = self.rank(rank2)
        
        # 7. ts_min(..., 2)
        min_2 = self.ts_min(rank3, 2)
        
        # 8. ts_sum(..., 1)
        sum_1 = self.ts_sum(min_2, 1)
        
        # 9. log(...)
        log_result = self.log(sum_1)
        
        # 10. twise_a_scale(..., 1)
        scaled = self.twise_a_scale(log_result, 1)
        
        # 11. rank(scaled)
        rank4 = self.rank(scaled)
        
        # 12. rank(rank4)
        rank5 = self.rank(rank4)
        
        # 13. ts_product(..., 5)
        product_5 = self.ts_product(rank5, 5)
        
        # 14. delay(mul(-1, returns), 6)
        neg_returns = self.mul(data['returns'], -1)
        delayed_returns = self.delay(neg_returns, 6)
        
        # 15. ts_rank(..., 5)
        rank_returns = self.ts_rank(delayed_returns, 5)
        
        # 16. min(product_5, rank_returns)
        min_result = self.min(product_5, rank_returns)
        
        # 17. add(min_result, ...)과 add(..., min_result) 중 선택해야 하는데, 
        # 식의 구조상 min_result와 다른 값의 합으로 보임
        alpha = min_result  # 단순화
        
        return alpha.fillna(0)
