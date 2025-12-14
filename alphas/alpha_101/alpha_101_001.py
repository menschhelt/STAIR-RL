import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

# 부모 디렉토리를 import 경로에 추가
from alphas.base.base import BaseAlpha

class alpha_101_001(BaseAlpha):
    """
    Alpha 101_001: mul(-1,ts_corr(rank(ts_delta(log({disk:volume}),2)),rank(div(sub({disk:close},{disk:open}),{disk:open})),6))
    
    볼륨 변화와 가격 모멘텀의 음의 상관관계 활용
    """
    # 기본 파라미터 정의
    default_params = {
        "delta_window": 2,
        "corr_window": 6
    }

    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"  # 팩터 중립화
    decay_period: int = 3
    
    
    @property
    def name(self) -> str:
        return "alpha_101_001"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_001 계산
        """
        # 1. log(volume)의 2기간 차분
        log_volume = self.log(data['volume'])
        volume_delta = self.ts_delta(log_volume, self.params["delta_window"])
        volume_rank = self.rank(volume_delta)
        
        # 2. (close - open) / open의 랭킹
        price_change = self.div(
            self.sub(data['close'], data['open']),
            data['open']
        )
        price_rank = self.rank(price_change)
        
        # 3. 6기간 상관계수 * -1
        corr = self.ts_corr(volume_rank, price_rank, self.params["corr_window"])
        # print(f"corr: {corr}")
        alpha = self.mul(corr, -1)
        
        return alpha.fillna(0)
