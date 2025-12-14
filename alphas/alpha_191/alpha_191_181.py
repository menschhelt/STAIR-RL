import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_181(BaseAlpha):
    """
    alpha191_181: div(countcond(or_(and_(gt({disk:close},{disk:open}),gt({disk:benchmarkindex_close},{disk:benchmarkindex_open})),and_(lt({disk:close},{disk:open}),lt({disk:benchmarkindex_close},{disk:benchmarkindex_open}))),20),20)
    
    자산과 벤치마크의 동조화 비율 (벤치마크 없이 단순화)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_181"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_181 계산 (벤치마크가 없는 경우 단순화)
        """
        close = data['close']
        open_price = data['open']
        
        # 벤치마크가 없으므로 시장을 자기 자신의 이동평균으로 근사
        market_close = self.ts_mean(close, 5)  # 5일 평균을 시장으로 근사
        market_open = self.ts_mean(open_price, 5)
        
        # 자산 상승/하락
        asset_up = self.gt(close, open_price)
        asset_down = self.lt(close, open_price)
        
        # 시장 상승/하락
        market_up = self.gt(market_close, market_open)
        market_down = self.lt(market_close, market_open)
        
        # 동조화 조건
        both_up = self.and_(asset_up, market_up)
        both_down = self.and_(asset_down, market_down)
        synchronized = self.or_(both_up, both_down)
        
        # 20일간 동조화 일수 계산 (countcond 근사)
        sync_count = self.ts_sum(self.condition(synchronized, 1, 0), 20)
        
        # 비율 계산
        alpha = self.div(sync_count, 20)
        
        return alpha.fillna(0.5)
