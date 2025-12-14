import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_074(BaseAlpha):
    """
    alpha191_074: div(countcond(gt({disk:close},and_({disk:open},lt({disk:benchmarkindex_close},{disk:benchmarkindex_open}))),50),countcond(lt({disk:benchmarkindex_close},{disk:benchmarkindex_open}),50))
    
    벤치마크 하락 시 개별 주식이 상승한 비율 (벤치마크 데이터가 없으면 단순화)
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_074"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_074 계산
        """
        close = data['close']
        open_price = data['open']
        
        # 벤치마크 인덱스 필드가 없으면 시장 평균으로 대체
        if 'benchmarkindex_close' not in data:
            # 시장 평균으로 근사 (rolling mean 사용)
            market_close = close.rolling(50).mean()
            market_open = open_price.rolling(50).mean()
        else:
            market_close = data['benchmarkindex_close']
            market_open = data['benchmarkindex_open']
        
        # 벤치마크 하락 조건
        market_down = self.lt(market_close, market_open)
        
        # 개별 주식 상승 조건
        stock_up = self.gt(close, open_price)
        
        # 벤치마크 하락 & 개별 주식 상승
        condition1 = self.and_(stock_up, market_down)
        
        # 50일간 카운트
        count1 = self.ts_sum(condition1.astype(float), 50)
        count2 = self.ts_sum(market_down.astype(float), 50)
        
        # 비율
        alpha = self.div(count1, count2)
        
        return alpha.fillna(0)
