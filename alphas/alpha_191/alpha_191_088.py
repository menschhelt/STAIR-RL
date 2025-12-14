import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_088(BaseAlpha):
    """
    alpha191_088: mul(2,sub(sma({disk:close},13,2),sub(sma({disk:close},27,2),sma(sub(sma({disk:close},13,2),sma({disk:close},27,2)),10,2))))
    
    2 * (SMA(13) - (SMA(27) - SMA(SMA(13) - SMA(27), 10))) = MACD 스타일 지표
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_088"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_088 계산
        """
        close = data['close']
        
        # SMA 계산
        sma_13 = self.sma(close, 13, 2)
        sma_27 = self.sma(close, 27, 2)
        
        # MACD 라인
        macd_line = self.sub(sma_13, sma_27)
        
        # 시그널 라인
        signal_line = self.sma(macd_line, 10, 2)
        
        # MACD 히스토그램
        macd_histogram = self.sub(sma_13, self.sub(sma_27, signal_line))
        
        # 2배
        alpha = self.mul(2, macd_histogram)
        
        return alpha.fillna(0)
