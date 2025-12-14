import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_191_029(BaseAlpha):
    """
    alpha191_029: ts_decayed_linear(pow(reg_resi(reg_resi(reg_resi({disk:returns},{disk:mkt},60),{disk:smb},60),{disk:hml},60),2),20)
    
    3 factor model 잔차의 제곱의 20일 decayed linear
    """
    
    # 알파별 처리 설정
    neutralizer_type: str = "factor"
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_191_029"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 191_029 계산
        """
        # returns 필드 확인 및 생성
        if 'returns' not in data:
            data['returns'] = data['close'].pct_change()
        
        returns = data['returns']
        
        # 팩터 필드들 확인 및 기본값 설정
        if 'mkt' not in data:
            data['mkt'] = returns  # 시장 수익률로 대체
        if 'smb' not in data:
            data['smb'] = 0  # SMB 팩터가 없으면 0으로 설정
        if 'hml' not in data:
            data['hml'] = 0  # HML 팩터가 없으면 0으로 설정
        
        mkt = data['mkt']
        smb = data['smb']
        hml = data['hml']
        
        # 3단계 회귀 잔차 계산
        # 1단계: returns vs mkt
        resi1 = self.reg_resi(returns, mkt, 60)
        
        # 2단계: resi1 vs smb
        resi2 = self.reg_resi(resi1, smb, 60)
        
        # 3단계: resi2 vs hml
        resi3 = self.reg_resi(resi2, hml, 60)
        
        # 제곱
        resi_squared = self.pow(resi3, 2)
        
        # 20일 decayed linear
        alpha = self.ts_decayed_linear(resi_squared, 20)
        
        return alpha.fillna(0)
