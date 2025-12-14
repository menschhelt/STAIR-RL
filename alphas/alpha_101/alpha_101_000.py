import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import sys
import os

from alphas.base.base import BaseAlpha

class alpha_101_000(BaseAlpha):
    """
    alpha101_000: sub(rank(ts_argmax(pow(condition(lt({disk:returns},0),ts_std({disk:returns},20),{disk:close}),2.),5)),0.5)
    
    수익률 조건부 변동성 vs 가격 비교의 argmax 랭킹
    """

    # 기본 파라미터 정의
    default_params = {
        "std_window": 20,
        "argmax_window": 5
    }

    # 알파별 처리 설정
    neutralizer_type: str = "factor"  # 평균 중립화
    decay_period: int = 3
    
    @property
    def name(self) -> str:
        return "alpha_101_000"
    
    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Alpha 101_000 계산
        """
        import logging
        logger = logging.getLogger(__name__)

        # 수익률 계산
        logger.debug(f"[alpha_101_000] Step 1: Calculating returns")
        if 'returns' not in data:
            returns = data['close'].pct_change()
        else:
            returns = data['returns']

        # 1. condition(lt(returns, 0), ts_std(returns, 20), close)
        logger.debug(f"[alpha_101_000] Step 2: Calculating ts_std (window={self.params['std_window']})")
        returns_std = self.ts_std(returns, self.params["std_window"])

        logger.debug(f"[alpha_101_000] Step 3: Comparing returns < 0")
        returns_negative = returns < 0

        logger.debug(f"[alpha_101_000] Step 4: Applying condition")
        condition_result = self.condition(
            returns_negative,
            returns_std,
            data['close']
        )

        # 2. pow(..., 2)
        logger.debug(f"[alpha_101_000] Step 5: Calculating pow(..., 2)")
        powered = self.pow(condition_result, 2)

        # 3. ts_argmax(..., 5)
        logger.debug(f"[alpha_101_000] Step 6: Calculating ts_argmax (window={self.params['argmax_window']})")
        argmax = self.ts_argmax(powered, self.params["argmax_window"])

        # 4. rank(...)
        logger.debug(f"[alpha_101_000] Step 7: Calculating rank")
        ranked = self.rank(argmax)

        # 5. sub(..., 0.5)
        logger.debug(f"[alpha_101_000] Step 8: Subtracting 0.5")
        alpha = ranked - 0.5

        logger.debug(f"[alpha_101_000] Step 9: Filling NaN values")
        result = alpha.fillna(0)

        logger.debug(f"[alpha_101_000] ✅ Completed! Result shape: {result.shape}")
        return result
