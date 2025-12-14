"""
Alpha 191_002: True Range Momentum Accumulator

DSL: ts_sum(condition(eq({disk:close},delay({disk:close},1)),0,sub({disk:close},condition(gt({disk:close},delay({disk:close},1)),min({disk:low},delay({disk:close},1)),max({disk:high},delay({disk:close},1))))),6)

This alpha accumulates directional price movements over 6 days, measuring the distance between
close price and the true range boundary. It captures momentum by summing directional price gaps
relative to the previous day's close and current day's range.

Strategy Type: Momentum / Directional Movement
Key Features: True range analysis, 6-day momentum accumulation, directional gap measurement
Time Horizon: 6-day cumulative momentum
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict

from alphas.base.base import BaseAlpha


class alpha_191_002(BaseAlpha):
    """
    Alpha 191_002: True Range Momentum Accumulator

    Accumulates directional price movements by measuring the gap between close price
    and true range boundary (min or max based on direction) over a 6-day window.
    """

    neutralizer_type: str = "factor"
    decay_period: int = 5

    @property
    def name(self) -> str:
        return "alpha_191_002"

    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Calculate Alpha 191_002

        Formula: ts_sum(condition(close == prev_close, 0,
                       close - condition(close > prev_close,
                                       min(low, prev_close),
                                       max(high, prev_close))), 6)

        Steps:
        1. Check if close equals previous close
        2. If close > prev_close, use min(low, prev_close) as boundary
        3. If close <= prev_close, use max(high, prev_close) as boundary
        4. Calculate: close - boundary (0 if close == prev_close)
        5. Sum over 6-day window
        """
        # Step 1: Get previous close
        prev_close = self.delay(data['close'], 1)

        # Step 2: Check conditions
        close_eq_prev = self.eq(data['close'], prev_close)
        close_gt_prev = self.gt(data['close'], prev_close)

        # Step 3: Calculate boundaries
        min_boundary = self.min(data['low'], prev_close)
        max_boundary = self.max(data['high'], prev_close)

        # Step 4: Inner condition - select boundary based on direction
        boundary = self.condition(
            close_gt_prev,
            min_boundary,
            max_boundary
        )

        # Step 5: Calculate directional gap
        directional_gap = self.sub(data['close'], boundary)

        # Step 6: Set to 0 if close == prev_close
        conditional_result = self.condition(
            close_eq_prev,
            0,  # scalar 0 대신
            directional_gap
        )

        # Step 7: 6-day rolling sum
        alpha = self.ts_sum(conditional_result, 6)

        return alpha
