"""
Alpha 191_001: Williams %R Momentum Reversal

DSL: mul(-1,ts_delta(div(sub(sub({disk:close},{disk:low}),sub({disk:high},{disk:close})),sub({disk:high},{disk:low})),1))

This alpha captures the daily change in a Williams %R-like indicator, measuring price position
within the high-low range. It identifies momentum reversals in intraday price action.

Strategy Type: Mean Reversion / Momentum Reversal
Key Features: Price range analysis, Williams %R derivative, short-term reversal signals
Time Horizon: 1-day momentum changes
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict

from alphas.base.base import BaseAlpha


class alpha_191_001(BaseAlpha):
    """
    Alpha 191_001: Williams %R Momentum Reversal

    Captures daily momentum changes in price position within the high-low range.
    Based on the derivative of Williams %R indicator.
    """

    neutralizer_type: str = "mean"
    decay_period: int = 3

    @property
    def name(self) -> str:
        return "alpha_191_001"

    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Calculate Alpha 191_001

        Formula: -1 * delta((2*close - high - low) / (high - low), 1)

        Steps:
        1. Calculate numerator: (close - low) - (high - close) = 2*close - high - low
        2. Calculate denominator: high - low
        3. Compute ratio (Williams %R component)
        4. Take 1-day difference
        5. Multiply by -1
        """
        # Step 1: Numerator = 2*close - high - low
        numerator = self.sub(
            self.sub(
                data['close'],
                data['low']
            ),
            self.sub(
                data['high'],
                data['close']
            )
        )

        # Step 2: Denominator = high - low
        denominator = self.sub(data['high'], data['low'])

        # Step 3: Williams %R component
        williams_r_component = self.div(numerator, denominator)

        # Step 4: 1-day delta
        delta_williams = self.ts_delta(williams_r_component, 1)

        # Step 5: Multiply by -1
        alpha = self.mul(-1, delta_williams)

        return alpha
