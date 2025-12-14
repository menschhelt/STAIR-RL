"""
Alpha 191_003: Volatility-Volume Regime Filter

DSL: condition(lt(add(div(ts_sum({disk:close},8),8),ts_std({disk:close},8)),div(ts_sum({disk:close},2),2)),mul(-1,1),condition(lt(div(ts_sum({disk:close},2),2),sub(div(ts_sum({disk:close},8),8),ts_std({disk:close},8))),1,condition(or_(lt(1,div({disk:volume},ts_mean({disk:volume},20))),eq(div({disk:volume},ts_mean({disk:volume},20)),1)),1,mul(-1,1))))

This alpha implements a three-tier regime classification system based on price volatility bands
and volume conditions. It identifies market regimes by comparing short-term (2-day) and medium-term
(8-day) moving averages against volatility-adjusted bands, with volume confirmation.

Strategy Type: Regime Detection / Volatility-based Classification
Key Features: Nested conditional logic, Bollinger-band-like volatility bands, volume ratio filter
Time Horizon: Short-term (2-8 day) regime identification
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict

from alphas.base.base import BaseAlpha


class alpha_191_003(BaseAlpha):
    """
    Alpha 191_003: Volatility-Volume Regime Filter

    Three-tier market regime classifier using volatility bands and volume conditions.
    Returns -1 (bearish), 0 (neutral), or 1 (bullish) based on price-volatility relationship.
    """

    neutralizer_type: str = "mean"
    decay_period: int = 4

    @property
    def name(self) -> str:
        return "alpha_191_003"

    def calculate(self, data: Dict[str, pd.DataFrame], pair: str = None) -> pd.DataFrame:
        """
        Calculate Alpha 191_003

        Formula: Nested conditions based on:
        1. If (MA8 + STD8) < MA2 → -1 (bearish breakout above upper band)
        2. Elif MA2 < (MA8 - STD8) → 1 (bullish breakout below lower band)
        3. Elif volume_ratio >= 1 → 1 (high volume)
        4. Else → -1 (low volume)

        Steps:
        1. Calculate 8-day and 2-day moving averages
        2. Calculate 8-day standard deviation
        3. Calculate 20-day volume moving average
        4. Apply nested conditional logic
        """
        # Step 1: Calculate moving averages
        ma_8 = self.ts_mean(data['close'], 8)
        ma_2 = self.ts_mean(data['close'], 2)

        # Step 2: Calculate volatility
        std_8 = self.ts_std(data['close'], 8)

        # Step 3: Calculate volume ratio
        vol_ma_20 = self.ts_mean(data['volume'], 20)
        vol_ratio = self.div(data['volume'], vol_ma_20)

        # Step 4: Calculate volatility bands
        upper_band = self.add(ma_8, std_8)
        lower_band = self.sub(ma_8, std_8)

        # Step 5: Define conditions
        # Condition 1: (MA8 + STD8) < MA2 (price breaks above upper band)
        cond1 = self.lt(upper_band, ma_2)

        # Condition 2: MA2 < (MA8 - STD8) (price breaks below lower band)
        cond2 = self.lt(ma_2, lower_band)

        # Condition 3: volume_ratio >= 1 (high volume)
        vol_gt_1 = self.lt(1, vol_ratio)  # 1 < vol_ratio
        vol_eq_1 = self.eq(vol_ratio, 1)
        cond3 = self.or_(vol_gt_1, vol_eq_1)

        # Step 6: Apply nested conditional logic
        # Inner condition (for else branch of cond2)
        inner_result = self.condition(cond3, 1, -1)

        # Middle condition (for else branch of cond1)
        middle_result = self.condition(cond2, 1, inner_result)

        # Outer condition (main)
        alpha = self.condition(cond1, -1, middle_result)

        return alpha
