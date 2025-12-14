"""
Position Sizer - NAV-based position sizing with leverage management.

Handles the conversion from raw actor outputs [-1, +1] to actual
position values in USDT, respecting leverage constraints.

Key features:
- NAV-based sizing: positions scaled by current portfolio NAV
- Leverage scaling: ensures total exposure doesn't exceed target leverage
- Long-short support: positive weights = long, negative = short
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PositionInfo:
    """Container for position sizing results."""
    final_weights: np.ndarray     # Scaled weights [-1, 1]^N
    position_values: np.ndarray   # USDT values per asset
    position_sizes: np.ndarray    # Contract quantities
    gross_exposure: float         # Sum of absolute weights
    net_exposure: float           # Sum of weights (long - short)
    leverage_ratio: float         # gross_exposure / target_leverage
    cash_ratio: float             # Available cash / NAV


class PositionSizer:
    """
    NAV-based position sizing with leverage management.

    Converts raw actor outputs in [-1, +1] range to actual position
    values, respecting the target leverage constraint.

    Example:
        sizer = PositionSizer(target_leverage=2.0)

        # Raw actor output (20 assets)
        w_raw = np.array([0.8, -0.5, 0.3, ...])  # From Tanh activation

        # Current portfolio state
        nav = 100_000  # $100K NAV
        prices = np.array([45000, 2500, 150, ...])  # Current prices

        # Get position sizing
        result = sizer.compute_positions(nav, w_raw, prices)
        # result.final_weights: scaled weights
        # result.position_values: USDT amounts per asset
    """

    def __init__(
        self,
        target_leverage: float = 2.0,
        min_position_value: float = 100.0,  # Minimum $100 per position
        max_single_weight: float = 0.5,     # Max 50% in single asset
        weight_change_threshold: float = 0.02,  # 2% deadband for churning prevention
    ):
        """
        Initialize position sizer.

        Args:
            target_leverage: Maximum gross exposure (e.g., 2.0 = 200%)
            min_position_value: Minimum position value in USDT
            max_single_weight: Maximum weight for single asset
            weight_change_threshold: Minimum weight change to trigger rebalancing (deadband)
        """
        self.target_leverage = target_leverage
        self.min_position_value = min_position_value
        self.max_single_weight = max_single_weight
        self.weight_change_threshold = weight_change_threshold

    def compute_positions(
        self,
        nav: float,
        weights_raw: np.ndarray,
        prices: np.ndarray,
        margin_used: float = 0.0,
        current_weights: Optional[np.ndarray] = None,
    ) -> PositionInfo:
        """
        Compute final positions from raw actor weights.

        Pipeline:
        1. Clip individual weights to max_single_weight
        2. Scale to target leverage if gross exposure exceeds limit
        3. Apply deadband filter (skip small weight changes)
        4. Compute position values (USDT)
        5. Compute position sizes (contracts)

        Args:
            nav: Current Net Asset Value in USDT
            weights_raw: Raw actor output [-1, 1]^N from Tanh
            prices: Current asset prices in USDT
            margin_used: Currently used margin (for cash ratio)
            current_weights: Current portfolio weights for deadband comparison

        Returns:
            PositionInfo with all sizing details
        """
        n_assets = len(weights_raw)

        # Step 1: Clip individual weights
        weights = np.clip(weights_raw, -self.max_single_weight, self.max_single_weight)

        # Step 2: Compute gross exposure
        gross_exposure = np.abs(weights).sum()

        # Step 3: Scale to target leverage if needed
        if gross_exposure > self.target_leverage:
            scale = self.target_leverage / gross_exposure
            weights = weights * scale
            gross_exposure = self.target_leverage

        # Step 4: Apply deadband filter (skip small weight changes to prevent churning)
        if current_weights is not None:
            weight_changes = np.abs(weights - current_weights)
            small_changes = weight_changes < self.weight_change_threshold
            weights[small_changes] = current_weights[small_changes]
            # Recalculate gross exposure after deadband
            gross_exposure = np.abs(weights).sum()

        final_weights = weights

        # Step 4: Compute position values (USDT)
        position_values = nav * final_weights

        # Step 5: Filter small positions
        small_mask = np.abs(position_values) < self.min_position_value
        position_values[small_mask] = 0.0
        final_weights[small_mask] = 0.0

        # Step 6: Compute position sizes (contracts)
        # Avoid division by zero for prices
        safe_prices = np.where(prices > 0, prices, 1.0)
        position_sizes = position_values / safe_prices

        # Step 7: Compute metrics
        net_exposure = final_weights.sum()
        leverage_ratio = gross_exposure / self.target_leverage if self.target_leverage > 0 else 0.0
        cash_ratio = (nav - margin_used) / nav if nav > 0 else 0.0

        return PositionInfo(
            final_weights=final_weights,
            position_values=position_values,
            position_sizes=position_sizes,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            leverage_ratio=leverage_ratio,
            cash_ratio=max(0.0, cash_ratio),  # Ensure non-negative
        )

    def compute_rebalance_orders(
        self,
        current_positions: np.ndarray,
        target_positions: np.ndarray,
        prices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute orders needed to rebalance from current to target positions.

        Args:
            current_positions: Current position sizes (contracts)
            target_positions: Target position sizes (contracts)
            prices: Current asset prices

        Returns:
            Tuple of (order_sizes, order_values):
            - order_sizes: contracts to buy (+) or sell (-)
            - order_values: USDT value of each order
        """
        order_sizes = target_positions - current_positions
        order_values = order_sizes * prices

        return order_sizes, order_values

    def compute_transaction_cost(
        self,
        order_values: np.ndarray,
        maker_fee: float = 0.0002,  # 0.02% for Binance Futures
        taker_fee: float = 0.0004,  # 0.04% for Binance Futures
        is_market_order: bool = True,
    ) -> float:
        """
        Compute estimated transaction costs.

        Args:
            order_values: USDT value of each order
            maker_fee: Maker fee rate
            taker_fee: Taker fee rate
            is_market_order: True for market orders (taker), False for limit (maker)

        Returns:
            Total transaction cost in USDT
        """
        fee_rate = taker_fee if is_market_order else maker_fee
        total_turnover = np.abs(order_values).sum()
        return total_turnover * fee_rate

    def compute_leverage_ratio(self, weights: np.ndarray) -> float:
        """
        Compute leverage ratio for given weights.

        leverage_ratio = Σ|w_i| / target_leverage

        This is included in portfolio state for the RL agent.

        Args:
            weights: Portfolio weights

        Returns:
            Leverage ratio (0 to 1+ range)
        """
        gross_exposure = np.abs(weights).sum()
        return gross_exposure / self.target_leverage if self.target_leverage > 0 else 0.0

    def compute_cash_ratio(self, nav: float, margin_used: float) -> float:
        """
        Compute cash ratio for portfolio state.

        cash_ratio = (NAV - Margin_Used) / NAV

        Args:
            nav: Current NAV
            margin_used: Currently used margin

        Returns:
            Cash ratio (0 to 1 range)
        """
        if nav <= 0:
            return 0.0
        return max(0.0, (nav - margin_used) / nav)


class MarginCalculator:
    """
    Margin calculation for futures positions.

    Different exchanges have different margin modes:
    - Cross margin: shared across all positions
    - Isolated margin: per-position margin
    """

    def __init__(
        self,
        initial_margin_rate: float = 0.01,  # 1% = 100x max leverage
        maintenance_margin_rate: float = 0.005,  # 0.5%
    ):
        """
        Initialize margin calculator.

        Args:
            initial_margin_rate: Required initial margin rate
            maintenance_margin_rate: Maintenance margin rate
        """
        self.initial_margin_rate = initial_margin_rate
        self.maintenance_margin_rate = maintenance_margin_rate

    def compute_required_margin(
        self,
        position_values: np.ndarray,
        leverage: float = 10.0,
    ) -> float:
        """
        Compute total required margin.

        Args:
            position_values: Position values in USDT
            leverage: Position leverage

        Returns:
            Required margin in USDT
        """
        notional = np.abs(position_values).sum()
        return notional / leverage

    def compute_margin_ratio(
        self,
        nav: float,
        position_values: np.ndarray,
    ) -> float:
        """
        Compute margin ratio (for liquidation risk).

        margin_ratio = NAV / (Σ|position_values| × maintenance_margin_rate)

        Args:
            nav: Current NAV
            position_values: Position values

        Returns:
            Margin ratio (higher = safer)
        """
        maintenance_margin = np.abs(position_values).sum() * self.maintenance_margin_rate
        if maintenance_margin <= 0:
            return float('inf')
        return nav / maintenance_margin


# ========== Standalone Testing ==========

if __name__ == '__main__':
    # Test position sizer
    sizer = PositionSizer(target_leverage=2.0)

    # Simulated scenario
    nav = 100_000  # $100K
    n_assets = 20

    # Random raw weights from actor
    np.random.seed(42)
    w_raw = np.random.uniform(-0.8, 0.8, n_assets)

    # Random prices
    prices = np.random.uniform(10, 50000, n_assets)

    # Compute positions
    result = sizer.compute_positions(nav, w_raw, prices)

    print(f"NAV: ${nav:,.0f}")
    print(f"Raw weights sum: {w_raw.sum():.3f}")
    print(f"Raw weights abs sum: {np.abs(w_raw).sum():.3f}")
    print(f"\nAfter scaling:")
    print(f"Final weights sum: {result.final_weights.sum():.3f}")
    print(f"Gross exposure: {result.gross_exposure:.3f}")
    print(f"Net exposure: {result.net_exposure:.3f}")
    print(f"Leverage ratio: {result.leverage_ratio:.3f}")
    print(f"Cash ratio: {result.cash_ratio:.3f}")
    print(f"\nPosition values range: ${result.position_values.min():,.0f} to ${result.position_values.max():,.0f}")
    print(f"Total notional: ${np.abs(result.position_values).sum():,.0f}")
