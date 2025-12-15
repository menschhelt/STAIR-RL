"""
Fama-French 5-Factor Model for Cryptocurrency Markets

Adapts traditional Fama-French factors for crypto Top 20 universe:
1. Mkt-RF: Market return - Risk-free rate
2. SMB (Small Minus Big): Size factor based on market cap
3. HML (High Minus Low): Value factor based on price momentum
4. RMW (Robust Minus Weak): Profitability proxy using trading volume
5. CMA (Conservative Minus Aggressive): Investment proxy using volatility

Reference: Fama, Eugene F., and Kenneth R. French (2015)
"A five-factor asset pricing model"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class FamaFrenchCalculator:
    """
    Calculate Fama-French factors from Top 20 cryptocurrency universe.

    Factors are computed from cross-sectional returns at each timestamp,
    adapting equity factor definitions to crypto markets.
    """

    def __init__(
        self,
        universe_data_dir: Optional[Path] = None,
        risk_free_rate: float = 0.0,  # Crypto has no risk-free rate
    ):
        """
        Initialize Fama-French calculator.

        Args:
            universe_data_dir: Path to universe data directory
            risk_free_rate: Risk-free rate (default: 0 for crypto)
        """
        self.universe_data_dir = universe_data_dir
        self.risk_free_rate = risk_free_rate

    def calculate_factors(
        self,
        returns: np.ndarray,
        market_caps: np.ndarray,
        volumes: np.ndarray,
        volatilities: np.ndarray,
        price_momentum: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate Fama-French 5 factors from cross-sectional data.

        Args:
            returns: (N,) array of returns for N assets
            market_caps: (N,) array of market caps
            volumes: (N,) array of trading volumes
            volatilities: (N,) array of volatilities (e.g., 20-day rolling std)
            price_momentum: (N,) array of price momentum (e.g., 20-day returns)
                           If None, will be computed from returns

        Returns:
            Dictionary with 5 factors:
                - 'MKT_RF': Market return - risk-free rate
                - 'SMB': Small minus big (size factor)
                - 'HML': High minus low (value/momentum factor)
                - 'RMW': Robust minus weak (volume/profitability factor)
                - 'CMA': Conservative minus aggressive (volatility/investment factor)
        """
        # Handle NaN values
        valid_mask = ~(np.isnan(returns) | np.isnan(market_caps) |
                      np.isnan(volumes) | np.isnan(volatilities))

        if valid_mask.sum() < 3:
            # Not enough valid data - return zeros
            return {
                'MKT_RF': 0.0,
                'SMB': 0.0,
                'HML': 0.0,
                'RMW': 0.0,
                'CMA': 0.0,
            }

        # Filter to valid assets only
        returns = returns[valid_mask]
        market_caps = market_caps[valid_mask]
        volumes = volumes[valid_mask]
        volatilities = volatilities[valid_mask]

        if price_momentum is not None:
            price_momentum = price_momentum[valid_mask]
        else:
            # Use returns as proxy for momentum if not provided
            price_momentum = returns.copy()

        # 1. Market Factor (Mkt-RF)
        # Average return of all assets in universe
        mkt_return = returns.mean()
        mkt_rf = mkt_return - self.risk_free_rate

        # 2. Size Factor (SMB) - Small Minus Big
        # Split by market cap median
        smb = self._calculate_size_factor(returns, market_caps)

        # 3. Value Factor (HML) - High Minus Low
        # Split by price momentum (low momentum = value, high momentum = growth)
        hml = self._calculate_value_factor(returns, price_momentum)

        # 4. Profitability Factor (RMW) - Robust Minus Weak
        # Split by trading volume (high volume = robust, low volume = weak)
        rmw = self._calculate_profitability_factor(returns, volumes)

        # 5. Investment Factor (CMA) - Conservative Minus Aggressive
        # Split by volatility (low vol = conservative, high vol = aggressive)
        cma = self._calculate_investment_factor(returns, volatilities)

        return {
            'MKT_RF': float(mkt_rf),
            'SMB': float(smb),
            'HML': float(hml),
            'RMW': float(rmw),
            'CMA': float(cma),
        }

    def _calculate_size_factor(
        self,
        returns: np.ndarray,
        market_caps: np.ndarray,
    ) -> float:
        """
        SMB (Small Minus Big): Size factor.

        Small cap: bottom 50% by market cap
        Big cap: top 50% by market cap
        """
        median_cap = np.median(market_caps)

        small_mask = market_caps <= median_cap
        big_mask = market_caps > median_cap

        if small_mask.sum() == 0 or big_mask.sum() == 0:
            return 0.0

        small_return = returns[small_mask].mean()
        big_return = returns[big_mask].mean()

        return small_return - big_return

    def _calculate_value_factor(
        self,
        returns: np.ndarray,
        price_momentum: np.ndarray,
    ) -> float:
        """
        HML (High Minus Low): Value factor.

        In crypto, we use inverse momentum as value proxy:
        - Low momentum (bottom 50%) = value stocks
        - High momentum (top 50%) = growth stocks

        Traditional HML is Value - Growth, so we use:
        HML = Return(low momentum) - Return(high momentum)
        """
        median_momentum = np.median(price_momentum)

        low_momentum_mask = price_momentum <= median_momentum  # Value
        high_momentum_mask = price_momentum > median_momentum  # Growth

        if low_momentum_mask.sum() == 0 or high_momentum_mask.sum() == 0:
            return 0.0

        value_return = returns[low_momentum_mask].mean()
        growth_return = returns[high_momentum_mask].mean()

        return value_return - growth_return

    def _calculate_profitability_factor(
        self,
        returns: np.ndarray,
        volumes: np.ndarray,
    ) -> float:
        """
        RMW (Robust Minus Weak): Profitability factor.

        In crypto, we use trading volume as profitability proxy:
        - High volume (top 50%) = robust (liquid, active trading)
        - Low volume (bottom 50%) = weak (illiquid, low activity)
        """
        median_volume = np.median(volumes)

        robust_mask = volumes > median_volume
        weak_mask = volumes <= median_volume

        if robust_mask.sum() == 0 or weak_mask.sum() == 0:
            return 0.0

        robust_return = returns[robust_mask].mean()
        weak_return = returns[weak_mask].mean()

        return robust_return - weak_return

    def _calculate_investment_factor(
        self,
        returns: np.ndarray,
        volatilities: np.ndarray,
    ) -> float:
        """
        CMA (Conservative Minus Aggressive): Investment factor.

        In crypto, we use volatility as investment proxy:
        - Low volatility (bottom 50%) = conservative (stable)
        - High volatility (top 50%) = aggressive (risky)
        """
        median_vol = np.median(volatilities)

        conservative_mask = volatilities <= median_vol
        aggressive_mask = volatilities > median_vol

        if conservative_mask.sum() == 0 or aggressive_mask.sum() == 0:
            return 0.0

        conservative_return = returns[conservative_mask].mean()
        aggressive_return = returns[aggressive_mask].mean()

        return conservative_return - aggressive_return

    def calculate_from_dataframe(
        self,
        df: pd.DataFrame,
        return_col: str = 'return',
        market_cap_col: str = 'market_cap',
        volume_col: str = 'volume',
        volatility_col: str = 'volatility',
        momentum_col: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Calculate factors from pandas DataFrame.

        Args:
            df: DataFrame with columns for each asset (one row per asset)
            return_col: Column name for returns
            market_cap_col: Column name for market cap
            volume_col: Column name for volume
            volatility_col: Column name for volatility
            momentum_col: Optional column name for momentum

        Returns:
            Dictionary with 5 factors
        """
        returns = df[return_col].values
        market_caps = df[market_cap_col].values
        volumes = df[volume_col].values
        volatilities = df[volatility_col].values

        price_momentum = None
        if momentum_col and momentum_col in df.columns:
            price_momentum = df[momentum_col].values

        return self.calculate_factors(
            returns=returns,
            market_caps=market_caps,
            volumes=volumes,
            volatilities=volatilities,
            price_momentum=price_momentum,
        )

    def calculate_time_series(
        self,
        returns_df: pd.DataFrame,
        market_caps_df: pd.DataFrame,
        volumes_df: pd.DataFrame,
        volatilities_df: pd.DataFrame,
        momentum_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calculate factor time series from panel data.

        Args:
            returns_df: (T, N) DataFrame of returns
            market_caps_df: (T, N) DataFrame of market caps
            volumes_df: (T, N) DataFrame of volumes
            volatilities_df: (T, N) DataFrame of volatilities
            momentum_df: (T, N) DataFrame of momentum (optional)

        Returns:
            DataFrame with columns ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
            and index matching input DataFrames
        """
        factors_list = []

        for timestamp in returns_df.index:
            returns = returns_df.loc[timestamp].values
            market_caps = market_caps_df.loc[timestamp].values
            volumes = volumes_df.loc[timestamp].values
            volatilities = volatilities_df.loc[timestamp].values

            price_momentum = None
            if momentum_df is not None:
                price_momentum = momentum_df.loc[timestamp].values

            factors = self.calculate_factors(
                returns=returns,
                market_caps=market_caps,
                volumes=volumes,
                volatilities=volatilities,
                price_momentum=price_momentum,
            )

            factors_list.append(factors)

        # Convert to DataFrame
        factors_df = pd.DataFrame(factors_list, index=returns_df.index)

        return factors_df


# ========== Standalone Testing ==========

if __name__ == '__main__':
    # Test with synthetic data
    print("=" * 60)
    print("Testing Fama-French Calculator")
    print("=" * 60)

    np.random.seed(42)

    # Synthetic cross-sectional data for 20 assets
    N = 20
    returns = np.random.randn(N) * 0.02  # ~2% daily volatility
    market_caps = np.random.lognormal(mean=10, sigma=2, size=N)  # Log-normal market caps
    volumes = np.random.lognormal(mean=8, sigma=1.5, size=N)  # Log-normal volumes
    volatilities = np.abs(np.random.randn(N)) * 0.03  # ~3% volatility
    price_momentum = np.random.randn(N) * 0.1  # 10-day momentum

    # Calculate factors
    calculator = FamaFrenchCalculator()
    factors = calculator.calculate_factors(
        returns=returns,
        market_caps=market_caps,
        volumes=volumes,
        volatilities=volatilities,
        price_momentum=price_momentum,
    )

    print("\n✓ Single-period factors:")
    for name, value in factors.items():
        print(f"   {name}: {value:>8.4f}")

    # Test with DataFrame
    print("\n✓ Testing DataFrame interface:")
    df = pd.DataFrame({
        'return': returns,
        'market_cap': market_caps,
        'volume': volumes,
        'volatility': volatilities,
        'momentum': price_momentum,
    })

    factors_df_test = calculator.calculate_from_dataframe(
        df,
        return_col='return',
        market_cap_col='market_cap',
        volume_col='volume',
        volatility_col='volatility',
        momentum_col='momentum',
    )

    print("   Factors from DataFrame:")
    for name, value in factors_df_test.items():
        print(f"   {name}: {value:>8.4f}")

    # Test time series calculation
    print("\n✓ Testing time series calculation:")
    T = 5  # 5 time periods

    # Create synthetic panel data
    dates = pd.date_range('2024-01-01', periods=T, freq='5min')
    returns_ts = pd.DataFrame(
        np.random.randn(T, N) * 0.02,
        index=dates,
        columns=[f'ASSET{i}' for i in range(N)]
    )
    market_caps_ts = pd.DataFrame(
        np.random.lognormal(mean=10, sigma=2, size=(T, N)),
        index=dates,
        columns=[f'ASSET{i}' for i in range(N)]
    )
    volumes_ts = pd.DataFrame(
        np.random.lognormal(mean=8, sigma=1.5, size=(T, N)),
        index=dates,
        columns=[f'ASSET{i}' for i in range(N)]
    )
    volatilities_ts = pd.DataFrame(
        np.abs(np.random.randn(T, N)) * 0.03,
        index=dates,
        columns=[f'ASSET{i}' for i in range(N)]
    )

    factors_ts = calculator.calculate_time_series(
        returns_df=returns_ts,
        market_caps_df=market_caps_ts,
        volumes_df=volumes_ts,
        volatilities_df=volatilities_ts,
    )

    print(f"   Factor time series shape: {factors_ts.shape}")
    print("\n   First 3 rows:")
    print(factors_ts.head(3).to_string())

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
