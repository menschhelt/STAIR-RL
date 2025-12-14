"""
Characteristic Engine
=====================
Calculates individual asset characteristics used for portfolio sorting.
These are NOT factors themselves, but characteristics used to form portfolios.

Examples:
- Market cap -> Sort into size quintiles -> CSMB factor
- Momentum -> Sort into momentum quintiles -> CMOM factor
- Volatility -> Sort into volatility quintiles -> CVOL factor

Adapted for stair-local Parquet-based data system.
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
import logging


class CharacteristicEngine:
    """
    Calculate individual asset characteristics for portfolio formation.

    NOT factors! These are characteristics used to sort assets.
    Factors are portfolio returns calculated AFTER sorting.
    """

    def __init__(self):
        """Initialize Characteristic Engine."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def calculate_market_cap(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        volume_window: int = 30
    ) -> float:
        """
        Estimate market cap proxy: price * average_volume.

        In absence of true market cap data, use price * volume as proxy.
        Higher value = larger market cap.

        Args:
            symbol: Trading symbol
            price_data: OHLCV DataFrame for symbol
            timestamp: Current timestamp
            volume_window: Days for volume averaging (default: 30)

        Returns:
            Market cap proxy (float)
        """
        if timestamp not in price_data.index:
            return np.nan

        idx = price_data.index.get_loc(timestamp)
        history = price_data.iloc[max(0, idx - volume_window):idx + 1]

        if len(history) == 0:
            return np.nan

        current_price = history.iloc[-1]['close']
        avg_volume = history['volume'].mean()

        if pd.isna(current_price) or pd.isna(avg_volume) or avg_volume == 0:
            return np.nan

        return current_price * avg_volume

    def calculate_momentum(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        window: int = 30
    ) -> float:
        """
        Calculate momentum: (price[t] / price[t-window]) - 1

        Liu et al. (2019) uses various windows: r1,0, r2,0, ..., r100,0
        - r1,0: 1-day momentum
        - r30,0: 30-day momentum (most common)
        - r100,0: 100-day momentum

        Args:
            symbol: Trading symbol
            price_data: OHLCV DataFrame
            timestamp: Current timestamp
            window: Lookback window in days (default: 30)

        Returns:
            Momentum (return over window)
        """
        if timestamp not in price_data.index:
            return np.nan

        idx = price_data.index.get_loc(timestamp)

        if idx < window:
            return np.nan

        current_price = price_data.iloc[idx]['close']
        past_price = price_data.iloc[idx - window]['close']

        if pd.isna(current_price) or pd.isna(past_price) or past_price == 0:
            return np.nan

        return (current_price / past_price) - 1

    def calculate_volatility(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        window: int = 30
    ) -> float:
        """
        Calculate return volatility (standard deviation).

        Args:
            symbol: Trading symbol
            price_data: OHLCV DataFrame
            timestamp: Current timestamp
            window: Lookback window (default: 30)

        Returns:
            Volatility (std of returns)
        """
        if timestamp not in price_data.index:
            return np.nan

        idx = price_data.index.get_loc(timestamp)
        history = price_data.iloc[max(0, idx - window):idx + 1]

        if len(history) < 2:
            return np.nan

        returns = history['close'].pct_change().dropna()

        if len(returns) == 0:
            return np.nan

        return returns.std()

    def calculate_volume_ratio(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        window: int = 30
    ) -> float:
        """
        Calculate volume ratio: current_volume / avg_volume

        Higher ratio = higher liquidity / trading activity

        Args:
            symbol: Trading symbol
            price_data: OHLCV DataFrame
            timestamp: Current timestamp
            window: Window for average volume (default: 30)

        Returns:
            Volume ratio (current / average)
        """
        if timestamp not in price_data.index:
            return np.nan

        idx = price_data.index.get_loc(timestamp)
        history = price_data.iloc[max(0, idx - window):idx + 1]

        if len(history) == 0:
            return np.nan

        current_volume = history.iloc[-1]['volume']
        avg_volume = history['volume'].mean()

        if pd.isna(current_volume) or pd.isna(avg_volume) or avg_volume == 0:
            return np.nan

        return current_volume / avg_volume

    def calculate_price_level(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        window: int = 100
    ) -> float:
        """
        Calculate price level: current_price / max_price_in_window

        Liu et al. (2019) MAXDPRC factor: maximum daily price
        Higher value = near all-time high

        Args:
            symbol: Trading symbol
            price_data: OHLCV DataFrame
            timestamp: Current timestamp
            window: Lookback window (default: 100)

        Returns:
            Price level ratio (0-1)
        """
        if timestamp not in price_data.index:
            return np.nan

        idx = price_data.index.get_loc(timestamp)
        history = price_data.iloc[max(0, idx - window):idx + 1]

        if len(history) == 0:
            return np.nan

        current_price = history.iloc[-1]['close']
        max_price = history['high'].max()

        if pd.isna(current_price) or pd.isna(max_price) or max_price == 0:
            return np.nan

        return current_price / max_price

    def calculate_funding_rate_avg(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        window: int = 30
    ) -> float:
        """
        Calculate average funding rate over window.

        Crypto-specific characteristic: reflects market sentiment.

        Args:
            symbol: Trading symbol
            price_data: OHLCV DataFrame with funding_rate column
            timestamp: Current timestamp
            window: Lookback window (default: 30)

        Returns:
            Average funding rate
        """
        if 'funding_rate' not in price_data.columns:
            return np.nan

        if timestamp not in price_data.index:
            return np.nan

        idx = price_data.index.get_loc(timestamp)
        history = price_data.iloc[max(0, idx - window):idx + 1]

        if len(history) == 0:
            return np.nan

        funding_rates = history['funding_rate'].dropna()

        if len(funding_rates) == 0:
            return np.nan

        return funding_rates.mean()

    def calculate_all_characteristics(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Calculate all characteristics for a single asset.

        Returns:
            {
                'market_cap': X,
                'momentum_1d': Y,
                'momentum_7d': Z,
                'momentum_30d': W,
                'momentum_60d': V,
                'volatility_30d': U,
                'volume_ratio': T,
                'price_level': S,
                'funding_rate_avg': R
            }
        """
        return {
            # Size
            'market_cap': self.calculate_market_cap(symbol, price_data, timestamp),

            # Momentum (various windows)
            'momentum_1d': self.calculate_momentum(symbol, price_data, timestamp, window=1),
            'momentum_7d': self.calculate_momentum(symbol, price_data, timestamp, window=7),
            'momentum_30d': self.calculate_momentum(symbol, price_data, timestamp, window=30),
            'momentum_60d': self.calculate_momentum(symbol, price_data, timestamp, window=60),
            'momentum_100d': self.calculate_momentum(symbol, price_data, timestamp, window=100),

            # Volatility
            'volatility_30d': self.calculate_volatility(symbol, price_data, timestamp, window=30),
            'volatility_60d': self.calculate_volatility(symbol, price_data, timestamp, window=60),

            # Liquidity
            'volume_ratio': self.calculate_volume_ratio(symbol, price_data, timestamp),

            # Price level
            'price_level': self.calculate_price_level(symbol, price_data, timestamp),

            # Crypto-specific
            'funding_rate_avg': self.calculate_funding_rate_avg(symbol, price_data, timestamp),
        }

    def calculate_characteristics_for_universe(
        self,
        universe_data: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp,
        characteristic_name: str = None
    ) -> Dict[str, float]:
        """
        Calculate specific characteristic for all assets in universe.

        Args:
            universe_data: {symbol: OHLCV DataFrame}
            timestamp: Current timestamp
            characteristic_name: Name of characteristic (e.g., 'market_cap', 'momentum_30d')
                                If None, returns all characteristics

        Returns:
            {symbol: characteristic_value} or {symbol: {all characteristics}}

        Example:
            >>> market_caps = engine.calculate_characteristics_for_universe(
            ...     universe_data=all_data,
            ...     timestamp=pd.Timestamp('2024-10-07'),
            ...     characteristic_name='market_cap'
            ... )
            >>> # {'BTCUSDT': 10000, 'ETHUSDT': 5000, ...}
        """
        if characteristic_name:
            # Single characteristic
            results = {}
            for symbol, price_data in universe_data.items():
                all_chars = self.calculate_all_characteristics(symbol, price_data, timestamp)
                if characteristic_name in all_chars:
                    results[symbol] = all_chars[characteristic_name]
            return results
        else:
            # All characteristics
            results = {}
            for symbol, price_data in universe_data.items():
                results[symbol] = self.calculate_all_characteristics(symbol, price_data, timestamp)
            return results


# ========== Standalone Testing ==========

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from datetime import datetime, timezone, timedelta

    # Generate sample OHLCV data
    np.random.seed(42)

    dates = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        periods=120,
        freq='D'
    )

    def generate_ohlcv(base_price: float) -> pd.DataFrame:
        """Generate random OHLCV data."""
        n = len(dates)
        returns = np.random.randn(n) * 0.03
        close = base_price * np.cumprod(1 + returns)

        return pd.DataFrame({
            'open': close * (1 + np.random.randn(n) * 0.01),
            'high': close * (1 + np.abs(np.random.randn(n) * 0.02)),
            'low': close * (1 - np.abs(np.random.randn(n) * 0.02)),
            'close': close,
            'volume': np.abs(np.random.randn(n) * 1000 + 5000),
            'funding_rate': np.random.randn(n) * 0.0001,
        }, index=dates)

    # Create universe data
    universe_data = {
        'BTCUSDT': generate_ohlcv(40000),
        'ETHUSDT': generate_ohlcv(2500),
        'SOLUSDT': generate_ohlcv(100),
    }

    # Test characteristics
    engine = CharacteristicEngine()
    timestamp = dates[-1]

    print("\n" + "=" * 60)
    print("CHARACTERISTIC ENGINE TEST")
    print("=" * 60)

    for symbol in universe_data:
        chars = engine.calculate_all_characteristics(symbol, universe_data[symbol], timestamp)
        print(f"\n{symbol}:")
        for name, value in chars.items():
            if not pd.isna(value):
                print(f"  {name}: {value:.6f}")

    # Test universe calculation
    print("\n" + "=" * 60)
    print("UNIVERSE CHARACTERISTICS (market_cap)")
    print("=" * 60)

    market_caps = engine.calculate_characteristics_for_universe(
        universe_data, timestamp, 'market_cap'
    )
    for symbol, cap in market_caps.items():
        print(f"  {symbol}: {cap:,.0f}")
