"""
Cross-Sectional Cryptocurrency Factor Engine
=============================================
Implements Liu, Tsyvinski, Wu (2019) cryptocurrency risk factors:
- CMKT: Crypto Market factor (equal-weighted universe return)
- CSMB: Crypto Size Minus Big (Small - Big portfolio returns)
- CMOM: Crypto Momentum (High - Low momentum returns)
- CVOL: Crypto Volatility (High - Low volatility returns) [optional]
- CLIQ: Crypto Liquidity (High - Low liquidity returns) [optional]

KEY DISTINCTION: These are PORTFOLIO RETURNS, not individual characteristics!

Adapted for stair-local Parquet-based data system.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging

from .portfolio_formation import PortfolioFormationEngine
from .characteristic_engine import CharacteristicEngine


class CryptoFactorEngine:
    """
    Calculate Liu et al. (2019) cryptocurrency factors using portfolio returns.

    Methodology:
    1. Calculate characteristics for each asset (market cap, momentum, etc.)
    2. Sort universe by characteristic
    3. Form portfolios (quintiles or top/bottom 30%)
    4. Calculate portfolio returns (value-weighted or equal-weighted)
    5. Factor = Long portfolio return - Short portfolio return
    """

    def __init__(
        self,
        portfolio_engine: PortfolioFormationEngine = None,
        characteristic_engine: CharacteristicEngine = None
    ):
        """
        Args:
            portfolio_engine: PortfolioFormationEngine instance
            characteristic_engine: CharacteristicEngine instance
        """
        self.portfolio_engine = portfolio_engine or PortfolioFormationEngine(weighting_method="value")
        self.char_engine = characteristic_engine or CharacteristicEngine()

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

    def calculate_CMKT(
        self,
        universe_symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp
    ) -> float:
        """
        CMKT: Crypto Market factor (equal-weighted universe return).

        This is the baseline market return that all assets are exposed to.

        Args:
            universe_symbols: List of available symbols
            price_data: {symbol: OHLCV DataFrame}
            timestamp: Current timestamp

        Returns:
            Market factor value (portfolio return)

        Example:
            >>> CMKT = engine.calculate_CMKT(
            ...     universe_symbols=['BTCUSDT', 'ETHUSDT', ...],
            ...     price_data=all_data,
            ...     timestamp=pd.Timestamp('2024-10-07')
            ... )
            >>> # CMKT ~ 0.015 (1.5% average market return)
        """
        # Use equal-weighted return (no market_caps parameter)
        temp_engine = PortfolioFormationEngine(weighting_method="equal")

        market_return = temp_engine.calculate_portfolio_return(
            portfolio_symbols=universe_symbols,
            price_data=price_data,
            timestamp=timestamp,
            market_caps=None
        )

        self.logger.debug(f"CMKT at {timestamp}: {market_return:.4f} ({market_return*100:.2f}%)")

        return market_return

    def calculate_CSMB(
        self,
        universe_symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp,
        market_caps: Dict[str, float] = None
    ) -> float:
        """
        CSMB: Crypto Size Minus Big (Small - Big portfolio returns).

        Size factor: Do small-cap assets outperform large-cap?

        Methodology (Liu et al. 2019):
        - Sort by market cap
        - Small portfolio: bottom 30%
        - Big portfolio: top 30%
        - CSMB = return(Small) - return(Big)

        Args:
            universe_symbols: List of available symbols
            price_data: {symbol: OHLCV DataFrame}
            timestamp: Current timestamp
            market_caps: {symbol: market_cap} (if None, calculate from price_data)

        Returns:
            Size factor value (long-short portfolio return)

        Example:
            >>> CSMB = engine.calculate_CSMB(
            ...     universe_symbols=symbols,
            ...     price_data=all_data,
            ...     timestamp=pd.Timestamp('2024-10-07'),
            ...     market_caps={'BTCUSDT': 10000, 'SHIBUSDT': 100, ...}
            ... )
            >>> # CSMB ~ 0.01 (small-cap outperform by 1%)
        """
        # Calculate market caps if not provided
        if market_caps is None:
            market_caps = self.char_engine.calculate_characteristics_for_universe(
                {s: price_data[s] for s in universe_symbols if s in price_data},
                timestamp,
                'market_cap'
            )

        # Form top/bottom portfolios by market cap
        big_portfolio, small_portfolio = self.portfolio_engine.form_top_bottom_portfolios(
            universe_symbols=universe_symbols,
            characteristics=market_caps,
            top_pct=0.3,
            bottom_pct=0.3
        )

        # Calculate long-short return (Small - Big)
        CSMB = self.portfolio_engine.calculate_long_short_return(
            long_portfolio=small_portfolio,  # Small (bottom 30%)
            short_portfolio=big_portfolio,   # Big (top 30%)
            price_data=price_data,
            timestamp=timestamp,
            market_caps=market_caps
        )

        self.logger.debug(
            f"CSMB at {timestamp}: {CSMB:.4f} ({CSMB*100:.2f}%) | "
            f"Small: {len(small_portfolio)}, Big: {len(big_portfolio)}"
        )

        return CSMB

    def calculate_CMOM(
        self,
        universe_symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp,
        momentum_window: int = 30,
        market_caps: Dict[str, float] = None
    ) -> float:
        """
        CMOM: Crypto Momentum factor (High - Low momentum returns).

        Momentum factor: Do high momentum assets continue to outperform?

        Methodology:
        - Sort by momentum (30-day return by default)
        - High portfolio: top 30%
        - Low portfolio: bottom 30%
        - CMOM = return(High) - return(Low)

        Args:
            universe_symbols: List of available symbols
            price_data: {symbol: OHLCV DataFrame}
            timestamp: Current timestamp
            momentum_window: Momentum lookback window (default: 30 days)
            market_caps: {symbol: market_cap} for value weighting (optional)

        Returns:
            Momentum factor value (long-short portfolio return)

        Example:
            >>> CMOM = engine.calculate_CMOM(
            ...     universe_symbols=symbols,
            ...     price_data=all_data,
            ...     timestamp=pd.Timestamp('2024-10-07'),
            ...     momentum_window=30
            ... )
            >>> # CMOM ~ 0.005 (winners continue winning by 0.5%)
        """
        # Calculate momentum for all symbols
        momentum_values = {}
        for symbol in universe_symbols:
            if symbol not in price_data:
                continue
            momentum = self.char_engine.calculate_momentum(
                symbol, price_data[symbol], timestamp, window=momentum_window
            )
            momentum_values[symbol] = momentum

        # Form top/bottom portfolios by momentum
        high_mom_portfolio, low_mom_portfolio = self.portfolio_engine.form_top_bottom_portfolios(
            universe_symbols=universe_symbols,
            characteristics=momentum_values,
            top_pct=0.3,
            bottom_pct=0.3
        )

        # Calculate long-short return (High - Low)
        CMOM = self.portfolio_engine.calculate_long_short_return(
            long_portfolio=high_mom_portfolio,   # High momentum
            short_portfolio=low_mom_portfolio,   # Low momentum
            price_data=price_data,
            timestamp=timestamp,
            market_caps=market_caps
        )

        self.logger.debug(
            f"CMOM at {timestamp}: {CMOM:.4f} ({CMOM*100:.2f}%) | "
            f"High: {len(high_mom_portfolio)}, Low: {len(low_mom_portfolio)}"
        )

        return CMOM

    def calculate_CVOL(
        self,
        universe_symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp,
        volatility_window: int = 30,
        market_caps: Dict[str, float] = None
    ) -> float:
        """
        CVOL: Crypto Volatility factor (Low - High volatility returns).

        Volatility factor: Do low volatility assets outperform?
        Note: Typically LONG low vol, SHORT high vol (opposite of momentum)

        Args:
            universe_symbols: List of available symbols
            price_data: {symbol: OHLCV DataFrame}
            timestamp: Current timestamp
            volatility_window: Volatility window (default: 30)
            market_caps: {symbol: market_cap} for weighting

        Returns:
            Volatility factor (Low vol - High vol return)
        """
        # Calculate volatility for all symbols
        volatility_values = {}
        for symbol in universe_symbols:
            if symbol not in price_data:
                continue
            vol = self.char_engine.calculate_volatility(
                symbol, price_data[symbol], timestamp, window=volatility_window
            )
            volatility_values[symbol] = vol

        # Form portfolios
        high_vol_portfolio, low_vol_portfolio = self.portfolio_engine.form_top_bottom_portfolios(
            universe_symbols=universe_symbols,
            characteristics=volatility_values,
            top_pct=0.3,
            bottom_pct=0.3
        )

        # Low vol - High vol (low vol premium)
        CVOL = self.portfolio_engine.calculate_long_short_return(
            long_portfolio=low_vol_portfolio,   # Low volatility
            short_portfolio=high_vol_portfolio, # High volatility
            price_data=price_data,
            timestamp=timestamp,
            market_caps=market_caps
        )

        self.logger.debug(f"CVOL at {timestamp}: {CVOL:.4f} ({CVOL*100:.2f}%)")

        return CVOL

    def calculate_CLIQ(
        self,
        universe_symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp,
        market_caps: Dict[str, float] = None
    ) -> float:
        """
        CLIQ: Crypto Liquidity factor (High - Low liquidity returns).

        Liquidity factor: Do high liquidity assets outperform?

        Args:
            universe_symbols: List of available symbols
            price_data: {symbol: OHLCV DataFrame}
            timestamp: Current timestamp
            market_caps: {symbol: market_cap}

        Returns:
            Liquidity factor (High liq - Low liq return)
        """
        # Calculate liquidity (volume ratio) for all symbols
        liquidity_values = {}
        for symbol in universe_symbols:
            if symbol not in price_data:
                continue
            liq = self.char_engine.calculate_volume_ratio(
                symbol, price_data[symbol], timestamp
            )
            liquidity_values[symbol] = liq

        # Form portfolios
        high_liq_portfolio, low_liq_portfolio = self.portfolio_engine.form_top_bottom_portfolios(
            universe_symbols=universe_symbols,
            characteristics=liquidity_values,
            top_pct=0.3,
            bottom_pct=0.3
        )

        # High liq - Low liq
        CLIQ = self.portfolio_engine.calculate_long_short_return(
            long_portfolio=high_liq_portfolio,
            short_portfolio=low_liq_portfolio,
            price_data=price_data,
            timestamp=timestamp,
            market_caps=market_caps
        )

        self.logger.debug(f"CLIQ at {timestamp}: {CLIQ:.4f} ({CLIQ*100:.2f}%)")

        return CLIQ

    def calculate_CPRL(
        self,
        universe_symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp,
        price_level_window: int = 100,
        market_caps: Dict[str, float] = None
    ) -> float:
        """
        CPRL: Crypto Price Level factor (Low - High price level returns).

        Price level factor: Do assets near their highs underperform?
        (Contrarian effect)

        Args:
            universe_symbols: List of available symbols
            price_data: {symbol: OHLCV DataFrame}
            timestamp: Current timestamp
            price_level_window: Window for max price calculation
            market_caps: {symbol: market_cap}

        Returns:
            Price level factor (Low level - High level return)
        """
        # Calculate price level for all symbols
        price_level_values = {}
        for symbol in universe_symbols:
            if symbol not in price_data:
                continue
            level = self.char_engine.calculate_price_level(
                symbol, price_data[symbol], timestamp, window=price_level_window
            )
            price_level_values[symbol] = level

        # Form portfolios
        high_level, low_level = self.portfolio_engine.form_top_bottom_portfolios(
            universe_symbols=universe_symbols,
            characteristics=price_level_values,
            top_pct=0.3,
            bottom_pct=0.3
        )

        # Low level - High level (contrarian: low level outperforms)
        CPRL = self.portfolio_engine.calculate_long_short_return(
            long_portfolio=low_level,
            short_portfolio=high_level,
            price_data=price_data,
            timestamp=timestamp,
            market_caps=market_caps
        )

        self.logger.debug(f"CPRL at {timestamp}: {CPRL:.4f} ({CPRL*100:.2f}%)")

        return CPRL

    def calculate_CREV(
        self,
        universe_symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp,
        reversal_window: int = 7,
        market_caps: Dict[str, float] = None
    ) -> float:
        """
        CREV: Crypto Short-term Reversal factor (Loser - Winner returns).

        Reversal factor: Do recent losers outperform recent winners?
        (Short-term mean reversion effect)

        Args:
            universe_symbols: List of available symbols
            price_data: {symbol: OHLCV DataFrame}
            timestamp: Current timestamp
            reversal_window: Lookback window for reversal (default: 7 days)
            market_caps: {symbol: market_cap}

        Returns:
            Reversal factor (Loser - Winner return)
        """
        # Calculate short-term momentum (which we'll reverse)
        momentum_values = {}
        for symbol in universe_symbols:
            if symbol not in price_data:
                continue
            mom = self.char_engine.calculate_momentum(
                symbol, price_data[symbol], timestamp, window=reversal_window
            )
            momentum_values[symbol] = mom

        # Form portfolios
        winners, losers = self.portfolio_engine.form_top_bottom_portfolios(
            universe_symbols=universe_symbols,
            characteristics=momentum_values,
            top_pct=0.3,
            bottom_pct=0.3
        )

        # Loser - Winner (reversal: losers outperform)
        CREV = self.portfolio_engine.calculate_long_short_return(
            long_portfolio=losers,
            short_portfolio=winners,
            price_data=price_data,
            timestamp=timestamp,
            market_caps=market_caps
        )

        self.logger.debug(f"CREV at {timestamp}: {CREV:.4f} ({CREV*100:.2f}%)")

        return CREV

    def calculate_all_factors(
        self,
        universe_symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp,
        market_caps: Dict[str, float] = None,
        include_optional: bool = False,
        windows: Dict[str, List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate all cryptocurrency factors at once.

        Args:
            universe_symbols: List of available symbols
            price_data: {symbol: OHLCV DataFrame}
            timestamp: Current timestamp
            market_caps: {symbol: market_cap} (calculated if None)
            include_optional: Include CVOL, CLIQ (default: False)
            windows: Multi-window configuration, e.g.:
                {
                    'momentum': ['1d', '7d', '30d', '60d', '180d'],
                    'volatility': ['7d', '14d', '30d', '60d'],
                    'liquidity': ['7d', '30d']
                }

        Returns:
            {
                'CMKT': 0.015,
                'CSMB': 0.003,
                'CMOM_30d': 0.005,
                'CMOM_60d': 0.007,
                'CVOL_30d': -0.002,  # if include_optional
                'CLIQ_30d': 0.001    # if include_optional
            }

        Example:
            >>> factors = engine.calculate_all_factors(
            ...     universe_symbols=['BTCUSDT', 'ETHUSDT', ...],
            ...     price_data=all_data,
            ...     timestamp=pd.Timestamp('2024-10-07'),
            ...     include_optional=True,
            ...     windows={'momentum': ['30d', '60d'], 'volatility': ['30d']}
            ... )
            >>> print(factors)
            {'CMKT': 0.015, 'CSMB': 0.003, 'CMOM_30d': 0.005, 'CMOM_60d': 0.007, 'CVOL_30d': -0.002}
        """
        # Calculate market caps once
        if market_caps is None:
            market_caps = self.char_engine.calculate_characteristics_for_universe(
                {s: price_data[s] for s in universe_symbols if s in price_data},
                timestamp,
                'market_cap'
            )

        factors = {
            'CMKT': self.calculate_CMKT(universe_symbols, price_data, timestamp),
            'CSMB': self.calculate_CSMB(universe_symbols, price_data, timestamp, market_caps),
        }

        # Multi-window support
        if windows is None:
            # Default: single window (backward compatible)
            windows = {
                'momentum': ['30d'],
                'volatility': ['30d'] if include_optional else [],
                'liquidity': ['30d'] if include_optional else []
            }

        # Parse window string to integer days
        def parse_window(window_str: str) -> int:
            """Convert '30d' -> 30, '7d' -> 7, etc."""
            if window_str.endswith('d'):
                return int(window_str[:-1])
            return int(window_str)

        # Momentum factors (multiple windows)
        for window_str in windows.get('momentum', ['30d']):
            window_days = parse_window(window_str)
            factor_name = f'CMOM_{window_str}' if len(windows.get('momentum', [])) > 1 else 'CMOM'
            factors[factor_name] = self.calculate_CMOM(
                universe_symbols, price_data, timestamp,
                momentum_window=window_days, market_caps=market_caps
            )

        # Volatility factors (optional, multiple windows)
        if include_optional:
            for window_str in windows.get('volatility', ['30d']):
                window_days = parse_window(window_str)
                factor_name = f'CVOL_{window_str}' if len(windows.get('volatility', [])) > 1 else 'CVOL'
                factors[factor_name] = self.calculate_CVOL(
                    universe_symbols, price_data, timestamp,
                    volatility_window=window_days, market_caps=market_caps
                )

        # Liquidity factors (optional, multiple windows)
        if include_optional:
            for window_str in windows.get('liquidity', ['30d']):
                factor_name = (
                    f'CLIQ_{window_str}'
                    if len(windows.get('liquidity', [])) > 1
                    else 'CLIQ'
                )
                factors[factor_name] = self.calculate_CLIQ(
                    universe_symbols, price_data, timestamp, market_caps=market_caps
                )

        # Price Level factors (optional, multiple windows)
        if include_optional:
            for window_str in windows.get('price_level', ['100d']):
                window_days = parse_window(window_str)
                factor_name = (
                    f'CPRL_{window_str}'
                    if len(windows.get('price_level', [])) > 1
                    else 'CPRL'
                )
                factors[factor_name] = self.calculate_CPRL(
                    universe_symbols, price_data, timestamp,
                    price_level_window=window_days, market_caps=market_caps
                )

        # Reversal factors (optional, multiple windows)
        if include_optional:
            for window_str in windows.get('reversal', ['7d']):
                window_days = parse_window(window_str)
                factor_name = (
                    f'CREV_{window_str}'
                    if len(windows.get('reversal', [])) > 1
                    else 'CREV'
                )
                factors[factor_name] = self.calculate_CREV(
                    universe_symbols, price_data, timestamp,
                    reversal_window=window_days, market_caps=market_caps
                )

        self.logger.info(
            f"Crypto factors at {timestamp}: " +
            ", ".join([f"{k}={v:.4f}" for k, v in factors.items()])
        )

        return factors


# ========== Standalone Testing ==========

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from datetime import datetime, timezone

    # Generate sample data
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
        }, index=dates)

    # Create sample universe
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
               'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT']

    price_data = {s: generate_ohlcv(100 + np.random.rand() * 1000) for s in symbols}

    # Test crypto factors
    engine = CryptoFactorEngine()
    timestamp = dates[-1]

    print("\n" + "=" * 60)
    print("CRYPTO FACTOR ENGINE TEST")
    print("=" * 60)

    # Calculate all factors
    factors = engine.calculate_all_factors(
        universe_symbols=symbols,
        price_data=price_data,
        timestamp=timestamp,
        include_optional=True,
        windows={
            'momentum': ['7d', '30d'],
            'volatility': ['30d'],
            'liquidity': ['30d'],
            'reversal': ['7d'],
            'price_level': ['100d']
        }
    )

    print("\nCalculated Factors:")
    for name, value in factors.items():
        print(f"  {name}: {value:.4f} ({value*100:.2f}%)")
