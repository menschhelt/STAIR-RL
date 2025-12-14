"""
Portfolio Formation Engine
==========================
Forms quintile portfolios and calculates portfolio returns for factor construction.
Implements Liu et al. (2019) and CAIA (2023) methodologies.

Adapted for stair-local Parquet-based data system.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging


class PortfolioFormationEngine:
    """
    Quintile portfolio formation and return calculation.

    Key features:
    - Sort universe by characteristics (size, momentum, etc.)
    - Form quintile portfolios (Q1-Q5) or top/bottom portfolios
    - Calculate value-weighted or equal-weighted returns
    - Handle edge cases (small universe, missing data)
    """

    def __init__(
        self,
        n_quintiles: int = 5,
        weighting_method: str = "value"
    ):
        """
        Args:
            n_quintiles: Number of quintiles (default: 5)
            weighting_method: "value" (market-cap weighted) or "equal"
        """
        if weighting_method not in ["value", "equal"]:
            raise ValueError(f"Invalid weighting_method: {weighting_method}")

        self.n_quintiles = n_quintiles
        self.weighting_method = weighting_method

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

    def form_quintile_portfolios(
        self,
        universe_symbols: List[str],
        characteristics: Dict[str, float],
        ascending: bool = True
    ) -> Dict[str, List[str]]:
        """
        Sort universe by characteristic and create quintile portfolios.

        Args:
            universe_symbols: List of available symbols
            characteristics: {symbol: characteristic_value}
            ascending: True for Q1=lowest, False for Q1=highest

        Returns:
            {Q1: [symbols], Q2: [symbols], ..., Q5: [symbols]}

        Example:
            >>> characteristics = {'BTCUSDT': 10000, 'ETHUSDT': 5000, ...}
            >>> portfolios = engine.form_quintile_portfolios(
            ...     universe_symbols=['BTCUSDT', 'ETHUSDT', ...],
            ...     characteristics=characteristics,
            ...     ascending=True  # Q1 = smallest market cap
            ... )
            >>> portfolios['Q1']  # Small cap portfolio
            ['SHIBUSDT', 'DOGEUSDT', ...]
        """
        # Filter valid symbols (have characteristic value)
        valid_symbols = [
            s for s in universe_symbols
            if s in characteristics and not pd.isna(characteristics[s])
        ]

        if len(valid_symbols) < self.n_quintiles:
            raise ValueError(
                f"Too few valid symbols ({len(valid_symbols)}) for "
                f"{self.n_quintiles} quintiles. Universe: {len(universe_symbols)}"
            )

        # Sort by characteristic
        sorted_symbols = sorted(
            valid_symbols,
            key=lambda s: characteristics[s],
            reverse=not ascending
        )

        # Create quintiles
        portfolios = {}
        quintile_size = len(sorted_symbols) // self.n_quintiles

        for i in range(self.n_quintiles):
            start_idx = i * quintile_size
            # Last quintile takes remainder
            end_idx = start_idx + quintile_size if i < self.n_quintiles - 1 else len(sorted_symbols)

            portfolios[f"Q{i+1}"] = sorted_symbols[start_idx:end_idx]

        self.logger.debug(
            f"Formed {self.n_quintiles} quintiles from {len(sorted_symbols)} symbols. "
            f"Sizes: {[len(portfolios[f'Q{i+1}']) for i in range(self.n_quintiles)]}"
        )

        return portfolios

    def form_top_bottom_portfolios(
        self,
        universe_symbols: List[str],
        characteristics: Dict[str, float],
        top_pct: float = 0.3,
        bottom_pct: float = 0.3
    ) -> Tuple[List[str], List[str]]:
        """
        Form top 30% and bottom 30% portfolios (Liu et al. 2019 methodology).

        Args:
            universe_symbols: List of available symbols
            characteristics: {symbol: characteristic_value}
            top_pct: Percentage for top portfolio (default: 0.3 = 30%)
            bottom_pct: Percentage for bottom portfolio (default: 0.3 = 30%)

        Returns:
            (top_portfolio, bottom_portfolio)

        Example:
            >>> # Size factor: Small - Big
            >>> top, bottom = engine.form_top_bottom_portfolios(
            ...     universe_symbols=symbols,
            ...     characteristics=market_caps,
            ...     top_pct=0.3, bottom_pct=0.3
            ... )
            >>> # top = Big (highest market cap 30%)
            >>> # bottom = Small (lowest market cap 30%)
        """
        # Filter valid symbols
        valid_symbols = [
            s for s in universe_symbols
            if s in characteristics and not pd.isna(characteristics[s])
        ]

        if len(valid_symbols) < 3:
            raise ValueError(
                f"Too few valid symbols ({len(valid_symbols)}) for top/bottom portfolios"
            )

        # Sort by characteristic (highest first)
        sorted_symbols = sorted(
            valid_symbols,
            key=lambda s: characteristics[s],
            reverse=True
        )

        # Calculate portfolio sizes
        n_top = max(1, int(len(sorted_symbols) * top_pct))
        n_bottom = max(1, int(len(sorted_symbols) * bottom_pct))

        top_portfolio = sorted_symbols[:n_top]
        bottom_portfolio = sorted_symbols[-n_bottom:]

        self.logger.debug(
            f"Formed top/bottom portfolios: "
            f"top={n_top} ({top_pct*100}%), "
            f"bottom={n_bottom} ({bottom_pct*100}%), "
            f"universe={len(sorted_symbols)}"
        )

        return top_portfolio, bottom_portfolio

    def calculate_portfolio_return(
        self,
        portfolio_symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp,
        market_caps: Dict[str, float] = None,
        window: int = 1
    ) -> float:
        """
        Calculate portfolio return (value-weighted or equal-weighted).

        Args:
            portfolio_symbols: List of symbols in portfolio
            price_data: {symbol: OHLCV DataFrame}
            timestamp: Current timestamp
            market_caps: {symbol: market_cap} for value weighting (optional)
            window: Lookback window in periods (default: 1)

        Returns:
            Portfolio return (e.g., 0.02 for 2%)

        Example:
            >>> portfolio = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
            >>> ret = engine.calculate_portfolio_return(
            ...     portfolio_symbols=portfolio,
            ...     price_data=all_data,
            ...     timestamp=pd.Timestamp('2024-10-07'),
            ...     market_caps={'BTCUSDT': 10000, 'ETHUSDT': 5000, 'SOLUSDT': 1000},
            ...     window=5  # 5-period return
            ... )
            >>> # Value-weighted: (10000*0.01 + 5000*0.02 + 1000*0.03) / 16000
        """
        if not portfolio_symbols:
            return 0.0

        returns = []
        weights = []

        for symbol in portfolio_symbols:
            if symbol not in price_data:
                continue

            df = price_data[symbol]

            # Timestamp must exist in data
            if timestamp not in df.index:
                continue

            # Get current and previous price
            idx = df.index.get_loc(timestamp)
            if idx < window:
                # Not enough history for window
                continue

            current_price = df.loc[timestamp, 'close']
            prev_price = df.iloc[idx - window]['close']

            # Calculate return
            symbol_return = (current_price / prev_price) - 1

            # Skip invalid returns
            if pd.isna(symbol_return) or np.isinf(symbol_return):
                continue

            returns.append(symbol_return)

            # Determine weight
            if self.weighting_method == "value" and market_caps:
                weight = market_caps.get(symbol, 1.0)
            else:
                weight = 1.0

            weights.append(weight)

        # No valid returns
        if not returns:
            self.logger.warning(
                f"No valid returns for portfolio at {timestamp}. "
                f"Symbols: {portfolio_symbols[:3]}..."
            )
            return 0.0

        # Calculate weighted return
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weighted_return = sum(
            r * w for r, w in zip(returns, weights)
        ) / total_weight

        return weighted_return

    def calculate_long_short_return(
        self,
        long_portfolio: List[str],
        short_portfolio: List[str],
        price_data: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp,
        market_caps: Dict[str, float] = None
    ) -> float:
        """
        Calculate long-short portfolio return (e.g., Small - Big for CSMB).

        Args:
            long_portfolio: Symbols to go long
            short_portfolio: Symbols to go short
            price_data: {symbol: OHLCV DataFrame}
            timestamp: Current timestamp
            market_caps: {symbol: market_cap} for value weighting

        Returns:
            Long-short return (long_return - short_return)

        Example:
            >>> # CSMB: Small - Big
            >>> small_return = 0.02  # Small portfolio +2%
            >>> big_return = 0.01    # Big portfolio +1%
            >>> CSMB = 0.02 - 0.01 = 0.01  # Size factor = 1%
        """
        long_return = self.calculate_portfolio_return(
            long_portfolio, price_data, timestamp, market_caps
        )

        short_return = self.calculate_portfolio_return(
            short_portfolio, price_data, timestamp, market_caps
        )

        return long_return - short_return


# ========== Standalone Testing ==========

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from datetime import datetime, timezone

    # Generate sample data
    np.random.seed(42)

    dates = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        periods=60,
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

    # Create sample data
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
               'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT']

    price_data = {s: generate_ohlcv(100 + np.random.rand() * 1000) for s in symbols}

    # Market caps (proxy: random)
    market_caps = {s: np.random.rand() * 10000 + 1000 for s in symbols}

    # Test portfolio formation
    engine = PortfolioFormationEngine(weighting_method='value')
    timestamp = dates[-1]

    print("\n" + "=" * 60)
    print("PORTFOLIO FORMATION TEST")
    print("=" * 60)

    # Form quintile portfolios
    quintiles = engine.form_quintile_portfolios(symbols, market_caps, ascending=True)
    print("\nQuintile Portfolios (by market cap, ascending):")
    for q, members in quintiles.items():
        print(f"  {q}: {members}")

    # Form top/bottom portfolios
    top, bottom = engine.form_top_bottom_portfolios(symbols, market_caps)
    print(f"\nTop 30% (Big): {top}")
    print(f"Bottom 30% (Small): {bottom}")

    # Calculate long-short return
    ls_return = engine.calculate_long_short_return(
        bottom, top, price_data, timestamp, market_caps
    )
    print(f"\nLong-Short Return (Small - Big): {ls_return:.4f} ({ls_return*100:.2f}%)")
