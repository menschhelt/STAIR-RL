#!/usr/bin/env python3
"""
Mock Data Generator for STAIR-RL Testing

Generates synthetic training data using OHLCV + technical indicators
instead of 292 alpha factors (which are still computing at 21%).

Purpose: Test training pipeline end-to-end without waiting for alphas.

Features (36 dimensions per asset):
  1-5:   OHLCV normalized
  6-10:  Multi-timeframe returns (1m, 5m, 15m, 1h, 4h)
  11-13: Volume features (pct_change, relative volume, quote volume)
  14-28: Technical indicators (RSI×2, MACD×3, Bollinger×3, ATR, ADX,
         Stochastic×2, OBV, MFI, Williams%R, CCI)
  29-33: Cross-sectional features (rank by volume/return, z-score,
         relative strength, beta to market)
  34-36: Momentum features (5/10/20-day momentum)

Usage:
    python scripts/generate_mock_data.py \
        --n-assets 10 \
        --start-date 2021-01-01 \
        --end-date 2021-02-28 \
        --output /home/work/data/stair-local/test_mock

Output Structure:
    /home/work/data/stair-local/test_mock/
    ├── binance/binance_futures_5m_202101_mock.parquet
    ├── universe/universe_history_mock.parquet
    └── features/mock_features/
        ├── BTCUSDT_features.parquet  (36 columns)
        └── ...
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default test assets (10 major crypto)
DEFAULT_ASSETS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT',
    'XRPUSDT', 'DOTUSDT', 'UNIUSDT', 'LTCUSDT', 'LINKUSDT'
]


class MockDataGenerator:
    """
    Generate mock training data from OHLCV + technical indicators.

    Uses actual Binance historical data and computes 36 features per asset.
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        data_dir: Path = Path('/home/work/data/stair-local'),
    ):
        """
        Initialize mock data generator.

        Args:
            symbols: List of crypto symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data_dir: Base data directory containing Binance parquet files
        """
        self.symbols = symbols
        self.start_date = pd.Timestamp(start_date, tz='UTC')
        self.end_date = pd.Timestamp(end_date, tz='UTC')
        self.data_dir = data_dir
        self.binance_dir = data_dir / 'binance'

        logger.info(f"Initialized MockDataGenerator")
        logger.info(f"  Symbols: {len(symbols)}")
        logger.info(f"  Period: {start_date} to {end_date}")

    def load_ohlcv(self, symbol: str) -> pd.DataFrame:
        """
        Load OHLCV data for a symbol from Binance parquet files.

        Args:
            symbol: Symbol (e.g., 'BTCUSDT')

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"Loading OHLCV for {symbol}...")

        # Find all parquet files in date range
        start_month = self.start_date.replace(day=1)
        end_month = self.end_date.replace(day=1)

        months = pd.date_range(start_month, end_month, freq='MS')

        dfs = []
        for month in months:
            # Expected filename: binance_futures_5m_YYYYMM.parquet
            filename = f"binance_futures_5m_{month.strftime('%Y%m')}.parquet"
            filepath = self.binance_dir / filename

            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                continue

            try:
                df = pd.read_parquet(filepath)

                # Filter by symbol and date range
                if 'symbol' in df.columns:
                    df = df[df['symbol'] == symbol]

                if 'timestamp' in df.columns:
                    df = df[(df['timestamp'] >= self.start_date) &
                           (df['timestamp'] <= self.end_date)]

                if len(df) > 0:
                    dfs.append(df)

            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")

        if not dfs:
            logger.error(f"No data found for {symbol}")
            return pd.DataFrame()

        # Combine and sort
        result = pd.concat(dfs, ignore_index=True)
        result = result.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"  Loaded {len(result):,} rows for {symbol}")

        return result

    def compute_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Compute 36 technical features from OHLCV.

        Args:
            df: OHLCV DataFrame
            symbol: Symbol name

        Returns:
            DataFrame with 36 feature columns
        """
        if df.empty:
            return pd.DataFrame()

        logger.info(f"Computing features for {symbol}...")

        # Make a copy to avoid modifying original
        features = pd.DataFrame(index=df.index)
        features['timestamp'] = df['timestamp']
        features['symbol'] = symbol

        # Extract OHLCV
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        v = df['volume'].values

        # --- Features 1-5: OHLCV Normalized ---
        # Normalize by close price
        features['f01_open_norm'] = o / c
        features['f02_high_norm'] = h / c
        features['f03_low_norm'] = l / c
        features['f04_close_norm'] = 1.0  # c / c = 1
        features['f05_volume_norm'] = v / (v.mean() + 1e-8)

        # --- Features 6-10: Multi-timeframe Returns ---
        features['f06_return_1bar'] = np.log(c / np.roll(c, 1))  # 5m
        features['f07_return_3bar'] = np.log(c / np.roll(c, 3))  # 15m
        features['f08_return_12bar'] = np.log(c / np.roll(c, 12))  # 1h
        features['f09_return_48bar'] = np.log(c / np.roll(c, 48))  # 4h
        features['f10_return_288bar'] = np.log(c / np.roll(c, 288))  # 1d

        # --- Features 11-13: Volume Features ---
        features['f11_volume_pct'] = pd.Series(v).pct_change().fillna(0).values
        features['f12_volume_rel'] = v / (pd.Series(v).rolling(20).mean().fillna(v.mean()).values + 1e-8)
        features['f13_quote_volume'] = v * c  # Quote volume (USDT)

        # --- Features 14-28: Technical Indicators ---

        # RSI (2 periods: 14, 28)
        features['f14_rsi14'] = self._compute_rsi(c, 14)
        features['f15_rsi28'] = self._compute_rsi(c, 28)

        # MACD (12, 26, 9)
        macd, signal, hist = self._compute_macd(c)
        features['f16_macd'] = macd
        features['f17_macd_signal'] = signal
        features['f18_macd_hist'] = hist

        # Bollinger Bands (20, 2)
        bb_upper, bb_middle, bb_lower = self._compute_bollinger(c, 20, 2)
        features['f19_bb_upper'] = (bb_upper - c) / (c + 1e-8)
        features['f20_bb_middle'] = (bb_middle - c) / (c + 1e-8)
        features['f21_bb_lower'] = (bb_lower - c) / (c + 1e-8)

        # ATR (14)
        features['f22_atr'] = self._compute_atr(h, l, c, 14)

        # ADX (14)
        features['f23_adx'] = self._compute_adx(h, l, c, 14)

        # Stochastic (14, 3)
        stoch_k, stoch_d = self._compute_stochastic(h, l, c, 14, 3)
        features['f24_stoch_k'] = stoch_k
        features['f25_stoch_d'] = stoch_d

        # OBV
        features['f26_obv'] = self._compute_obv(c, v)

        # MFI (14)
        features['f27_mfi'] = self._compute_mfi(h, l, c, v, 14)

        # Williams %R (14)
        features['f28_willr'] = self._compute_williams_r(h, l, c, 14)

        # --- Features 29-33: Cross-sectional (will be computed later) ---
        # Placeholder zeros (will be filled in cross-sectional step)
        features['f29_rank_volume'] = 0.0
        features['f30_rank_return'] = 0.0
        features['f31_zscore'] = 0.0
        features['f32_rel_strength'] = 0.0
        features['f33_beta'] = 0.0

        # --- Features 34-36: Momentum ---
        features['f34_mom5'] = np.log(c / np.roll(c, 60))   # 5-hour (60 bars)
        features['f35_mom10'] = np.log(c / np.roll(c, 120))  # 10-hour
        features['f36_mom20'] = np.log(c / np.roll(c, 240))  # 20-hour

        # Fill NaN with 0 (first few rows will have NaN due to rolling)
        features = features.fillna(0.0)

        logger.info(f"  Computed {len(features.columns)-2} features")

        return features

    # ========== Technical Indicator Helper Functions ==========

    def _compute_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute RSI indicator."""
        deltas = np.diff(prices, prepend=prices[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = pd.Series(gains).rolling(period).mean().fillna(0).values
        avg_loss = pd.Series(losses).rolling(period).mean().fillna(0).values

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        return rsi / 100.0  # Normalize to [0, 1]

    def _compute_macd(
        self,
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute MACD indicator."""
        ema_fast = pd.Series(prices).ewm(span=fast).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow).mean().values

        macd = ema_fast - ema_slow
        macd_signal = pd.Series(macd).ewm(span=signal).mean().values
        macd_hist = macd - macd_signal

        # Normalize by price
        price_avg = prices.mean()
        return (macd / price_avg,
                macd_signal / price_avg,
                macd_hist / price_avg)

    def _compute_bollinger(
        self,
        prices: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Bollinger Bands."""
        middle = pd.Series(prices).rolling(period).mean().fillna(prices.mean()).values
        std = pd.Series(prices).rolling(period).std().fillna(prices.std()).values

        upper = middle + std_dev * std
        lower = middle - std_dev * std

        return upper, middle, lower

    def _compute_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Compute Average True Range."""
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = pd.Series(true_range).rolling(period).mean().fillna(0).values

        # Normalize by price
        return atr / (close + 1e-8)

    def _compute_adx(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Compute Average Directional Index (simplified)."""
        # Directional movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # ATR
        atr = self._compute_atr(high, low, close, period)

        # Directional Indicators
        plus_di = pd.Series(plus_dm).rolling(period).mean().values / (atr * close + 1e-8)
        minus_di = pd.Series(minus_dm).rolling(period).mean().values / (atr * close + 1e-8)

        # ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = pd.Series(dx).rolling(period).mean().fillna(0).values

        return adx

    def _compute_stochastic(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Stochastic Oscillator."""
        lowest_low = pd.Series(low).rolling(k_period).min().fillna(low.min()).values
        highest_high = pd.Series(high).rolling(k_period).max().fillna(high.max()).values

        stoch_k = (close - lowest_low) / (highest_high - lowest_low + 1e-8)
        stoch_d = pd.Series(stoch_k).rolling(d_period).mean().fillna(0).values

        return stoch_k, stoch_d

    def _compute_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Compute On-Balance Volume."""
        price_change = np.diff(close, prepend=close[0])
        obv_direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
        obv = np.cumsum(obv_direction * volume)

        # Normalize by total volume
        return obv / (volume.sum() + 1e-8)

    def _compute_mfi(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Compute Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        price_change = np.diff(typical_price, prepend=typical_price[0])
        positive_flow = np.where(price_change > 0, money_flow, 0)
        negative_flow = np.where(price_change < 0, money_flow, 0)

        positive_mf = pd.Series(positive_flow).rolling(period).sum().fillna(0).values
        negative_mf = pd.Series(negative_flow).rolling(period).sum().fillna(0).values

        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-8)))

        return mfi / 100.0  # Normalize to [0, 1]

    def _compute_williams_r(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Compute Williams %R."""
        highest_high = pd.Series(high).rolling(period).max().fillna(high.max()).values
        lowest_low = pd.Series(low).rolling(period).min().fillna(low.min()).values

        willr = (highest_high - close) / (highest_high - lowest_low + 1e-8)

        return willr

    def compute_cross_sectional_features(
        self,
        all_features: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute cross-sectional features across all assets.

        Args:
            all_features: Dict of {symbol: features_df}

        Returns:
            Updated dict with cross-sectional features filled
        """
        logger.info("Computing cross-sectional features...")

        # Align all dataframes by timestamp
        timestamps = None
        for symbol, df in all_features.items():
            if timestamps is None:
                timestamps = df['timestamp'].values
            else:
                # Use intersection of timestamps
                timestamps = np.intersect1d(timestamps, df['timestamp'].values)

        if len(timestamps) == 0:
            logger.warning("No common timestamps found")
            return all_features

        # Initialize cross-sectional feature columns for all symbols
        for symbol in all_features.keys():
            df = all_features[symbol]
            df['f29_rank_volume'] = 0.0
            df['f30_rank_return'] = 0.0
            df['f31_zscore'] = 0.0
            df['f32_rel_strength'] = 0.0
            df['f33_beta'] = 0.0
            all_features[symbol] = df

        # Compute for each timestamp
        for ts in tqdm(timestamps, desc="Cross-sectional"):
            # Gather values for all symbols at this timestamp
            volumes = []
            returns = []
            symbols_at_ts = []

            for symbol, df in all_features.items():
                row = df[df['timestamp'] == ts]
                if len(row) > 0:
                    volumes.append(row['f05_volume_norm'].values[0])
                    returns.append(row['f10_return_288bar'].values[0])
                    symbols_at_ts.append(symbol)

            if len(symbols_at_ts) < 2:
                continue

            volumes = np.array(volumes)
            returns = np.array(returns)

            # Compute ranks
            rank_volume = pd.Series(volumes).rank(pct=True).values
            rank_return = pd.Series(returns).rank(pct=True).values

            # Z-scores
            zscore = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Relative strength (return vs mean)
            rel_strength = returns / (np.abs(returns).mean() + 1e-8)

            # Beta to market (use BTC as market proxy if available)
            beta = np.ones_like(returns)
            if 'BTCUSDT' in symbols_at_ts:
                btc_idx = symbols_at_ts.index('BTCUSDT')
                btc_return = returns[btc_idx]
                beta = returns / (btc_return + 1e-8)

            # Update features
            for i, symbol in enumerate(symbols_at_ts):
                df = all_features[symbol]
                mask = df['timestamp'] == ts

                df.loc[mask, 'f29_rank_volume'] = rank_volume[i]
                df.loc[mask, 'f30_rank_return'] = rank_return[i]
                df.loc[mask, 'f31_zscore'] = zscore[i]
                df.loc[mask, 'f32_rel_strength'] = rel_strength[i]
                df.loc[mask, 'f33_beta'] = beta[i]

        logger.info("  Cross-sectional features computed")

        return all_features

    def save_mock_data(
        self,
        all_features: Dict[str, pd.DataFrame],
        output_dir: Path
    ):
        """
        Save mock data in expected format.

        Args:
            all_features: Dict of {symbol: features_df}
            output_dir: Output directory
        """
        logger.info(f"Saving mock data to {output_dir}...")

        # Create directories
        (output_dir / 'features' / 'mock_features').mkdir(parents=True, exist_ok=True)

        # Save feature files
        for symbol, df in all_features.items():
            output_path = output_dir / 'features' / 'mock_features' / f'{symbol}_features.parquet'

            # Select only feature columns (f01-f36)
            feature_cols = [col for col in df.columns if col.startswith('f')]
            feature_df = df[['timestamp', 'symbol'] + feature_cols].copy()

            feature_df.to_parquet(output_path, index=False)
            logger.info(f"  Saved {symbol}: {len(feature_df):,} rows, {len(feature_cols)} features")

        # Create universe history (all symbols for entire period)
        universe_data = []
        for symbol in self.symbols:
            universe_data.append({
                'symbol': symbol,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'asset_type': 'crypto',
                'base_currency': symbol.replace('USDT', ''),
                'quote_currency': 'USDT',
            })

        universe_df = pd.DataFrame(universe_data)
        universe_path = output_dir / 'universe' / 'universe_history_mock.parquet'
        universe_path.parent.mkdir(parents=True, exist_ok=True)
        universe_df.to_parquet(universe_path, index=False)

        logger.info(f"  Saved universe: {len(universe_df)} symbols")
        logger.info("✓ Mock data generation complete!")

    def generate(self, output_dir: Path):
        """
        Main generation pipeline.

        Args:
            output_dir: Output directory
        """
        logger.info("=" * 60)
        logger.info("MOCK DATA GENERATION")
        logger.info("=" * 60)

        # Step 1: Load OHLCV for all symbols
        all_features = {}

        for symbol in self.symbols:
            ohlcv = self.load_ohlcv(symbol)

            if ohlcv.empty:
                logger.warning(f"Skipping {symbol} (no data)")
                continue

            # Step 2: Compute features
            features = self.compute_features(ohlcv, symbol)
            all_features[symbol] = features

        if not all_features:
            logger.error("No features generated for any symbol!")
            return

        # Step 3: Compute cross-sectional features
        all_features = self.compute_cross_sectional_features(all_features)

        # Step 4: Save data
        self.save_mock_data(all_features, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Generate mock training data')
    parser.add_argument('--n-assets', type=int, default=10,
                        help='Number of assets (default: 10)')
    parser.add_argument('--start-date', type=str, default='2021-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2021-02-28',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str,
                        default='/home/work/data/stair-local/test_mock',
                        help='Output directory')
    parser.add_argument('--data-dir', type=str,
                        default='/home/work/data/stair-local',
                        help='Base data directory')

    args = parser.parse_args()

    # Select top N assets
    symbols = DEFAULT_ASSETS[:args.n_assets]

    logger.info(f"Generating mock data for {len(symbols)} assets")
    logger.info(f"  Symbols: {', '.join(symbols)}")
    logger.info(f"  Period: {args.start_date} to {args.end_date}")
    logger.info(f"  Output: {args.output}")

    # Generate
    generator = MockDataGenerator(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=Path(args.data_dir),
    )

    generator.generate(Path(args.output))

    logger.info("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
