#!/usr/bin/env python3
"""
Feature Building Script.

Pre-calculates all features for efficient training:
1. Alpha 101/191 factors
2. Risk factor loadings (beta_MKT, beta_SMB, beta_MOM, alpha_resid)
3. Sentiment scores (if available)
4. Universe history

This is Phase 1 of the 2-phase training approach:
- Phase 1 (this script): Heavy computation, results cached
- Phase 2 (training): Simple lookup, arithmetic only

Usage:
    python scripts/build_features.py --start 2021-01-01 --end 2024-12-31
    python scripts/build_features.py --alphas-only
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config, DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_universe_history(start_date: str, end_date: str, config: Config):
    """Build universe history (Top 20 by volume daily)."""
    from universe.manager import UniverseManager
    from datetime import datetime

    logger.info("=" * 60)
    logger.info("Building universe history")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    # Load Binance OHLCV data
    binance_dir = DATA_DIR / 'binance'
    if not binance_dir.exists():
        logger.error(f"Binance data not found at {binance_dir}")
        return

    universe_mgr = UniverseManager(
        binance_data_dir=binance_dir,
        universe_dir=DATA_DIR / 'universe',
        config=config.universe,
    )

    # Parse dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

    # Build universe history
    universe_history = universe_mgr.build_universe_history(
        start_date=start_dt,
        end_date=end_dt,
    )

    logger.info(f"Universe history saved: {len(universe_history)} records")


def build_alpha_features(start_date: str, end_date: str, config: Config):
    """Calculate Alpha 101/191 for all symbols."""
    from features.alpha_adapter import AlphaAdapter
    from features.alpha_calculator import AlphaCalculator
    from datetime import datetime, timezone

    logger.info("=" * 60)
    logger.info("Building alpha features")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    # Load OHLCV data
    binance_dir = DATA_DIR / 'binance'
    if not binance_dir.exists():
        logger.error(f"Binance data not found at {binance_dir}")
        return

    # Get all symbols from data
    parquet_files = list(binance_dir.glob('*.parquet'))
    if not parquet_files:
        logger.error("No parquet files found in Binance directory")
        return

    adapter = AlphaAdapter()
    calculator = AlphaCalculator(
        binance_data_dir=binance_dir,
        cache_dir=DATA_DIR / 'features' / 'alpha_cache',
        alpha_adapter=adapter,
    )

    # Parse dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    # Process all symbols
    cache_paths = calculator.precalculate_all(
        start_date=start_dt,
        end_date=end_dt,
        parallel=True,
    )

    logger.info(f"Alpha features completed: {len(cache_paths)} symbols cached")


def build_risk_factors(start_date: str, end_date: str, config: Config):
    """Calculate risk factor loadings."""
    from factors.loading_matrix import LoadingMatrixCalculator
    from factors.crypto_factors import CryptoFactorCalculator

    logger.info("=" * 60)
    logger.info("Building risk factor loadings")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    # Initialize calculators
    factor_calc = CryptoFactorCalculator()
    loading_calc = LoadingMatrixCalculator()

    # Load OHLCV data
    binance_dir = DATA_DIR / 'binance'
    if not binance_dir.exists():
        logger.error(f"Binance data not found at {binance_dir}")
        return

    # Load universe
    universe_path = DATA_DIR / 'universe' / 'universe_history.parquet'
    if not universe_path.exists():
        logger.error("Universe history not found. Run build_universe_history first.")
        return

    universe_df = pd.read_parquet(universe_path)

    # Calculate factor returns (market-wide)
    factor_returns = factor_calc.calculate_factor_returns(
        data_dir=binance_dir,
        start_date=start_date,
        end_date=end_date,
    )

    # Save factor returns
    output_dir = DATA_DIR / 'features' / 'factor_cache'
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_returns.to_parquet(output_dir / 'factor_returns.parquet')
    logger.info(f"Factor returns saved: {factor_returns.shape}")

    # Calculate loading matrix for each symbol
    for symbol in universe_df['symbol'].unique():
        if pd.isna(symbol):
            continue

        try:
            # Load symbol returns
            symbol_returns = load_symbol_returns(binance_dir, symbol, start_date, end_date)
            if symbol_returns is None:
                continue

            # Calculate loadings
            loadings = loading_calc.calculate_loadings(
                asset_returns=symbol_returns,
                factor_returns=factor_returns,
            )

            # Calculate residual alpha
            residual = loading_calc.calculate_residual(
                asset_returns=symbol_returns,
                factor_returns=factor_returns,
                loadings=loadings,
            )

            # Save
            loading_df = pd.DataFrame({
                'beta_MKT': loadings.get('CMKT', 0),
                'beta_SMB': loadings.get('CSMB', 0),
                'beta_MOM': loadings.get('CMOM', 0),
                'alpha_resid': residual,
            })
            loading_df.to_parquet(output_dir / f'{symbol}_loadings.parquet')

        except Exception as e:
            logger.debug(f"Failed to calculate loadings for {symbol}: {e}")
            continue

    logger.info("Risk factor loadings completed")


def load_symbol_returns(data_dir: Path, symbol: str, start_date: str, end_date: str):
    """Load returns for a specific symbol."""
    # Find parquet files
    parquet_files = list(data_dir.glob('*.parquet'))

    returns_list = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(
                pf,
                filters=[('symbol', '==', symbol)],
            )
            if len(df) > 0:
                returns_list.append(df)
        except Exception:
            continue

    if not returns_list:
        return None

    # Combine and calculate returns
    combined = pd.concat(returns_list).sort_values('timestamp')
    combined = combined.drop_duplicates(subset=['timestamp'])

    # Filter by date range
    combined = combined[
        (combined['timestamp'] >= start_date) &
        (combined['timestamp'] <= end_date)
    ]

    if len(combined) < 20:
        return None

    # Calculate returns
    combined['return'] = combined['close'].pct_change()

    return combined.set_index('timestamp')['return'].dropna()


def build_sentiment_features(start_date: str, end_date: str, config: Config):
    """Calculate sentiment features from GDELT/Nostr."""
    from text_processing.finbert_processor import FinBERTProcessor
    from text_processing.cryptobert_processor import CryptoBERTProcessor
    from text_processing.text_aggregator import TextAggregator

    logger.info("=" * 60)
    logger.info("Building sentiment features")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    # Check if text data exists
    gdelt_dir = DATA_DIR / 'gdelt'
    nostr_dir = DATA_DIR / 'nostr'

    output_dir = DATA_DIR / 'features' / 'sentiment_cache'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process GDELT with FinBERT
    if gdelt_dir.exists():
        logger.info("Processing GDELT with FinBERT...")
        finbert = FinBERTProcessor(
            model_name=config.text_processing.finbert_model,
            batch_size=config.text_processing.batch_size,
        )
        finbert.process_directory(
            input_dir=gdelt_dir,
            output_dir=output_dir,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        logger.warning(f"GDELT data not found at {gdelt_dir}")

    # Process Nostr with CryptoBERT
    if nostr_dir.exists():
        logger.info("Processing Nostr with CryptoBERT...")
        cryptobert = CryptoBERTProcessor(
            model_name=config.text_processing.cryptobert_model,
            batch_size=config.text_processing.batch_size,
        )
        cryptobert.process_directory(
            input_dir=nostr_dir,
            output_dir=output_dir,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        logger.warning(f"Nostr data not found at {nostr_dir}")

    # Aggregate sentiment by time window
    logger.info("Aggregating sentiment features...")
    aggregator = TextAggregator(
        window=config.text_processing.aggregation_window,
    )
    aggregator.aggregate(
        input_dir=output_dir,
        output_path=output_dir / 'aggregated_sentiment.parquet',
    )

    logger.info("Sentiment features completed")


def build_all_features(start_date: str, end_date: str, config: Config):
    """Build all features in correct order."""
    logger.info("=" * 60)
    logger.info("Building ALL features")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    # 1. Universe history (required first)
    logger.info("\n[1/5] Building universe history...")
    build_universe_history(start_date, end_date, config)

    # 2. Alpha features
    logger.info("\n[2/4] Building alpha features...")
    build_alpha_features(start_date, end_date, config)

    # 3. Risk factor loadings
    logger.info("\n[3/4] Building risk factor loadings...")
    build_risk_factors(start_date, end_date, config)

    # 5. Sentiment (optional, can be slow)
    # logger.info("\n[5/5] Building sentiment features...")
    # build_sentiment_features(start_date, end_date, config)

    logger.info("\n" + "=" * 60)
    logger.info("All features completed")
    logger.info("=" * 60)

    # Print summary
    print_feature_summary()


def print_feature_summary():
    """Print summary of cached features."""
    logger.info("\nFeature Cache Summary:")
    logger.info("-" * 40)

    dirs = {
        'Universe': DATA_DIR / 'universe',
        'Alpha Cache': DATA_DIR / 'features' / 'alpha_cache',
        'Factor Cache': DATA_DIR / 'features' / 'factor_cache',
        'Sentiment Cache': DATA_DIR / 'features' / 'sentiment_cache',
    }

    for name, path in dirs.items():
        if path.exists():
            files = list(path.glob('*.parquet'))
            total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
            logger.info(f"  {name}: {len(files)} files, {total_size:.1f} MB")
        else:
            logger.info(f"  {name}: Not found")


def main():
    parser = argparse.ArgumentParser(
        description='Build features for STAIR-RL training'
    )
    parser.add_argument(
        '--start', type=str, default='2021-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end', type=str, default=None,
        help='End date (YYYY-MM-DD), defaults to today'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file'
    )

    # Feature selection flags
    parser.add_argument('--all', action='store_true', help='Build all features')
    parser.add_argument('--universe-only', action='store_true', help='Build universe only')
    parser.add_argument('--alphas-only', action='store_true', help='Build alphas only')
    parser.add_argument('--factors-only', action='store_true', help='Build risk factors only')
    parser.add_argument('--sentiment-only', action='store_true', help='Build sentiment only')

    args = parser.parse_args()

    # Set end date to today if not specified
    if args.end is None:
        args.end = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # Load config
    if args.config:
        config = Config.from_yaml(Path(args.config))
    else:
        config = Config()

    # Create output directories
    (DATA_DIR / 'features' / 'alpha_cache').mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'features' / 'factor_cache').mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'features' / 'sentiment_cache').mkdir(parents=True, exist_ok=True)

    # Run selected feature builders
    if args.universe_only:
        build_universe_history(args.start, args.end, config)
    elif args.alphas_only:
        build_alpha_features(args.start, args.end, config)
    elif args.factors_only:
        build_risk_factors(args.start, args.end, config)
    elif args.sentiment_only:
        build_sentiment_features(args.start, args.end, config)
    else:
        # Default: build all
        build_all_features(args.start, args.end, config)


if __name__ == '__main__':
    main()
