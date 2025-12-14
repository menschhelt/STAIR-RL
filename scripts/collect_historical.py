#!/usr/bin/env python3
"""
Historical Data Collection Script.

Collects historical data from all sources:
1. Binance Futures OHLCV (5-minute bars)
2. FRED macroeconomic indicators
3. yfinance market data (VIX, S&P 500, etc.)
4. GDELT news articles (optional)
5. Nostr posts (optional)

Usage:
    python scripts/collect_historical.py --start 2021-01-01 --end 2024-12-31
    python scripts/collect_historical.py --source binance --start 2024-01-01
    python scripts/collect_historical.py --all
"""

import asyncio
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config, DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def collect_binance(start_date: str, end_date: str, config: Config):
    """Collect Binance Futures OHLCV data."""
    from collectors.binance_futures import BinanceFuturesCollector

    logger.info("=" * 60)
    logger.info("Starting Binance Futures data collection")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    collector = BinanceFuturesCollector(
        data_dir=DATA_DIR / 'binance',
        config=config.binance,
    )

    # Get universe symbols
    from universe.manager import UniverseManager
    universe_mgr = UniverseManager(config.universe)

    # Get symbols to collect
    # For historical backfill, we collect all USDT perpetual futures
    symbols = await collector.get_all_perpetual_symbols()
    logger.info(f"Found {len(symbols)} perpetual symbols")

    # Collect OHLCV data
    await collector.collect_historical(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval=config.binance.interval,
    )

    logger.info("Binance Futures collection completed")


async def collect_fred(start_date: str, end_date: str, config: Config):
    """Collect FRED macroeconomic data."""
    from collectors.fred_local import FREDLocalCollector

    logger.info("=" * 60)
    logger.info("Starting FRED data collection")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    collector = FREDLocalCollector(
        data_dir=DATA_DIR / 'macro',
        config=config.fred,
    )

    await collector.collect_all_indicators(
        start_date=start_date,
        end_date=end_date,
    )

    logger.info("FRED collection completed")


async def collect_yfinance(start_date: str, end_date: str, config: Config):
    """Collect yfinance market data."""
    from collectors.yfinance_local import YFinanceLocalCollector

    logger.info("=" * 60)
    logger.info("Starting yfinance data collection")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    collector = YFinanceLocalCollector(
        data_dir=DATA_DIR / 'macro',
        config=config.yfinance,
    )

    await collector.collect_all_tickers(
        start_date=start_date,
        end_date=end_date,
    )

    logger.info("yfinance collection completed")


async def collect_gdelt(start_date: str, end_date: str, config: Config):
    """Collect GDELT news articles."""
    from collectors.gdelt_local import GDELTLocalCollector

    logger.info("=" * 60)
    logger.info("Starting GDELT data collection")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    collector = GDELTLocalCollector(
        data_dir=DATA_DIR / 'gdelt',
        config=config.gdelt,
    )

    await collector.collect_historical(
        start_date=start_date,
        end_date=end_date,
    )

    logger.info("GDELT collection completed")


async def collect_nostr(start_date: str, end_date: str, config: Config):
    """Collect Nostr posts."""
    from collectors.nostr_local import NostrLocalCollector

    logger.info("=" * 60)
    logger.info("Starting Nostr data collection")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    collector = NostrLocalCollector(
        data_dir=DATA_DIR / 'nostr',
        config=config.nostr,
    )

    await collector.collect_historical(
        start_date=start_date,
        end_date=end_date,
        kinds=config.nostr.kinds,
    )

    logger.info("Nostr collection completed")


async def collect_all(start_date: str, end_date: str, config: Config):
    """Collect all data sources."""
    logger.info("=" * 60)
    logger.info("Starting FULL data collection")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    # Collect in order of importance
    # 1. Binance (critical for backtesting)
    await collect_binance(start_date, end_date, config)

    # 2. Macro data (FRED + yfinance)
    await collect_fred(start_date, end_date, config)
    await collect_yfinance(start_date, end_date, config)

    # 3. Text data (optional, slower)
    # await collect_gdelt(start_date, end_date, config)
    # await collect_nostr(start_date, end_date, config)

    logger.info("=" * 60)
    logger.info("All data collection completed")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Collect historical data for STAIR-RL backtesting'
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
        '--source', type=str, default='all',
        choices=['all', 'binance', 'fred', 'yfinance', 'gdelt', 'nostr'],
        help='Data source to collect'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file'
    )

    args = parser.parse_args()

    # Set end date to today if not specified
    if args.end is None:
        args.end = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # Load config
    if args.config:
        config = Config.from_yaml(Path(args.config))
    else:
        config = Config()

    # Create data directories
    (DATA_DIR / 'binance').mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'macro').mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'gdelt').mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'nostr').mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'universe').mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'features').mkdir(parents=True, exist_ok=True)

    # Run collection
    source_map = {
        'all': collect_all,
        'binance': collect_binance,
        'fred': collect_fred,
        'yfinance': collect_yfinance,
        'gdelt': collect_gdelt,
        'nostr': collect_nostr,
    }

    collect_fn = source_map[args.source]

    logger.info(f"Data will be saved to: {DATA_DIR}")
    asyncio.run(collect_fn(args.start, args.end, config))


if __name__ == '__main__':
    main()
