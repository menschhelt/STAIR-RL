"""
Factor Loading Cache System
Caches factor loadings to avoid redundant calculations
"""

import pandas as pd
from typing import Dict, Optional
from datetime import datetime


class FactorCache:
    """
    Time-based cache for factor loadings

    Features:
    - Automatic cache invalidation based on time
    - Per-pair factor loadings storage
    - Scheduled refresh

    Usage:
        cache = FactorCache(update_frequency_minutes=60)

        # Check if update needed
        if cache.should_update(current_time):
            # Calculate new factor loadings
            for pair in pairs:
                loadings = calculate_factor_loadings(pair)
                cache.set(current_time, pair, loadings)
            cache.mark_updated(current_time)

        # Get cached loadings
        loadings = cache.get(current_time, pair)
    """

    def __init__(self, update_frequency_minutes: int = 60):
        """
        Args:
            update_frequency_minutes: How often to recalculate factors (default: 1 hour)
        """
        self.update_frequency = update_frequency_minutes
        self.cache = {}  # {timestamp: {pair: DataFrame}}
        self.last_update = None
        self.update_count = 0

    def should_update(self, current_time: pd.Timestamp) -> bool:
        """
        Check if cache needs to be updated

        Args:
            current_time: Current timestamp

        Returns:
            True if update is needed
        """
        if self.last_update is None:
            return True

        # Calculate time difference in minutes
        time_diff = (current_time - self.last_update).total_seconds() / 60

        return time_diff >= self.update_frequency

    def get(self, timestamp: pd.Timestamp, pair: str) -> Optional[pd.DataFrame]:
        """
        Get cached factor loadings

        Args:
            timestamp: Timestamp key
            pair: Trading pair

        Returns:
            Factor loadings DataFrame or None if not cached
        """
        # Use last_update as key (all pairs share same timestamp)
        if self.last_update is None:
            return None

        return self.cache.get(self.last_update, {}).get(pair)

    def set(self, timestamp: pd.Timestamp, pair: str, factor_loadings: pd.DataFrame):
        """
        Store factor loadings in cache

        Args:
            timestamp: Timestamp key
            pair: Trading pair
            factor_loadings: Factor loadings DataFrame
        """
        if timestamp not in self.cache:
            self.cache[timestamp] = {}

        self.cache[timestamp][pair] = factor_loadings.copy()

    def mark_updated(self, timestamp: pd.Timestamp):
        """
        Mark cache as updated at given timestamp

        Args:
            timestamp: Update timestamp
        """
        self.last_update = timestamp
        self.update_count += 1

        # Clean old cache entries (keep only last 2)
        if len(self.cache) > 2:
            # Sort by timestamp and keep latest 2
            sorted_times = sorted(self.cache.keys())
            for old_time in sorted_times[:-2]:
                del self.cache[old_time]

    def clear(self):
        """Clear all cache"""
        self.cache = {}
        self.last_update = None
        self.update_count = 0

    def get_stats(self) -> Dict:
        """
        Get cache statistics

        Returns:
            Dictionary with cache stats
        """
        total_pairs = sum(len(pairs) for pairs in self.cache.values())

        return {
            'update_frequency_minutes': self.update_frequency,
            'last_update': self.last_update,
            'update_count': self.update_count,
            'cached_timestamps': len(self.cache),
            'total_cached_pairs': total_pairs,
        }

    def __repr__(self):
        stats = self.get_stats()
        return (
            f"FactorCache("
            f"freq={stats['update_frequency_minutes']}min, "
            f"updates={stats['update_count']}, "
            f"cached_pairs={stats['total_cached_pairs']})"
        )


# Example usage
if __name__ == '__main__':
    import numpy as np

    # Create cache with 60-minute update frequency
    cache = FactorCache(update_frequency_minutes=60)

    # Simulate time series
    timestamps = pd.date_range('2024-01-01', periods=10, freq='30min')
    pairs = ['BTC/USDT:USDT', 'ETH/USDT:USDT']

    for i, ts in enumerate(timestamps):
        if cache.should_update(ts):
            print(f"\n⏰ Update at {ts}")

            # Simulate factor calculation
            for pair in pairs:
                # Create dummy factor loadings
                factor_loadings = pd.DataFrame({
                    'momentum': np.random.randn(5),
                    'volatility': np.random.randn(5),
                }, index=pd.date_range(ts, periods=5, freq='1h'))

                cache.set(ts, pair, factor_loadings)

            cache.mark_updated(ts)
        else:
            print(f"⏭️  Skip (cached) at {ts}")

        # Try to get cached data
        btc_loadings = cache.get(ts, 'BTC/USDT:USDT')
        if btc_loadings is not None:
            print(f"   ✅ Got cached BTC loadings: {btc_loadings.shape}")

    print(f"\n{cache}")
    print(f"Stats: {cache.get_stats()}")
