"""
Base Local Collector - Parquet-based data collection without PostgreSQL.

Replaces the PostgreSQL-based BaseCollector from data-collector with local
Parquet file storage for backtesting research.
"""

import json
import asyncio
import aiohttp
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Literal
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import DATA_DIR


class BaseLocalCollector(ABC):
    """
    Abstract base class for local Parquet-based data collectors.

    Key differences from PostgreSQL-based BaseCollector:
    - No database connection required
    - Parquet file writing with configurable partitioning
    - Local JSON-based checkpointing
    - Async HTTP client for API calls

    Partition strategies:
    - 'monthly': binance_futures_YYYYMM.parquet
    - 'weekly': nostr_YYYYWW.parquet (ISO week number)
    """

    def __init__(
        self,
        data_dir: Path,
        partition_strategy: Literal['monthly', 'weekly'] = 'monthly',
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize the local collector.

        Args:
            data_dir: Directory to store Parquet files
            partition_strategy: 'monthly' or 'weekly' partitioning
            checkpoint_dir: Directory for checkpoint files (default: data_dir/.checkpoints)
        """
        self.data_dir = Path(data_dir)
        self.partition_strategy = partition_strategy
        self.checkpoint_dir = checkpoint_dir or self.data_dir / '.checkpoints'

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # HTTP session (lazy initialization)
        self._session: Optional[aiohttp.ClientSession] = None

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the collector."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    @property
    @abstractmethod
    def collector_name(self) -> str:
        """Return the collector name for file naming and logging."""
        pass

    @property
    @abstractmethod
    def schema(self) -> pa.Schema:
        """Return the PyArrow schema for the Parquet files."""
        pass

    @abstractmethod
    async def collect(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """
        Collect data for the specified date range.

        Args:
            start_date: Start of collection period
            end_date: End of collection period

        Returns:
            Number of records collected
        """
        pass

    # ========== HTTP Session Management ==========

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close_session(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()

    # ========== Partition Key Generation ==========

    def get_partition_key(self, dt: datetime) -> str:
        """
        Generate partition key based on strategy.

        Args:
            dt: Datetime to generate key for

        Returns:
            Partition key string (e.g., '202401' or '202401W01')
        """
        if self.partition_strategy == 'monthly':
            return dt.strftime('%Y%m')
        elif self.partition_strategy == 'weekly':
            # ISO week format: YYYYWWW (e.g., 2024W01)
            iso_year, iso_week, _ = dt.isocalendar()
            return f"{iso_year}W{iso_week:02d}"
        else:
            raise ValueError(f"Unknown partition strategy: {self.partition_strategy}")

    def get_partition_date_range(self, partition_key: str) -> tuple[datetime, datetime]:
        """
        Get the date range for a partition key.

        Args:
            partition_key: Partition key (e.g., '202401' or '2024W01')

        Returns:
            (start_date, end_date) tuple
        """
        if self.partition_strategy == 'monthly':
            year = int(partition_key[:4])
            month = int(partition_key[4:6])
            start = datetime(year, month, 1)
            # Get last day of month
            if month == 12:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, month + 1, 1)
            return start, end

        elif self.partition_strategy == 'weekly':
            # Parse YYYYWWW format
            year = int(partition_key[:4])
            week = int(partition_key[5:7])
            # ISO week 1 starts on the first Thursday
            start = datetime.strptime(f'{year} {week} 1', '%G %V %u')
            end = start + timedelta(days=7)
            return start, end

        else:
            raise ValueError(f"Unknown partition strategy: {self.partition_strategy}")

    def get_file_path(self, partition_key: str) -> Path:
        """
        Get the file path for a partition.

        Args:
            partition_key: Partition key

        Returns:
            Full path to the Parquet file
        """
        filename = f"{self.collector_name}_{partition_key}.parquet"
        return self.data_dir / filename

    # ========== Parquet File Operations ==========

    def write_parquet(
        self,
        df: pd.DataFrame,
        partition_key: str,
        mode: Literal['append', 'overwrite'] = 'append'
    ):
        """
        Write DataFrame to partitioned Parquet file.

        Args:
            df: DataFrame to write
            partition_key: Partition key for file naming
            mode: 'append' or 'overwrite'
        """
        if df.empty:
            self.logger.warning(f"Empty DataFrame, skipping write for {partition_key}")
            return

        file_path = self.get_file_path(partition_key)

        # Convert DataFrame to PyArrow Table
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)

        if mode == 'overwrite' or not file_path.exists():
            # Write new file
            pq.write_table(table, file_path, compression='snappy')
            self.logger.info(f"Wrote {len(df)} records to {file_path}")

        elif mode == 'append':
            # Read existing data and append
            existing_table = pq.read_table(file_path)
            combined_table = pa.concat_tables([existing_table, table])

            # Remove duplicates if primary key columns exist
            combined_df = combined_table.to_pandas()
            pk_cols = self._get_primary_key_columns()
            if pk_cols:
                combined_df = combined_df.drop_duplicates(subset=pk_cols, keep='last')
                combined_table = pa.Table.from_pandas(
                    combined_df, schema=self.schema, preserve_index=False
                )

            pq.write_table(combined_table, file_path, compression='snappy')
            self.logger.info(f"Appended {len(df)} records to {file_path}")

    def read_parquet(
        self,
        partition_key: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Read Parquet files.

        Args:
            partition_key: Specific partition to read
            start_date: Start date for multi-partition read
            end_date: End date for multi-partition read
            columns: Specific columns to read

        Returns:
            DataFrame with data
        """
        if partition_key:
            file_path = self.get_file_path(partition_key)
            if file_path.exists():
                return pq.read_table(file_path, columns=columns).to_pandas()
            else:
                return pd.DataFrame()

        # Multi-partition read
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date required for multi-partition read")

        dfs = []
        current = start_date
        seen_keys = set()

        while current <= end_date:
            key = self.get_partition_key(current)
            if key not in seen_keys:
                seen_keys.add(key)
                file_path = self.get_file_path(key)
                if file_path.exists():
                    df = pq.read_table(file_path, columns=columns).to_pandas()
                    dfs.append(df)

            # Move to next partition
            if self.partition_strategy == 'monthly':
                if current.month == 12:
                    current = datetime(current.year + 1, 1, 1)
                else:
                    current = datetime(current.year, current.month + 1, 1)
            else:  # weekly
                current = current + timedelta(days=7)

        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            # Filter to exact date range
            if 'timestamp' in result.columns:
                result = result[
                    (result['timestamp'] >= start_date) &
                    (result['timestamp'] <= end_date)
                ]
            elif 'created_at' in result.columns:
                result = result[
                    (result['created_at'] >= start_date) &
                    (result['created_at'] <= end_date)
                ]
            return result

        return pd.DataFrame()

    def _get_primary_key_columns(self) -> List[str]:
        """
        Return primary key columns for deduplication.
        Override in subclass if needed.
        """
        return []

    # ========== Checkpoint Management ==========

    def get_checkpoint_path(self) -> Path:
        """Get the checkpoint file path."""
        return self.checkpoint_dir / f"{self.collector_name}_checkpoint.json"

    def load_checkpoint(self) -> Optional[datetime]:
        """
        Load the last successful checkpoint.

        Returns:
            Last checkpoint datetime or None if no checkpoint exists
        """
        checkpoint_path = self.get_checkpoint_path()
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                return datetime.fromisoformat(data['last_timestamp'])
        return None

    def save_checkpoint(self, timestamp: datetime):
        """
        Save checkpoint after successful collection.

        Args:
            timestamp: Timestamp to save as checkpoint
        """
        checkpoint_path = self.get_checkpoint_path()
        with open(checkpoint_path, 'w') as f:
            json.dump({
                'last_timestamp': timestamp.isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'collector': self.collector_name,
            }, f, indent=2)
        self.logger.debug(f"Saved checkpoint: {timestamp}")

    def clear_checkpoint(self):
        """Clear the checkpoint file."""
        checkpoint_path = self.get_checkpoint_path()
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            self.logger.info("Checkpoint cleared")

    # ========== Utility Methods ==========

    def list_partitions(self) -> List[str]:
        """
        List all existing partition keys.

        Returns:
            List of partition keys sorted by date
        """
        pattern = f"{self.collector_name}_*.parquet"
        files = list(self.data_dir.glob(pattern))

        keys = []
        for f in files:
            # Extract partition key from filename
            name = f.stem  # Remove .parquet
            prefix = f"{self.collector_name}_"
            if name.startswith(prefix):
                key = name[len(prefix):]
                keys.append(key)

        return sorted(keys)

    def get_data_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collected data.

        Returns:
            Dictionary with stats
        """
        partitions = self.list_partitions()
        total_records = 0
        total_size = 0

        for key in partitions:
            file_path = self.get_file_path(key)
            if file_path.exists():
                metadata = pq.read_metadata(file_path)
                total_records += metadata.num_rows
                total_size += file_path.stat().st_size

        return {
            'collector': self.collector_name,
            'partitions': len(partitions),
            'total_records': total_records,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'first_partition': partitions[0] if partitions else None,
            'last_partition': partitions[-1] if partitions else None,
        }


class SimpleRateLimiter:
    """
    Simple rate limiter without Redis dependency.
    Uses in-memory sliding window.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until rate limit allows a new request."""
        async with self._lock:
            now = asyncio.get_event_loop().time()

            # Remove old requests outside window
            cutoff = now - self.window_seconds
            self.requests = [t for t in self.requests if t > cutoff]

            # Wait if at limit
            while len(self.requests) >= self.max_requests:
                # Wait for oldest request to expire
                sleep_time = self.requests[0] - cutoff + 0.1
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                now = asyncio.get_event_loop().time()
                cutoff = now - self.window_seconds
                self.requests = [t for t in self.requests if t > cutoff]

            # Record this request
            self.requests.append(now)
