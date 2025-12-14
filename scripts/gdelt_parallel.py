#!/usr/bin/env python3
"""
GDELT Parallel Collection - Split by quarter for maximum parallelization.
24 processes (2020 Q1 ~ 2025 Q4)
"""
import asyncio
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, '/home/work/RL/stair-local')

from collectors.gdelt_local import GDELTLocalCollector

# Define quarters
QUARTERS = [
    # 2020
    ("2020-01-01", "2020-03-31", "2020_Q1"),
    ("2020-04-01", "2020-06-30", "2020_Q2"),
    ("2020-07-01", "2020-09-30", "2020_Q3"),
    ("2020-10-01", "2020-12-31", "2020_Q4"),
    # 2021
    ("2021-01-01", "2021-03-31", "2021_Q1"),
    ("2021-04-01", "2021-06-30", "2021_Q2"),
    ("2021-07-01", "2021-09-30", "2021_Q3"),
    ("2021-10-01", "2021-12-31", "2021_Q4"),
    # 2022
    ("2022-01-01", "2022-03-31", "2022_Q1"),
    ("2022-04-01", "2022-06-30", "2022_Q2"),
    ("2022-07-01", "2022-09-30", "2022_Q3"),
    ("2022-10-01", "2022-12-31", "2022_Q4"),
    # 2023
    ("2023-01-01", "2023-03-31", "2023_Q1"),
    ("2023-04-01", "2023-06-30", "2023_Q2"),
    ("2023-07-01", "2023-09-30", "2023_Q3"),
    ("2023-10-01", "2023-12-31", "2023_Q4"),
    # 2024
    ("2024-01-01", "2024-03-31", "2024_Q1"),
    ("2024-04-01", "2024-06-30", "2024_Q2"),
    ("2024-07-01", "2024-09-30", "2024_Q3"),
    ("2024-10-01", "2024-12-31", "2024_Q4"),
    # 2025
    ("2025-01-01", "2025-03-31", "2025_Q1"),
    ("2025-04-01", "2025-06-30", "2025_Q2"),
    ("2025-07-01", "2025-09-30", "2025_Q3"),
    ("2025-10-01", "2025-11-30", "2025_Q4"),  # Ends Nov 30
]

def run_quarter(args):
    """Run collection for a single quarter."""
    start_str, end_str, quarter_name = args
    
    start = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_str, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    
    print(f"[{quarter_name}] Starting: {start_str} to {end_str}")
    
    async def collect():
        collector = GDELTLocalCollector()
        total = await collector.collect(start, end, scrape_content=True)
        await collector.close_session()
        return total
    
    try:
        total = asyncio.run(collect())
        print(f"[{quarter_name}] DONE! {total:,} records")
        return (quarter_name, total, None)
    except Exception as e:
        print(f"[{quarter_name}] ERROR: {e}")
        return (quarter_name, 0, str(e))

def main():
    print("=" * 60)
    print("GDELT PARALLEL COLLECTION - 24 Quarters")
    print("=" * 60)
    print(f"Total quarters: {len(QUARTERS)}")
    print(f"CPU cores available: {mp.cpu_count()}")
    print()
    
    # Use 20 workers (leave some CPU headroom)
    num_workers = min(20, len(QUARTERS))
    print(f"Using {num_workers} parallel workers")
    print()
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(run_quarter, QUARTERS)
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    total_records = 0
    errors = []
    for quarter_name, count, error in results:
        if error:
            errors.append((quarter_name, error))
            print(f"  {quarter_name}: ERROR - {error}")
        else:
            total_records += count
            print(f"  {quarter_name}: {count:,} records")
    
    print()
    print(f"Total records: {total_records:,}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\nFailed quarters:")
        for q, e in errors:
            print(f"  - {q}: {e}")

if __name__ == "__main__":
    main()
