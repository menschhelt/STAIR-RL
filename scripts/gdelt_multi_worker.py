#!/usr/bin/env python3
"""
GDELT Multi-Worker Collection Script

Rate limit 검증 결과:
- SimpleRateLimiter는 프로세스별 (in-memory)
- GDELT 스크래핑은 각 기사 URL별로 발생 → 프로세스 간 충돌 없음
- 안전하게 병렬화 가능

전략: 24개 분기 × 20 워커 → 최대 20개 동시 실행
"""
import sys
sys.path.insert(0, '/home/work/RL/stair-local')

import asyncio
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path
import logging
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [Worker-%(process)d] %(message)s',
    datefmt='%H:%M:%S'
)

# 24개 분기 정의 (2020-Q1 ~ 2025-Q4)
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
    ("2025-10-01", "2025-11-30", "2025_Q4"),  # 11월 30일까지
]


def run_quarter(args):
    """단일 분기 수집 (별도 프로세스에서 실행)"""
    start_str, end_str, quarter_name = args

    # 이벤트 루프 생성
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        from collectors.gdelt_local import GDELTLocalCollector

        start = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_str, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        )

        logging.info(f"[{quarter_name}] Starting: {start_str} ~ {end_str}")
        start_time = time.time()

        collector = GDELTLocalCollector()
        total = loop.run_until_complete(
            collector.collect(start, end, scrape_content=True)
        )
        loop.run_until_complete(collector.close_session())

        elapsed = time.time() - start_time
        logging.info(f"[{quarter_name}] Complete! {total:,} articles in {elapsed/60:.1f}min")

        return (quarter_name, total, elapsed)

    except Exception as e:
        logging.error(f"[{quarter_name}] ERROR: {e}")
        return (quarter_name, 0, 0)
    finally:
        loop.close()


def main():
    """메인 함수"""
    print("=" * 60)
    print("GDELT Multi-Worker Collection")
    print("=" * 60)
    print(f"Total quarters: {len(QUARTERS)}")

    # CPU 코어 수 기반 워커 수 결정 (최대 20)
    num_workers = min(20, mp.cpu_count(), len(QUARTERS))
    print(f"Workers: {num_workers}")
    print("=" * 60)

    start_time = time.time()

    # 멀티프로세싱 풀 실행
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(run_quarter, QUARTERS)

    total_elapsed = time.time() - start_time

    # 결과 요약
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_articles = 0
    for quarter_name, count, elapsed in results:
        status = "OK" if count > 0 else "FAIL"
        print(f"  {quarter_name}: {count:>8,} articles [{status}]")
        total_articles += count

    print("-" * 60)
    print(f"  TOTAL: {total_articles:,} articles")
    print(f"  TIME: {total_elapsed/3600:.1f} hours")
    print("=" * 60)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
