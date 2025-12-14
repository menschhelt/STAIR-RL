#!/usr/bin/env python3
"""
Nostr Multi-Relay Collection Script

Rate limit 검증 결과:
- SimpleRateLimiter는 프로세스별 (in-memory)
- 모든 프로세스가 동일 릴레이 접속 시 → 릴레이 과부하
- 해결책: 각 프로세스를 다른 릴레이에 할당

전략: 4개 릴레이 × 기간 분할 → 4배 병렬화
- relay.nostr.band (메인, 과거 데이터 지원)
- relay.damus.io (대형 릴레이)
- nos.lol (인기 릴레이)
- nostr.wine (paid relay, 안정적)

주의: 2022년 12월 이전 데이터는 대부분 없음 (Nostr 초기)
"""
import sys
sys.path.insert(0, '/home/work/RL/stair-local')

import asyncio
import multiprocessing as mp
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(process)d] %(message)s',
    datefmt='%H:%M:%S'
)

# 릴레이별 할당 (무료 릴레이만!)
RELAYS = [
    "wss://relay.nostr.band",   # 과거 데이터 최고
    "wss://relay.damus.io",     # 대형
    "wss://nos.lol",            # 인기
    "wss://purplerelay.com",    # 무료 대형 릴레이
]

# 연도별 분할 (각 릴레이가 1년씩)
PERIODS = [
    # (시작, 종료, 릴레이 인덱스)
    ("2021-01-01", "2021-12-31", 0),  # relay.nostr.band - 최초기 데이터
    ("2022-01-01", "2022-12-31", 1),  # relay.damus.io
    ("2023-01-01", "2023-12-31", 2),  # nos.lol
    ("2024-01-01", "2024-12-31", 3),  # purplerelay.com
    ("2025-01-01", "2025-11-30", 0),  # relay.nostr.band - 최신
]


def run_period(args):
    """단일 기간 수집 (별도 프로세스에서 실행)"""
    start_str, end_str, relay_idx = args
    relay = RELAYS[relay_idx]

    # 이벤트 루프 생성
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        from collectors.nostr_local import NostrLocalCollector
        from config.settings import NostrConfig, RateLimitConfig

        start = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_str, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        )

        # 릴레이 설정 오버라이드 - 백업 없이 자기 릴레이만 사용 (충돌 방지)
        config = NostrConfig(
            relay=relay,
            backup_relays=[],  # 빈 리스트! 다른 릴레이로 폴백하지 않음
            rate_limit=RateLimitConfig(max_requests=60, window_seconds=60),  # 보수적 60req/min
        )

        logging.info(f"[{start_str[:4]}] Starting: {start_str} ~ {end_str}")
        logging.info(f"[{start_str[:4]}] Primary relay: {relay}")
        start_time = time.time()

        collector = NostrLocalCollector(config=config)
        total = loop.run_until_complete(collector.collect(start, end))
        loop.run_until_complete(collector.close_session())

        elapsed = time.time() - start_time
        logging.info(f"[{start_str[:4]}] Complete! {total:,} records in {elapsed/60:.1f}min")

        return (start_str[:4], relay, total, elapsed)

    except Exception as e:
        logging.error(f"[{start_str[:4]}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return (start_str[:4], relay, 0, 0)
    finally:
        loop.close()


async def run_sequential_single_relay():
    """단일 릴레이 순차 수집 (fallback)"""
    from collectors.nostr_local import NostrLocalCollector

    start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 11, 30, 23, 59, 59, tzinfo=timezone.utc)

    logging.info(f"Starting sequential collection: {start.date()} ~ {end.date()}")
    logging.info("Using single relay with short timeout for faster failure recovery")

    collector = NostrLocalCollector()
    total = await collector.collect(start, end)
    await collector.close_session()

    return total


def main():
    """메인 함수"""
    print("=" * 60)
    print("Nostr Multi-Relay Collection")
    print("=" * 60)
    print(f"Relays: {len(RELAYS)}")
    print(f"Periods: {len(PERIODS)}")

    for relay in RELAYS:
        print(f"  - {relay}")
    print("=" * 60)

    start_time = time.time()

    # 멀티프로세싱 풀 실행 (릴레이 수만큼)
    num_workers = min(len(PERIODS), mp.cpu_count())
    print(f"Workers: {num_workers}")

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(run_period, PERIODS)

    total_elapsed = time.time() - start_time

    # 결과 요약
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_records = 0
    for year, relay, count, elapsed in results:
        relay_short = relay.split("//")[1][:20]
        status = "OK" if count > 0 else "SPARSE/FAIL"
        print(f"  {year}: {count:>8,} records [{relay_short}] [{status}]")
        total_records += count

    print("-" * 60)
    print(f"  TOTAL: {total_records:,} records")
    print(f"  TIME: {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} hours)")
    print("=" * 60)

    # 경고: 2021-2022년 데이터가 적을 수 있음
    if total_records < 100:
        print("\n⚠️  WARNING: Low record count.")
        print("    Nostr data before 2022-12 is sparse.")
        print("    This is expected behavior.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
