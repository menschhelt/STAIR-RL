"""
Nostr Local Collector - Kind 1 (Text) + Kind 9735 (Zap) with Historical Queries.

Uses relay.nostr.band for historical data queries with since/until filters.
Collects both text notes and Zap receipts for weighted sentiment analysis.

File naming: nostr_YYYYWW.parquet (weekly partitioning by ISO week)

Key features:
- Historical queries via relay.nostr.band (best for since/until filters)
- Kind 1: Text notes with crypto keywords
- Kind 9735: Zap receipts (Bitcoin Lightning tips)
- Zap amount extraction for weighted sentiment
"""

import asyncio
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Set
import pandas as pd
import pyarrow as pa
import websockets
from websockets.exceptions import ConnectionClosed

from collectors.base_local_collector import BaseLocalCollector, SimpleRateLimiter
from config.settings import DATA_DIR, NostrConfig


class NostrLocalCollector(BaseLocalCollector):
    """
    Collects Nostr events (Kind 1 + Kind 9735) to local Parquet files.

    Data Schema:
    - event_id: string (primary key)
    - pubkey: string
    - kind: int32 (1=text, 9735=zap)
    - created_at: datetime64[ns, UTC]
    - content: string
    - zap_amount_sats: int64 (nullable, for kind 9735)
    - zap_target_event: string (nullable, event being zapped)
    - zap_sender: string (nullable)
    - zap_recipient: string (nullable)
    - hashtags: list[string]
    - cryptobert_bearish: float64 (placeholder, filled by NLP processor)
    - cryptobert_neutral: float64
    - cryptobert_bullish: float64
    """

    # Nostr relay for historical queries
    HISTORICAL_RELAY = 'wss://relay.nostr.band'

    # Backup relays
    BACKUP_RELAYS = [
        'wss://relay.damus.io',
        'wss://nos.lol',
        'wss://nostr.wine',
    ]

    # Crypto-related keywords for filtering Kind 1 (일반 크립토)
    CRYPTO_KEYWORDS = [
        # 메이저 코인
        'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol',
        # 일반 크립토
        'crypto', 'cryptocurrency', 'altcoin', 'defi', 'nft',
        # 라이트닝/노스트르
        'lightning', 'sats', 'satoshi', 'nostr', 'zap',
        # 트레이딩 용어
        'bullish', 'bearish', 'trading', 'pump', 'dump', 'moon', 'rekt',
        # 크립토 슬랭
        'hodl', 'degen', 'gm', 'wagmi', 'ngmi', 'fud', 'fomo',
    ]

    # [CRITICAL] 위기 감지 키워드 - 시스템 리스크 지표
    CRISIS_KEYWORDS = [
        # 1. 스테이블코인/페깅 (테라-루나 유형)
        'depeg', 'de-peg', 'unpeg', 'lose peg', 'off-peg',
        'death spiral', 'hyperinflation', 'parity',
        'ust', 'luna', 'algorithmic stablecoin',
        # 2. 거래소/유동성 위기 (FTX 유형)
        'withdrawal', 'withdrawals', 'halt', 'halted', 'suspend', 'suspended',
        'insolvent', 'insolvency', 'bankrupt', 'bankruptcy', 'liquidation',
        'bankrun', 'bank run', 'bailout', 'reserves', 'proof of reserves',
        'ftx', 'alameda', 'celsius', 'voyager', '3ac', 'three arrows',
        # 3. 사기/범죄/해킹
        'hack', 'hacked', 'hacker', 'exploit', 'bridge hack', 'drained',
        'scam', 'ponzi', 'rug', 'rugpull', 'rug pull', 'exit scam',
        'sec', 'cftc', 'lawsuit', 'investigation', 'arrest', 'fraud',
        # 4. 극단적 패닉 표현
        'capitulation', 'crash', 'collapse', 'panic', 'blood', 'bloodbath',
        'contagion', 'systemic', 'cascade', 'domino',
    ]

    # 전체 필터링용 합집합
    ALL_KEYWORDS = CRYPTO_KEYWORDS + CRISIS_KEYWORDS

    # Nostr event kinds
    KIND_TEXT = 1
    KIND_ZAP_RECEIPT = 9735

    def __init__(
        self,
        config: Optional[NostrConfig] = None,
    ):
        """
        Initialize Nostr collector.

        Args:
            config: NostrConfig instance
        """
        super().__init__(
            data_dir=DATA_DIR / 'nostr',
            partition_strategy='weekly',
        )

        self.config = config or NostrConfig()
        self.relay_url = self.config.relay
        # None = use defaults, [] = no backups (explicit)
        self.backup_relays = self.config.backup_relays if self.config.backup_relays is not None else self.BACKUP_RELAYS
        self.keywords = self.config.crypto_keywords or self.ALL_KEYWORDS
        self.kinds = self.config.kinds or [self.KIND_TEXT, self.KIND_ZAP_RECEIPT]

        # Initialize rate limiter from config
        rate_cfg = self.config.rate_limit
        self.rate_limiter = SimpleRateLimiter(
            max_requests=rate_cfg.max_requests,
            window_seconds=rate_cfg.window_seconds
        )

        # Compile keyword pattern for efficient matching
        self.keyword_pattern = re.compile(
            '|'.join(re.escape(kw) for kw in self.keywords),
            re.IGNORECASE
        )

        self.logger.info(
            f"Initialized NostrLocalCollector with relay={self.relay_url}, "
            f"kinds={self.kinds}, keywords={len(self.keywords)}, "
            f"rate_limit={rate_cfg.max_requests}/{rate_cfg.window_seconds}s"
        )

    @property
    def collector_name(self) -> str:
        return "nostr"

    @property
    def schema(self) -> pa.Schema:
        return pa.schema([
            ('event_id', pa.string()),
            ('pubkey', pa.string()),
            ('kind', pa.int32()),
            ('created_at', pa.timestamp('ns', tz='UTC')),
            ('content', pa.string()),
            ('zap_amount_sats', pa.int64()),
            ('zap_target_event', pa.string()),
            ('zap_sender', pa.string()),
            ('zap_recipient', pa.string()),
            ('hashtags', pa.list_(pa.string())),
            ('cryptobert_bearish', pa.float64()),
            ('cryptobert_neutral', pa.float64()),
            ('cryptobert_bullish', pa.float64()),
        ])

    def _get_primary_key_columns(self) -> List[str]:
        return ['event_id']

    # ========== Content Processing ==========

    def _clean_nostr_content(self, content: str) -> str:
        """
        Clean Nostr content by removing protocol-specific elements.

        Args:
            content: Raw Nostr note content

        Returns:
            Cleaned content string
        """
        # Remove nostr protocol identifiers (nostr:npub1..., nostr:note1..., etc.)
        content = re.sub(r'nostr:n[a-z]+1[qpzry9x8gf2tvdw0s3jn54khce6mua7l]+', '', content)

        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n', '\n', content)
        content = re.sub(r' +', ' ', content)

        return content.strip()

    def _has_crypto_keyword(self, text: str) -> bool:
        """
        Check if text contains any crypto keyword.

        Args:
            text: Text to check

        Returns:
            True if any keyword found
        """
        return bool(self.keyword_pattern.search(text))

    def _extract_hashtags(self, content: str) -> List[str]:
        """
        Extract hashtags from content.

        Args:
            content: Note content

        Returns:
            List of hashtags (without #)
        """
        hashtags = re.findall(r'#(\w+)', content)
        return [tag.lower() for tag in hashtags]

    # ========== Zap Parsing ==========

    def _parse_zap_event(self, event: Dict) -> Dict[str, Any]:
        """
        Parse Kind 9735 (Zap Receipt) event.

        Zap Receipt structure:
        - tags contain:
          - ['e', '<event_id>'] - event being zapped (if zapping a note)
          - ['p', '<pubkey>'] - recipient pubkey
          - ['bolt11', '<invoice>'] - Lightning invoice
          - ['description', '<json>'] - contains zap request with amount

        Args:
            event: Nostr event dict

        Returns:
            Dict with zap details
        """
        tags = event.get('tags', [])
        result = {
            'zap_amount_sats': None,
            'zap_target_event': None,
            'zap_sender': None,
            'zap_recipient': None,
        }

        for tag in tags:
            if len(tag) < 2:
                continue

            tag_name = tag[0]

            if tag_name == 'e':
                result['zap_target_event'] = tag[1]

            elif tag_name == 'p':
                result['zap_recipient'] = tag[1]

            elif tag_name == 'bolt11':
                # Try to decode amount from bolt11 invoice
                bolt11 = tag[1]
                amount = self._decode_bolt11_amount(bolt11)
                if amount:
                    result['zap_amount_sats'] = amount

            elif tag_name == 'description':
                # Parse zap request JSON for amount and sender
                try:
                    zap_request = json.loads(tag[1])
                    result['zap_sender'] = zap_request.get('pubkey')

                    # Amount might be in zap request tags
                    for req_tag in zap_request.get('tags', []):
                        if req_tag[0] == 'amount' and len(req_tag) > 1:
                            # Amount is in millisatoshis
                            # Handle float strings like '10000.0'
                            result['zap_amount_sats'] = int(float(req_tag[1])) // 1000

                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

        return result

    def _decode_bolt11_amount(self, bolt11: str) -> Optional[int]:
        """
        Decode amount from BOLT11 Lightning invoice.

        The amount is encoded in the human-readable part:
        - lnbc<amount><multiplier>
        - Multipliers: m (milli), u (micro), n (nano), p (pico)

        Args:
            bolt11: BOLT11 invoice string

        Returns:
            Amount in satoshis or None
        """
        try:
            # Remove prefix (lnbc, lntb, lnbcrt)
            if bolt11.startswith('lnbc'):
                hrp_amount = bolt11[4:]
            elif bolt11.startswith('lntb'):
                hrp_amount = bolt11[4:]
            elif bolt11.startswith('lnbcrt'):
                hrp_amount = bolt11[6:]
            else:
                return None

            # Find where the amount ends (first letter after digits)
            match = re.match(r'^(\d+)([munp])?', hrp_amount)
            if not match:
                return None

            amount_str = match.group(1)
            multiplier = match.group(2)

            amount = int(amount_str)

            # Convert to satoshis based on multiplier
            # Base unit is BTC
            if multiplier == 'm':
                # milli-BTC = 0.001 BTC = 100,000 sats
                sats = amount * 100_000
            elif multiplier == 'u':
                # micro-BTC = 0.000001 BTC = 100 sats
                sats = amount * 100
            elif multiplier == 'n':
                # nano-BTC = 0.000000001 BTC = 0.1 sats
                sats = amount // 10
            elif multiplier == 'p':
                # pico-BTC = 0.000000000001 BTC = 0.0001 sats
                sats = amount // 10_000
            else:
                # No multiplier = BTC
                sats = amount * 100_000_000

            return sats

        except (ValueError, AttributeError):
            return None

    # ========== WebSocket Communication ==========

    async def _subscribe_and_collect(
        self,
        relay_url: str,
        filters: List[Dict],
        timeout_seconds: int = 30,
        max_retries: int = 4,
    ) -> List[Dict]:
        """
        Subscribe to relay and collect events with exponential backoff.

        Args:
            relay_url: WebSocket relay URL
            filters: Nostr filter dicts
            timeout_seconds: Timeout for EOSE (End of Stored Events)
            max_retries: Maximum retry attempts (backoff: 1s, 2s, 4s, 8s)

        Returns:
            List of event dicts
        """
        events = []
        subscription_id = str(uuid.uuid4())[:8]

        for attempt in range(max_retries):
            if attempt > 0:
                backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s
                self.logger.warning(f"Retry {attempt}/{max_retries} after {backoff}s backoff")
                await asyncio.sleep(backoff)

            try:
                async with websockets.connect(
                    relay_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    # Send subscription request
                    req_message = json.dumps(["REQ", subscription_id] + filters)
                    await ws.send(req_message)

                    self.logger.debug(f"Sent subscription {subscription_id} to {relay_url}")

                    # Collect events until EOSE or timeout
                    try:
                        while True:
                            message = await asyncio.wait_for(
                                ws.recv(),
                                timeout=timeout_seconds
                            )

                            data = json.loads(message)

                            if data[0] == "EVENT" and data[1] == subscription_id:
                                event = data[2]
                                events.append(event)

                            elif data[0] == "EOSE" and data[1] == subscription_id:
                                self.logger.debug(f"EOSE received for {subscription_id}")
                                break

                            elif data[0] == "NOTICE":
                                self.logger.warning(f"Relay notice: {data[1]}")

                    except asyncio.TimeoutError:
                        self.logger.warning(f"Timeout waiting for events from {relay_url}")

                    # Close subscription
                    close_message = json.dumps(["CLOSE", subscription_id])
                    await ws.send(close_message)

                # Success! Return events
                return events

            except ConnectionClosed as e:
                self.logger.warning(f"Connection closed (attempt {attempt+1}/{max_retries}): {e}")
                continue  # Retry with backoff
            except Exception as e:
                self.logger.error(f"Error collecting from {relay_url} (attempt {attempt+1}/{max_retries}): {e}")
                continue  # Retry with backoff

        # All retries exhausted
        self.logger.error(f"Failed to collect from {relay_url} after {max_retries} attempts")
        return events

    # ========== Data Collection ==========

    async def collect_historical(
        self,
        start_date: datetime,
        end_date: datetime,
        kinds: Optional[List[int]] = None,
        batch_hours: int = 24,
    ) -> List[Dict]:
        """
        Collect historical events using since/until filters.

        relay.nostr.band is optimized for historical queries.

        Args:
            start_date: Start of collection period
            end_date: End of collection period
            kinds: Event kinds to collect
            batch_hours: Hours per batch (to avoid overwhelming relay)

        Returns:
            List of events
        """
        kinds = kinds or self.kinds
        all_events = []

        current_start = start_date
        while current_start < end_date:
            current_end = min(
                current_start + timedelta(hours=batch_hours),
                end_date
            )

            # Build filter for this batch
            filters = [{
                'kinds': kinds,
                'since': int(current_start.timestamp()),
                'until': int(current_end.timestamp()),
                'limit': 5000,  # relay.nostr.band supports high limits
            }]

            self.logger.info(
                f"Collecting {current_start.date()} to {current_end.date()}, kinds={kinds}"
            )

            # Rate limit before each WebSocket query
            await self.rate_limiter.acquire()

            # Try primary relay first
            events = await self._subscribe_and_collect(
                self.relay_url,
                filters,
                timeout_seconds=60,
            )

            if not events and current_start < datetime(2023, 1, 1, tzinfo=timezone.utc):
                # Try backup relays for older data
                for backup in self.backup_relays:
                    self.logger.info(f"Trying backup relay: {backup}")
                    # Rate limit before backup relay query
                    await self.rate_limiter.acquire()
                    events = await self._subscribe_and_collect(
                        backup,
                        filters,
                        timeout_seconds=60,
                    )
                    if events:
                        break

            all_events.extend(events)
            self.logger.info(f"Collected {len(events)} events for batch")

            # Move to next batch
            current_start = current_end

        return all_events

    def _process_events(self, events: List[Dict]) -> pd.DataFrame:
        """
        Process raw Nostr events into DataFrame.

        Args:
            events: List of raw event dicts

        Returns:
            DataFrame with processed data
        """
        records = []
        seen_ids: Set[str] = set()

        for event in events:
            event_id = event.get('id')

            # Skip duplicates
            if not event_id or event_id in seen_ids:
                continue
            seen_ids.add(event_id)

            kind = event.get('kind')
            content = event.get('content', '')
            created_at = event.get('created_at', 0)

            # Filter Kind 1 by crypto keywords
            if kind == self.KIND_TEXT:
                if not self._has_crypto_keyword(content):
                    continue
                content = self._clean_nostr_content(content)

            record = {
                'event_id': event_id,
                'pubkey': event.get('pubkey', '')[:16],  # Abbreviated
                'kind': kind,
                'created_at': pd.Timestamp(created_at, unit='s', tz='UTC'),
                'content': content[:5000] if content else '',  # Limit content length
                'zap_amount_sats': None,
                'zap_target_event': None,
                'zap_sender': None,
                'zap_recipient': None,
                'hashtags': self._extract_hashtags(content),
                'cryptobert_bearish': None,  # Filled by NLP processor
                'cryptobert_neutral': None,
                'cryptobert_bullish': None,
            }

            # Parse Zap details for Kind 9735
            if kind == self.KIND_ZAP_RECEIPT:
                zap_data = self._parse_zap_event(event)
                record.update(zap_data)

            records.append(record)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        self.logger.info(
            f"Processed {len(df)} events: "
            f"{len(df[df['kind'] == 1])} texts, "
            f"{len(df[df['kind'] == 9735])} zaps"
        )

        return df

    async def collect(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> int:
        """
        Collect Nostr data for the date range.

        Args:
            start_date: Start of collection period
            end_date: End of collection period

        Returns:
            Number of records collected
        """
        self.logger.info(f"Starting Nostr collection from {start_date} to {end_date}")

        # Collect historical events
        events = await self.collect_historical(start_date, end_date)

        if not events:
            self.logger.warning("No events collected")
            return 0

        # Process events into DataFrame
        df = self._process_events(events)

        if df.empty:
            return 0

        # Group by partition and write
        df['partition_key'] = df['created_at'].apply(self.get_partition_key)
        total_records = 0

        for key, group in df.groupby('partition_key'):
            group_df = group.drop(columns=['partition_key'])
            self.write_parquet(group_df, key, mode='append')
            total_records += len(group_df)

        # Save checkpoint
        self.save_checkpoint(end_date)

        self.logger.info(f"Collection complete: {total_records} records")
        return total_records


# ========== Standalone Execution ==========

async def main():
    """Run historical Nostr collection."""
    import argparse

    parser = argparse.ArgumentParser(description='Nostr Data Collector')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    collector = NostrLocalCollector()
    total = await collector.collect(start_date, end_date)

    print(f"Collected {total} records")
    print(f"Stats: {collector.get_data_stats()}")


if __name__ == '__main__':
    asyncio.run(main())
