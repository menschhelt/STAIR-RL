"""
GDELT Local Collector - News with Direct Scraping and FinBERT Sentiment.

Adapts the existing GDELTCollector for local Parquet storage.
Fetches news URLs from BigQuery/CSV, scrapes content, and applies FinBERT.

File naming: gdelt_YYYYWW.parquet (weekly partitioning by ISO week)

Key features:
- BigQuery for URL list with metadata
- Direct content scraping (not delegating to RAG Builder)
- FinBERT sentiment analysis (Positive/Negative/Neutral)
- NumMentions as weight for importance
"""

import asyncio
import aiohttp
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import pandas as pd
import pyarrow as pa
from bs4 import BeautifulSoup

from collectors.base_local_collector import BaseLocalCollector, SimpleRateLimiter
from config.settings import DATA_DIR, GDELTConfig


class GDELTLocalCollector(BaseLocalCollector):
    """
    Collects GDELT news data to local Parquet files.

    Pipeline:
    1. BigQuery/CSV: Get URL list with metadata
    2. Scraper: Fetch content in batches
    3. FinBERT: Process sentiment (deferred to text_processing module)
    4. Write to Parquet

    Data Schema:
    - url: string (primary key)
    - published_at: datetime64[ns, UTC]
    - source: string (publication name)
    - title: string
    - content: string (scraped full text)
    - tone_gdelt: float64 (original GDELT tone)
    - num_mentions: int64 (for weighting)
    - finbert_positive: float64 (placeholder)
    - finbert_negative: float64 (placeholder)
    - finbert_neutral: float64 (placeholder)
    - themes: list[string]
    - entities_org: list[string]
    """

    # GDELT Master file list URL
    MASTER_FILE_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
    MASTER_FILE_LATEST = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

    # ===========================================================================
    # 1. CORE_CRYPTO_THEMES: 엄격한 필터 (수집 단계)
    # - CRISISLEX는 여기서 제외 (노이즈 방지) → 위기 태깅 단계에서 활용
    # ===========================================================================
    CORE_CRYPTO_THEMES = {
        'ECON_BITCOIN',
        'ECON_CRYPTOCURRENCY',
        'ECON_BLOCKCHAIN',
        'WB_632_ELECTRONIC_CASH',
        'WB_1862_BITCOIN',            # 세계은행 분류
        'TAX_FNCACT_CRYPTOCURRENCY',
        'LEADER_CRYPTO',              # 인물 중심 뉴스 (CZ, SBF, Vitalik 등)
    }

    # ===========================================================================
    # 2. CORE_CRYPTO_KEYWORDS: 제목/본문 필터
    # - 동음이의어(LUNA, TERRA) 제거하고 구체화
    # - 생태계 용어(WEB3, NFT) + 위기 용어(FTX, DEPEG) 포함
    # ===========================================================================
    CORE_CRYPTO_KEYWORDS = [
        # 메이저 자산
        'BITCOIN', 'BTC', 'ETHEREUM', 'ETH',
        'CRYPTO', 'CRYPTOCURRENCY', 'BLOCKCHAIN', 'ALTCOIN',

        # 생태계/트렌드
        'DEFI', 'NFT', 'WEB3', 'METAVERSE', 'DAO', 'SMART CONTRACT',
        'STABLECOIN',

        # 주요 거래소 및 인프라
        'BINANCE', 'COINBASE', 'KRAKEN', 'FTX',

        # 위기/사건 식별 (구체화된 표현)
        'TERRA LUNA', 'TERRAUSD', 'LUNA COIN',  # 단순 'LUNA' 제외
        'DO KWON', 'SBF', 'SAM BANKMAN',
        'DEPEG', 'RUG PULL',
    ]

    # ===========================================================================
    # 3. CRISIS 관련 테마/키워드 (Feature 태깅용)
    # - 수집된 뉴스에 is_crisis 플래그 부여
    # ===========================================================================
    CRISIS_THEMES = {
        'CRISISLEX_T03_DEAD', 'CRISISLEX_C07_SAFETY',
        'ECON_BANKRUPTCY', 'ECON_FRAUD', 'ECON_SCAM',
        'LEGAL_SUIT', 'LEADER_ARREST',
    }

    CRISIS_KEYWORDS = [
        'HACK', 'HACKED', 'EXPLOIT', 'SCAM', 'PONZI',
        'BANKRUPT', 'BANKRUPTCY', 'INSOLVENT', 'INSOLVENCY',
        'COLLAPSE', 'CRASH', 'PANIC', 'CAPITULATION',
        'DEPEG', 'DE-PEG', 'UNPEG',
        'HALT', 'HALTED', 'SUSPEND', 'SUSPENDED',
        'ARREST', 'JAIL', 'FRAUD', 'LAWSUIT',
        'WITHDRAWAL', 'BANK RUN', 'LIQUIDITY CRISIS',
    ]

    def __init__(
        self,
        config: Optional[GDELTConfig] = None,
        use_bigquery: bool = False,
    ):
        """
        Initialize GDELT collector.

        Args:
            config: GDELTConfig instance
            use_bigquery: Use BigQuery (requires credentials) vs CSV files
        """
        super().__init__(
            data_dir=DATA_DIR / 'gdelt',
            partition_strategy='weekly',
        )

        self.config = config or GDELTConfig()
        self.use_bigquery = use_bigquery
        self.scrape_batch_size = self.config.scrape_batch_size

        # Rate limiter for scraping
        self.scrape_limiter = SimpleRateLimiter(
            max_requests=20,  # 20 requests per second
            window_seconds=1
        )

        self.logger.info(
            f"Initialized GDELTLocalCollector with "
            f"use_bigquery={use_bigquery}, batch_size={self.scrape_batch_size}"
        )

    @property
    def collector_name(self) -> str:
        return "gdelt"

    @property
    def schema(self) -> pa.Schema:
        return pa.schema([
            ('url', pa.string()),
            ('published_at', pa.timestamp('ns', tz='UTC')),
            ('source', pa.string()),
            ('title', pa.string()),
            ('content', pa.string()),
            ('tone_gdelt', pa.float64()),
            ('num_mentions', pa.int64()),
            ('is_crisis', pa.int32()),  # 위기 뉴스 플래그 (0 or 1)
            ('finbert_positive', pa.float64()),
            ('finbert_negative', pa.float64()),
            ('finbert_neutral', pa.float64()),
            ('themes', pa.list_(pa.string())),
            ('entities_org', pa.list_(pa.string())),
        ])

    def _get_primary_key_columns(self) -> List[str]:
        return ['url']

    # ========== GDELT CSV Parsing ==========

    def _is_crypto_related(self, themes: str, title: str = '') -> bool:
        """
        Check if article is crypto-related based on themes and title.
        Uses strict filtering to avoid noise (e.g., CRISISLEX excluded here).

        Args:
            themes: GDELT V2Themes string (semicolon-separated)
            title: Article title

        Returns:
            True if crypto-related
        """
        if not isinstance(themes, str):
            themes = ''
        if not isinstance(title, str):
            title = ''

        # A. 테마 매칭 (set intersection for efficiency)
        article_themes = set(themes.upper().split(';'))
        if not article_themes.isdisjoint(self.CORE_CRYPTO_THEMES):
            return True

        # B. 제목 키워드 매칭
        title_upper = title.upper()
        for keyword in self.CORE_CRYPTO_KEYWORDS:
            if keyword in title_upper:
                return True

        return False

    def _get_crisis_flag(self, themes: str, title: str = '') -> int:
        """
        Check if crypto news has crisis/risk attributes.
        Used for feature tagging after collection.

        Args:
            themes: GDELT V2Themes string (semicolon-separated)
            title: Article title

        Returns:
            1 if crisis-related, 0 otherwise
        """
        if not isinstance(themes, str):
            themes = ''
        if not isinstance(title, str):
            title = ''

        # A. 위기 테마 체크
        article_themes = set(themes.upper().split(';'))
        is_crisis_theme = not article_themes.isdisjoint(self.CRISIS_THEMES)

        # B. 위기 키워드 체크
        title_upper = title.upper()
        is_crisis_keyword = any(kw in title_upper for kw in self.CRISIS_KEYWORDS)

        return 1 if (is_crisis_theme or is_crisis_keyword) else 0

    def _parse_gdelt_tone(self, tone_str: str) -> float:
        """
        Parse GDELT V2Tone into normalized score.

        V2Tone format: "tone,posScore,negScore,polarity,activityDensity,..."
        Tone ranges roughly from -10 to +10.

        Args:
            tone_str: V2Tone string

        Returns:
            Normalized tone (-1.0 to 1.0)
        """
        try:
            if not tone_str:
                return 0.0

            parts = tone_str.split(',')
            raw_tone = float(parts[0])

            # Normalize from [-10, 10] to [-1, 1]
            normalized = max(-1.0, min(1.0, raw_tone / 10.0))
            return normalized

        except (ValueError, IndexError):
            return 0.0

    def _extract_entities(self, entities_str: str) -> List[str]:
        """
        Extract organization entities from GDELT V2Organizations.

        Args:
            entities_str: Semicolon-separated entity string

        Returns:
            List of entity names
        """
        if not entities_str:
            return []

        entities = []
        for entity in entities_str.split(';'):
            # Format: "Entity Name,offset" - we only want the name
            parts = entity.split(',')
            if parts:
                name = parts[0].strip()
                if name and len(name) > 2:
                    entities.append(name)

        return entities[:10]  # Limit to top 10

    def _extract_themes(self, themes_str: str) -> List[str]:
        """
        Extract relevant themes from GDELT V2Themes.

        Args:
            themes_str: Semicolon-separated themes string

        Returns:
            List of theme names
        """
        if not themes_str:
            return []

        themes = []
        for theme in themes_str.split(';'):
            parts = theme.split(',')
            if parts:
                name = parts[0].strip()
                if name:
                    themes.append(name)

        # Filter to keep crypto-related themes
        crypto_themes = [
            t for t in themes
            if any(ct in t.upper() for ct in ['ECON', 'CRYPTO', 'BITCOIN', 'BLOCK'])
        ]

        return crypto_themes[:10]  # Limit

    # ========== Content Scraping ==========

    async def scrape_url(self, url: str, timeout: int = 15) -> Optional[str]:
        """
        Scrape content from a URL.

        Args:
            url: Article URL
            timeout: Request timeout in seconds

        Returns:
            Extracted text content or None
        """
        try:
            await self.scrape_limiter.acquire()

            session = await self.get_session()
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)',
                'Accept': 'text/html,application/xhtml+xml',
            }

            async with session.get(url, headers=headers, timeout=timeout) as response:
                if response.status != 200:
                    return None

                html = await response.text()
                content = self._extract_text_from_html(html)
                return content

        except asyncio.TimeoutError:
            self.logger.debug(f"Timeout scraping {url}")
            return None
        except Exception as e:
            self.logger.debug(f"Error scraping {url}: {e}")
            return None

    def _extract_text_from_html(self, html: str) -> str:
        """
        Extract main text content from HTML.

        Args:
            html: Raw HTML content

        Returns:
            Cleaned text content
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                tag.decompose()

            # Try to find main content area
            main_content = None

            # Common content selectors
            selectors = [
                'article',
                '[class*="article"]',
                '[class*="content"]',
                '[class*="story"]',
                'main',
                '.post-content',
                '#content',
            ]

            for selector in selectors:
                element = soup.select_one(selector)
                if element:
                    main_content = element
                    break

            if main_content is None:
                main_content = soup.body

            if main_content is None:
                return ''

            # Get text
            text = main_content.get_text(separator=' ', strip=True)

            # Clean up
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

            # Limit length
            return text[:10000] if text else ''

        except Exception as e:
            self.logger.debug(f"Error extracting text: {e}")
            return ''

    async def scrape_batch(self, urls: List[str]) -> Dict[str, str]:
        """
        Scrape multiple URLs concurrently.

        Args:
            urls: List of URLs to scrape

        Returns:
            Dict mapping URL -> content
        """
        results = {}
        semaphore = asyncio.Semaphore(self.scrape_batch_size)

        async def scrape_with_semaphore(url: str):
            async with semaphore:
                content = await self.scrape_url(url)
                if content:
                    results[url] = content

        tasks = [scrape_with_semaphore(url) for url in urls]
        await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info(f"Scraped {len(results)}/{len(urls)} URLs successfully")
        return results

    # ========== CSV-based Collection (No BigQuery) ==========

    async def fetch_gdelt_csv(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch GDELT data by downloading and parsing CSV files.

        This method doesn't require BigQuery credentials.
        Downloads GKG (Global Knowledge Graph) files.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with article metadata
        """
        session = await self.get_session()

        # Get master file list
        async with session.get(self.MASTER_FILE_URL) as response:
            if response.status != 200:
                self.logger.error(f"Failed to fetch master file list: {response.status}")
                return pd.DataFrame()

            master_list = await response.text()

        # Parse master file list to find GKG files in date range
        gkg_files = []
        for line in master_list.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3:
                file_url = parts[2]
                # GKG files contain '.gkg.' in filename
                if '.gkg.' in file_url:
                    # Extract date from filename
                    match = re.search(r'(\d{14})', file_url)
                    if match:
                        file_date = datetime.strptime(match.group(1), '%Y%m%d%H%M%S')
                        file_date = file_date.replace(tzinfo=timezone.utc)
                        if start_date <= file_date <= end_date:
                            gkg_files.append((file_date, file_url))

        self.logger.info(f"Found {len(gkg_files)} GKG files in date range")

        # Download and parse files (process all files in range)
        all_records = []
        total_files = len(gkg_files)
        self.logger.info(f"Processing {total_files} GKG files...")

        for idx, (file_date, file_url) in enumerate(sorted(gkg_files)):
            try:
                records = await self._parse_gkg_file(file_url)
                all_records.extend(records)

                # Progress logging every 100 files
                if (idx + 1) % 100 == 0:
                    self.logger.info(f"Progress: {idx + 1}/{total_files} files processed, {len(all_records)} crypto articles found")

                # Delay between files (0.05s = 20 req/sec, fast but safe)
                await asyncio.sleep(0.05)

            except Exception as e:
                self.logger.warning(f"Error parsing {file_url}: {e}")
                continue

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        self.logger.info(f"Collected {len(df)} crypto-related articles from GDELT")

        return df

    async def _parse_gkg_file(self, file_url: str) -> List[Dict]:
        """
        Download and parse a GDELT GKG CSV file.

        Args:
            file_url: URL to GKG zip file

        Returns:
            List of article records
        """
        import zipfile
        import io

        session = await self.get_session()

        async with session.get(file_url) as response:
            if response.status != 200:
                return []

            content = await response.read()

        records = []

        # GKG files are zipped CSVs
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for filename in zf.namelist():
                    with zf.open(filename) as f:
                        # GKG format: tab-separated, many columns
                        for line in f:
                            try:
                                parts = line.decode('utf-8', errors='ignore').split('\t')
                                if len(parts) < 15:
                                    continue

                                # GKG 2.0 columns (relevant ones):
                                # 0: GKGRECORDID
                                # 1: DATE (YYYYMMDDHHMMSS)
                                # 2: SourceCollectionIdentifier
                                # 3: SourceCommonName
                                # 4: DocumentIdentifier (URL)
                                # 7: V2Themes
                                # 8: V2Locations
                                # 9: V2Persons
                                # 10: V2Organizations
                                # 14: V2Tone
                                # 15: V2Counts (contains NumMentions info)

                                url = parts[4] if len(parts) > 4 else ''
                                themes = parts[7] if len(parts) > 7 else ''
                                source = parts[3] if len(parts) > 3 else ''

                                # Filter for crypto-related
                                if not self._is_crypto_related(themes, ''):
                                    continue

                                # Parse date
                                date_str = parts[1] if len(parts) > 1 else ''
                                try:
                                    published_at = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                                    published_at = published_at.replace(tzinfo=timezone.utc)
                                except:
                                    continue

                                # Parse tone
                                tone_str = parts[14] if len(parts) > 14 else ''
                                tone = self._parse_gdelt_tone(tone_str)

                                # Parse entities
                                org_str = parts[10] if len(parts) > 10 else ''
                                entities = self._extract_entities(org_str)

                                # Extract themes
                                theme_list = self._extract_themes(themes)

                                # Parse num_mentions from V2Counts
                                counts_str = parts[15] if len(parts) > 15 else ''
                                num_mentions = self._parse_num_mentions(counts_str)

                                # Generate title from URL
                                title = self._generate_title_from_url(url)

                                record = {
                                    'url': url,
                                    'published_at': published_at,
                                    'source': source,
                                    'title': title,
                                    'content': '',  # Filled by scraping
                                    'tone_gdelt': tone,
                                    'num_mentions': num_mentions,
                                    'finbert_positive': None,
                                    'finbert_negative': None,
                                    'finbert_neutral': None,
                                    'themes': theme_list,
                                    'entities_org': entities,
                                    'is_crisis': self._get_crisis_flag(themes, title),
                                }
                                records.append(record)

                            except Exception as e:
                                continue

        except Exception as e:
            self.logger.error(f"Error parsing GKG file: {e}")

        return records

    def _parse_num_mentions(self, counts_str: str) -> int:
        """
        Parse number of mentions from V2Counts.

        Args:
            counts_str: V2Counts string

        Returns:
            Number of mentions
        """
        try:
            if not counts_str:
                return 1

            # V2Counts format: "TYPE#COUNT#LOCATION#..."
            total = 1
            for item in counts_str.split(';'):
                parts = item.split('#')
                if len(parts) >= 2:
                    count = int(parts[1])
                    total = max(total, count)

            return total

        except (ValueError, IndexError):
            return 1

    def _generate_title_from_url(self, url: str) -> str:
        """
        Generate article title from URL slug.

        Args:
            url: Article URL

        Returns:
            Generated title
        """
        try:
            # Extract last path segment
            path = url.split('/')[-1]
            # Remove extension and query params
            path = re.sub(r'\.(html|htm|php|asp|aspx).*$', '', path, flags=re.IGNORECASE)
            path = path.split('?')[0]
            # Convert hyphens/underscores to spaces
            title = re.sub(r'[-_]', ' ', path)
            # Title case
            title = title.title()
            return title[:200] if title else url[:200]

        except:
            return url[:200]

    # ========== Main Collection ==========

    async def collect(
        self,
        start_date: datetime,
        end_date: datetime,
        scrape_content: bool = True,
    ) -> int:
        """
        Collect GDELT news data for the date range.
        Processes data week by week to manage memory and allow resume.

        Args:
            start_date: Start of collection period
            end_date: End of collection period
            scrape_content: Whether to scrape article content

        Returns:
            Number of records collected
        """
        self.logger.info(f"Starting GDELT collection from {start_date} to {end_date}")

        # Generate weekly date ranges
        weeks = []
        current = start_date
        while current < end_date:
            week_end = min(current + timedelta(days=7), end_date)
            weeks.append((current, week_end))
            current = week_end

        self.logger.info(f"Processing {len(weeks)} weeks of data")
        total_records = 0

        for week_idx, (week_start, week_end) in enumerate(weeks):
            self.logger.info(f"Week {week_idx + 1}/{len(weeks)}: {week_start.date()} to {week_end.date()}")

            # Fetch metadata for this week
            df = await self.fetch_gdelt_csv(week_start, week_end)

            if df.empty:
                self.logger.info(f"No articles found for week {week_idx + 1}")
                continue

            # Deduplicate by URL
            df = df.drop_duplicates(subset=['url'], keep='first')
            self.logger.info(f"Found {len(df)} unique articles for this week")

            # Scrape content
            if scrape_content and len(df) > 0:
                urls = df['url'].tolist()
                content_map = await self.scrape_batch(urls)

                # Update content column
                df['content'] = df['url'].map(content_map).fillna('')

                # Remove articles where scraping failed
                df = df[df['content'].str.len() > 100]
                self.logger.info(f"Retained {len(df)} articles after scraping")

            if df.empty:
                continue

            # Group by partition and write
            df['partition_key'] = df['published_at'].apply(self.get_partition_key)
            week_records = 0

            for key, group in df.groupby('partition_key'):
                group_df = group.drop(columns=['partition_key'])
                self.write_parquet(group_df, key, mode='append')
                week_records += len(group_df)

            total_records += week_records
            self.logger.info(f"Week {week_idx + 1} complete: {week_records} records (total: {total_records})")

            # Save checkpoint after each week
            self.save_checkpoint(week_end)

        self.logger.info(f"Collection complete: {total_records} records")
        return total_records


# ========== Standalone Execution ==========

async def main():
    """Run historical GDELT collection."""
    import argparse

    parser = argparse.ArgumentParser(description='GDELT News Collector')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-scrape', action='store_true', help='Skip content scraping')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    async with GDELTLocalCollector() as collector:
        total = await collector.collect(
            start_date=start_date,
            end_date=end_date,
            scrape_content=not args.no_scrape,
        )
        print(f"Collected {total} records")
        print(f"Stats: {collector.get_data_stats()}")


if __name__ == '__main__':
    asyncio.run(main())
