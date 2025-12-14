"""
Weighted Sentiment - Apply Zap/Mentions weights to sentiment scores.

Core formula:
- Nostr:  weighted_sentiment = sentiment_scores × log(1 + Zap_Amount)
- GDELT:  weighted_sentiment = sentiment_scores × log(1 + NumMentions)

This is the "Costly Signaling" mechanism:
- Spam bots (Zap=0): [0.9, 0.1, 0.0] × 0 = [0, 0, 0] → RL ignores
- Insider (Zap=100万): [0.9, 0.1, 0.0] × 13.8 = [12.4, 1.38, 0] → Strong signal

The weight amplifies trustworthy signals and suppresses noise.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging


class WeightedSentiment:
    """
    Applies cost-based weights to sentiment scores.

    Nostr: Zap Amount (actual money transferred)
    GDELT: NumMentions (how widely reported)

    Both represent "skin in the game" - trustworthiness indicators.
    """

    def __init__(self):
        """Initialize Weighted Sentiment calculator."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def apply_weight(
        self,
        sentiment_scores: np.ndarray,
        weight: float,
    ) -> np.ndarray:
        """
        Apply log weight to sentiment scores.

        Formula: weighted = scores × log(1 + weight)

        Args:
            sentiment_scores: Array of sentiment probabilities [3]
            weight: Zap amount (sats) or NumMentions

        Returns:
            Weighted sentiment array
        """
        # log1p(x) = log(1 + x), handles weight=0 gracefully
        log_weight = np.log1p(weight)
        return sentiment_scores * log_weight

    def process_nostr_dataframe(
        self,
        df: pd.DataFrame,
        zap_column: str = 'zap_amount_sats',
        sentiment_columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Apply Zap weights to Nostr sentiment scores.

        Args:
            df: DataFrame with sentiment and zap data
            zap_column: Column with Zap amounts
            sentiment_columns: Sentiment probability columns

        Returns:
            DataFrame with weighted sentiment columns
        """
        if df.empty:
            return df

        if sentiment_columns is None:
            sentiment_columns = ['cryptobert_bearish', 'cryptobert_neutral', 'cryptobert_bullish']

        # Check columns exist
        missing = [c for c in sentiment_columns if c not in df.columns]
        if missing:
            self.logger.warning(f"Missing columns: {missing}")
            return df

        # Get zap weights
        if zap_column in df.columns:
            zap_weights = df[zap_column].fillna(0)
        else:
            # No zap data - use 1 (neutral weight)
            zap_weights = pd.Series(1, index=df.index)
            self.logger.warning(f"Column {zap_column} not found, using weight=1")

        # Calculate log weights
        log_weights = np.log1p(zap_weights)

        # Apply weights
        for col in sentiment_columns:
            weighted_col = f"{col}_weighted"
            df[weighted_col] = df[col] * log_weights

        return df

    def process_gdelt_dataframe(
        self,
        df: pd.DataFrame,
        mentions_column: str = 'num_mentions',
        sentiment_columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Apply NumMentions weights to GDELT sentiment scores.

        Args:
            df: DataFrame with sentiment and mentions data
            mentions_column: Column with mention counts
            sentiment_columns: Sentiment probability columns

        Returns:
            DataFrame with weighted sentiment columns
        """
        if df.empty:
            return df

        if sentiment_columns is None:
            sentiment_columns = ['finbert_positive', 'finbert_negative', 'finbert_neutral']

        # Check columns exist
        missing = [c for c in sentiment_columns if c not in df.columns]
        if missing:
            self.logger.warning(f"Missing columns: {missing}")
            return df

        # Get mention weights
        if mentions_column in df.columns:
            mention_weights = df[mentions_column].fillna(1)
        else:
            # No mention data - use 1 (neutral weight)
            mention_weights = pd.Series(1, index=df.index)
            self.logger.warning(f"Column {mentions_column} not found, using weight=1")

        # Calculate log weights
        log_weights = np.log1p(mention_weights)

        # Apply weights
        for col in sentiment_columns:
            weighted_col = f"{col}_weighted"
            df[weighted_col] = df[col] * log_weights

        return df


class TextAggregator:
    """
    Aggregates text sentiment to match market data granularity.

    Converts per-event sentiment to time-windowed features:
    - Mean sentiment in window
    - Weighted mean (by Zap or Mentions)
    - Volume (count of events)
    """

    def __init__(
        self,
        window: str = '5min',
    ):
        """
        Initialize Text Aggregator.

        Args:
            window: Aggregation window (e.g., '5min', '1h', '1d')
        """
        self.window = window
        self.weighted_sentiment = WeightedSentiment()

        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def aggregate_nostr(
        self,
        df: pd.DataFrame,
        timestamp_column: str = 'created_at',
    ) -> pd.DataFrame:
        """
        Aggregate Nostr sentiment by time window.

        Output columns (~6 features):
        - nostr_bearish_weighted_mean
        - nostr_neutral_weighted_mean
        - nostr_bullish_weighted_mean
        - nostr_event_count
        - nostr_total_zap_sats
        - nostr_avg_zap_sats

        Args:
            df: Nostr DataFrame with sentiment scores
            timestamp_column: Timestamp column name

        Returns:
            Aggregated DataFrame indexed by time window
        """
        if df.empty:
            return pd.DataFrame()

        # Apply weights first
        df = self.weighted_sentiment.process_nostr_dataframe(df)

        # Set timestamp as index
        df = df.set_index(pd.DatetimeIndex(df[timestamp_column]))

        # Weighted sentiment columns
        weighted_cols = [
            'cryptobert_bearish_weighted',
            'cryptobert_neutral_weighted',
            'cryptobert_bullish_weighted',
        ]

        # Check columns exist
        available_weighted = [c for c in weighted_cols if c in df.columns]
        if not available_weighted:
            self.logger.warning("No weighted sentiment columns found")
            return pd.DataFrame()

        # Resample and aggregate
        agg_dict = {}
        for col in available_weighted:
            agg_dict[col] = 'sum'  # Sum because weights are already applied

        # Add counts
        agg_dict['event_id'] = 'count'

        # Add zap stats if available
        if 'zap_amount_sats' in df.columns:
            agg_dict['zap_amount_sats'] = ['sum', 'mean']

        result = df.resample(self.window).agg(agg_dict)

        # Flatten column names
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ['_'.join(col).strip('_') for col in result.columns.values]

        # Rename columns
        rename_map = {
            'cryptobert_bearish_weighted_sum': 'nostr_bearish_weighted',
            'cryptobert_neutral_weighted_sum': 'nostr_neutral_weighted',
            'cryptobert_bullish_weighted_sum': 'nostr_bullish_weighted',
            'event_id_count': 'nostr_event_count',
            'zap_amount_sats_sum': 'nostr_total_zap_sats',
            'zap_amount_sats_mean': 'nostr_avg_zap_sats',
        }
        result = result.rename(columns=rename_map)

        # ========== Forward-fill sentiment, fillna(0) for counts ==========
        # Nostr는 실시간이지만 5분 윈도우 내 이벤트 없을 수 있음
        # 분위기(sentiment)는 새 이벤트가 올 때까지 유지되어야 함
        # limit=6: 30분까지 유지 (Nostr는 GDELT보다 빈번하므로 더 짧게)
        sentiment_cols = ['nostr_bearish_weighted', 'nostr_neutral_weighted', 'nostr_bullish_weighted']
        for col in sentiment_cols:
            if col in result.columns:
                result[col] = result[col].ffill(limit=6)

        # Count/amount는 그 순간의 팩트이므로 0 유지
        count_cols = ['nostr_event_count', 'nostr_total_zap_sats', 'nostr_avg_zap_sats']
        for col in count_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0)

        # 나머지 NaN (맨 처음 구간 등)은 0으로
        result = result.fillna(0)

        return result

    def aggregate_gdelt(
        self,
        df: pd.DataFrame,
        timestamp_column: str = 'published_at',
    ) -> pd.DataFrame:
        """
        Aggregate GDELT sentiment by time window.

        Output columns (~6 features):
        - gdelt_positive_weighted_mean
        - gdelt_negative_weighted_mean
        - gdelt_neutral_weighted_mean
        - gdelt_article_count
        - gdelt_total_mentions
        - gdelt_avg_mentions

        Args:
            df: GDELT DataFrame with sentiment scores
            timestamp_column: Timestamp column name

        Returns:
            Aggregated DataFrame indexed by time window
        """
        if df.empty:
            return pd.DataFrame()

        # Apply weights first
        df = self.weighted_sentiment.process_gdelt_dataframe(df)

        # Set timestamp as index
        df = df.set_index(pd.DatetimeIndex(df[timestamp_column]))

        # Weighted sentiment columns
        weighted_cols = [
            'finbert_positive_weighted',
            'finbert_negative_weighted',
            'finbert_neutral_weighted',
        ]

        # Check columns exist
        available_weighted = [c for c in weighted_cols if c in df.columns]
        if not available_weighted:
            self.logger.warning("No weighted sentiment columns found")
            return pd.DataFrame()

        # Resample and aggregate
        agg_dict = {}
        for col in available_weighted:
            agg_dict[col] = 'sum'

        # Add counts
        agg_dict['url'] = 'count'

        # Add mention stats if available
        if 'num_mentions' in df.columns:
            agg_dict['num_mentions'] = ['sum', 'mean']

        result = df.resample(self.window).agg(agg_dict)

        # Flatten column names
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ['_'.join(col).strip('_') for col in result.columns.values]

        # Rename columns
        rename_map = {
            'finbert_positive_weighted_sum': 'gdelt_positive_weighted',
            'finbert_negative_weighted_sum': 'gdelt_negative_weighted',
            'finbert_neutral_weighted_sum': 'gdelt_neutral_weighted',
            'url_count': 'gdelt_article_count',
            'num_mentions_sum': 'gdelt_total_mentions',
            'num_mentions_mean': 'gdelt_avg_mentions',
        }
        result = result.rename(columns=rename_map)

        # ========== Forward-fill sentiment, fillna(0) for counts ==========
        # GDELT는 15분 주기 업데이트, 5분 윈도우로 쪼개면 sparse
        # 분위기(sentiment)는 새 뉴스가 올 때까지 유지되어야 함
        # limit=3: 15분까지 유지 (GDELT 업데이트 주기 = 15분)
        sentiment_cols = ['gdelt_positive_weighted', 'gdelt_negative_weighted', 'gdelt_neutral_weighted']
        for col in sentiment_cols:
            if col in result.columns:
                result[col] = result[col].ffill(limit=3)

        # Count/mentions는 그 순간의 팩트이므로 0 유지
        count_cols = ['gdelt_article_count', 'gdelt_total_mentions', 'gdelt_avg_mentions']
        for col in count_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0)

        # 나머지 NaN (맨 처음 구간 등)은 0으로
        result = result.fillna(0)

        return result

    def merge_text_features(
        self,
        nostr_agg: pd.DataFrame,
        gdelt_agg: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge Nostr and GDELT aggregated features.

        Args:
            nostr_agg: Aggregated Nostr features
            gdelt_agg: Aggregated GDELT features

        Returns:
            Merged DataFrame with all text features
        """
        if nostr_agg.empty and gdelt_agg.empty:
            return pd.DataFrame()

        if nostr_agg.empty:
            return gdelt_agg

        if gdelt_agg.empty:
            return nostr_agg

        # Outer join to keep all timestamps
        result = nostr_agg.join(gdelt_agg, how='outer')
        result = result.fillna(0)

        return result


# ========== Pipeline Function ==========

def process_text_pipeline(
    nostr_df: pd.DataFrame,
    gdelt_df: pd.DataFrame,
    aggregation_window: str = '5min',
    cryptobert_device: str = None,
    finbert_device: str = None,
) -> pd.DataFrame:
    """
    Full text processing pipeline.

    Steps:
    1. Process Nostr texts through CryptoBERT
    2. Process GDELT texts through FinBERT
    3. Apply Zap/Mentions weights
    4. Aggregate by time window
    5. Merge into final feature DataFrame

    Args:
        nostr_df: Nostr DataFrame with 'content' column
        gdelt_df: GDELT DataFrame with 'content' column
        aggregation_window: Time window for aggregation
        cryptobert_device: Device for CryptoBERT ('cuda:0', 'cuda:1', etc.)
        finbert_device: Device for FinBERT

    Returns:
        DataFrame with aggregated text features (~12 columns)
    """
    from text_processing.cryptobert_processor import CryptoBERTProcessor
    from text_processing.finbert_processor import FinBERTProcessor

    # Process Nostr with CryptoBERT
    if not nostr_df.empty and 'content' in nostr_df.columns:
        cryptobert = CryptoBERTProcessor(device=cryptobert_device)
        nostr_df = cryptobert.process_dataframe(
            nostr_df,
            text_column='content',
            output_prefix='cryptobert',
            show_progress=True,
        )

    # Process GDELT with FinBERT
    if not gdelt_df.empty and 'content' in gdelt_df.columns:
        finbert = FinBERTProcessor(device=finbert_device)
        gdelt_df = finbert.process_dataframe(
            gdelt_df,
            text_column='content',
            output_prefix='finbert',
            show_progress=True,
        )

    # Aggregate
    aggregator = TextAggregator(window=aggregation_window)
    nostr_agg = aggregator.aggregate_nostr(nostr_df)
    gdelt_agg = aggregator.aggregate_gdelt(gdelt_df)

    # Merge
    result = aggregator.merge_text_features(nostr_agg, gdelt_agg)

    return result


# ========== Standalone Testing ==========

if __name__ == '__main__':
    # Test weighted sentiment
    ws = WeightedSentiment()

    # Test cases
    test_cases = [
        ("Spam bot (Zap=0)", [0.9, 0.05, 0.05], 0),
        ("Normal user (Zap=100)", [0.7, 0.2, 0.1], 100),
        ("Whale (Zap=100,000)", [0.8, 0.1, 0.1], 100_000),
        ("Insider (Zap=1,000,000)", [0.95, 0.03, 0.02], 1_000_000),
    ]

    print("Weighted Sentiment Test:\n")
    print(f"{'Description':<25} {'Raw Scores':<25} {'Weight':<15} {'Weighted':<25}")
    print("-" * 90)

    for desc, scores, weight in test_cases:
        raw = np.array(scores)
        weighted = ws.apply_weight(raw, weight)
        print(f"{desc:<25} {str(raw):<25} {weight:<15} {str(np.round(weighted, 2)):<25}")
