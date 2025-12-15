"""
STAIR-RL Local Data Collection & Backtesting System - Configuration

설정 관리 구조:
- config/universe_config.yaml: 모든 날짜/파라미터 관리
- config/settings.py: Python dataclass 정의 및 YAML 로딩

주의사항:
1. Look-ahead Bias 금지: Scaler.fit()은 Train 데이터로만
2. Test Set 오염 금지: Test 결과 보고 모델 수정 절대 금지
3. Nostr 데이터는 2022.12부터 활성화, 이전은 GDELT fallback
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import yaml


# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path("/home/work/data/stair-local")  # NFS mount with 684GB available
CONFIG_DIR = BASE_DIR / "config"

# Alpha factors path (self-contained for open source)
ALPHA_BASE_PATH = BASE_DIR / "alphas"


@dataclass
class RateLimitConfig:
    """Rate limiter configuration for API calls"""
    max_requests: int = 60
    window_seconds: int = 60


@dataclass
class UniverseConfig:
    """Dynamic universe configuration"""
    top_n: int = 20
    min_volume_usd: float = 10_000_000  # $10M minimum
    rebalance_hour_utc: int = 0  # 00:00 UTC daily


@dataclass
class BinanceConfig:
    """Binance futures collector configuration"""
    interval: str = "5m"
    partition: str = "monthly"  # File partitioning strategy
    base_url: str = "https://fapi.binance.com"
    weight_limit: int = 2400
    max_limit: int = 1500
    rate_limit: RateLimitConfig = field(default_factory=lambda: RateLimitConfig(240, 60))


@dataclass
class NostrConfig:
    """Nostr collector configuration"""
    relay: str = "wss://relay.nostr.band"
    backup_relays: List[str] = field(default_factory=lambda: [
        'wss://relay.damus.io', 'wss://nos.lol', 'wss://nostr.wine'
    ])
    kinds: List[int] = field(default_factory=lambda: [1, 9735])  # Text + Zap
    partition: str = "weekly"
    crypto_keywords: List[str] = field(default_factory=lambda: [
        'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
        'lightning', 'sats', 'satoshi', 'hodl', 'nostr', 'zap',
        'bullish', 'bearish', 'defi', 'nft', 'solana', 'sol'
    ])
    rate_limit: RateLimitConfig = field(default_factory=lambda: RateLimitConfig(30, 60))


@dataclass
class GDELTConfig:
    """GDELT collector configuration"""
    partition: str = "weekly"
    scrape_batch_size: int = 50
    finbert_batch_size: int = 32
    crypto_themes: List[str] = field(default_factory=lambda: [
        'ECON_BITCOIN', 'ECON_CRYPTOCURRENCY', 'ECON_BLOCKCHAIN',
        'WB_632_ELECTRONIC_CASH', 'TAX_FNCACT_CRYPTOCURRENCY'
    ])
    rate_limit: RateLimitConfig = field(default_factory=lambda: RateLimitConfig(20, 1))


@dataclass
class FREDConfig:
    """FRED collector configuration"""
    partition: str = "monthly"
    rate_limit: RateLimitConfig = field(default_factory=lambda: RateLimitConfig(120, 60))


@dataclass
class YFinanceConfig:
    """yfinance collector configuration"""
    partition: str = "monthly"
    rate_limit: RateLimitConfig = field(default_factory=lambda: RateLimitConfig(60, 60))


@dataclass
class AlphaConfig:
    """Alpha factor configuration"""
    alpha_101_dir: Path = ALPHA_BASE_PATH / "alpha_101"
    # alpha_191_dir: Path = ALPHA_BASE_PATH / "alpha_191"  # DISABLED: Only using Alpha 101
    skip_alphas: List[str] = field(default_factory=lambda: [])


@dataclass
class TextProcessingConfig:
    """NLP processing configuration"""
    aggregation_window: str = "5min"
    cryptobert_model: str = "ElKulako/cryptobert"  # Nostr용
    finbert_model: str = "ProsusAI/finbert"  # GDELT용
    batch_size: int = 256  # GPU batch size (A100 optimized)


@dataclass
class CollectionConfig:
    """데이터 수집 기간 설정 (YAML에서 로드)"""
    macro_start: str = "2009-01-01"
    macro_end: str = "2023-12-31"
    crypto_start: str = "2020-01-01"
    crypto_end: str = "2023-12-31"
    nostr_active_from: str = "2022-12-01"  # 이전은 GDELT fallback


@dataclass
class BacktestConfig:
    """
    Backtesting configuration (YAML에서 로드).

    Buffer: 2020.01 ~ 2020.12 (지표 계산용 웜업)
    Train:  2021.01 ~ 2022.06 (18개월)
    Val:    2022.07 ~ 2022.12 (6개월)
    Test:   2023.01 ~ 2023.12 (12개월)
    """
    granularity: str = "5m"

    # Buffer (rolling window 웜업용)
    buffer_start: str = "2020-01-01"
    buffer_end: str = "2020-12-31"

    # Train / Val / Test
    train_start: str = "2021-01-01"
    train_end: str = "2022-06-30"
    val_start: str = "2022-07-01"
    val_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    test_end: str = "2023-12-31"

    # Binance Futures fee structure (VIP 0)
    taker_fee: float = 0.0004  # 0.04% per trade
    maker_fee: float = 0.0002  # 0.02% per trade
    slippage: float = 0.0005   # 0.05% estimated slippage


@dataclass
class CQLSACConfig:
    """Phase 1: CQL-SAC Offline Pre-training"""
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 1e-3
    batch_size: int = 128  # Increased for better GPU utilization (was 32)
    replay_buffer_size: int = 1_000_000
    lambda_cql: float = 1.0  # CQL regularization
    lambda_gp: float = 10.0  # Gradient penalty
    alpha: float = 0.2  # SAC temperature
    tau: float = 0.005  # Target network update
    gamma: float = 0.99  # Discount factor
    training_steps: int = 500_000


@dataclass
class PPOCVaRConfig:
    """Phase 2: PPO-CVaR Online Fine-tuning"""
    learning_rate: float = 1e-4
    clip_epsilon: float = 0.2
    ppo_epochs: int = 10
    horizon: int = 2048
    batch_size: int = 64
    gae_lambda: float = 0.95
    gamma: float = 0.99
    alpha_cvar: float = 0.95  # Confidence level
    kappa: float = 0.05  # Risk tolerance (5%)
    training_steps: int = 100_000


@dataclass
class RetrainingConfig:
    """Phase 3: Event-Driven Adaptive Retraining"""
    lambda_anchor: float = 5.0  # Anchor regularization
    rollback_threshold: float = 0.05  # 5% performance drop


@dataclass
class RLConfig:
    """RL training configuration"""
    target_leverage: float = 2.0  # 200% gross exposure
    cql_sac: CQLSACConfig = field(default_factory=CQLSACConfig)
    ppo_cvar: PPOCVaRConfig = field(default_factory=PPOCVaRConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)


@dataclass
class Config:
    """Main configuration container"""
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    nostr: NostrConfig = field(default_factory=NostrConfig)
    gdelt: GDELTConfig = field(default_factory=GDELTConfig)
    fred: FREDConfig = field(default_factory=FREDConfig)
    yfinance: YFinanceConfig = field(default_factory=YFinanceConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    text_processing: TextProcessingConfig = field(default_factory=TextProcessingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        def parse_rate_limit(d: dict) -> RateLimitConfig:
            """Parse rate_limit nested dict"""
            rl = d.get('rate_limit', {})
            return RateLimitConfig(
                max_requests=rl.get('max_requests', 60),
                window_seconds=rl.get('window_seconds', 60)
            )

        # Parse binance config
        binance_data = data.get('binance', {})
        binance_cfg = BinanceConfig(
            interval=binance_data.get('interval', '5m'),
            partition=binance_data.get('partition', 'monthly'),
            rate_limit=parse_rate_limit(binance_data)
        )

        # Parse nostr config
        nostr_data = data.get('nostr', {})
        nostr_cfg = NostrConfig(
            relay=nostr_data.get('relay', 'wss://relay.nostr.band'),
            backup_relays=nostr_data.get('backup_relays', []),
            kinds=nostr_data.get('kinds', [1, 9735]),
            partition=nostr_data.get('partition', 'weekly'),
            rate_limit=parse_rate_limit(nostr_data)
        )

        # Parse gdelt config
        gdelt_data = data.get('gdelt', {})
        gdelt_cfg = GDELTConfig(
            partition=gdelt_data.get('partition', 'weekly'),
            scrape_batch_size=gdelt_data.get('scrape_batch_size', 50),
            finbert_batch_size=gdelt_data.get('finbert_batch_size', 32),
            rate_limit=parse_rate_limit(gdelt_data)
        )

        # Parse fred/yfinance config
        fred_data = data.get('fred', {})
        fred_cfg = FREDConfig(
            partition=fred_data.get('partition', 'monthly'),
            rate_limit=parse_rate_limit(fred_data)
        )

        yfinance_data = data.get('yfinance', {})
        yfinance_cfg = YFinanceConfig(
            partition=yfinance_data.get('partition', 'monthly'),
            rate_limit=parse_rate_limit(yfinance_data)
        )

        # Parse alpha config
        alpha_data = data.get('alpha', {})
        alpha_cfg = AlphaConfig(
            alpha_101_dir=Path(alpha_data.get('alpha_101_dir', ALPHA_BASE_PATH / "alpha_101"))
        )

        # Parse RL config with nested sub-configs
        rl_data = data.get('rl', {})
        cql_data = rl_data.get('cql_sac', {})
        ppo_data = rl_data.get('ppo_cvar', {})
        retrain_data = rl_data.get('retraining', {})

        rl_cfg = RLConfig(
            target_leverage=rl_data.get('target_leverage', 2.0),
            cql_sac=CQLSACConfig(
                learning_rate_actor=cql_data.get('learning_rate_actor', 3e-4),
                learning_rate_critic=cql_data.get('learning_rate_critic', 1e-3),
                batch_size=cql_data.get('batch_size', 256),
                replay_buffer_size=cql_data.get('replay_buffer_size', 1_000_000),
                lambda_cql=cql_data.get('lambda_cql', 1.0),
                lambda_gp=cql_data.get('lambda_gp', 10.0),
                alpha=cql_data.get('alpha', 0.2),
                tau=cql_data.get('tau', 0.005),
                gamma=cql_data.get('gamma', 0.99),
                training_steps=cql_data.get('training_steps', 500_000)
            ),
            ppo_cvar=PPOCVaRConfig(
                learning_rate=ppo_data.get('learning_rate', 1e-4),
                clip_epsilon=ppo_data.get('clip_epsilon', 0.2),
                ppo_epochs=ppo_data.get('ppo_epochs', 10),
                horizon=ppo_data.get('horizon', 2048),
                batch_size=ppo_data.get('batch_size', 64),
                gae_lambda=ppo_data.get('gae_lambda', 0.95),
                gamma=ppo_data.get('gamma', 0.99),
                alpha_cvar=ppo_data.get('alpha_cvar', 0.95),
                kappa=ppo_data.get('kappa', 0.05),
                training_steps=ppo_data.get('training_steps', 100_000)
            ),
            retraining=RetrainingConfig(
                lambda_anchor=retrain_data.get('lambda_anchor', 5.0),
                rollback_threshold=retrain_data.get('rollback_threshold', 0.05)
            )
        )

        # Parse collection config
        collection_data = data.get('collection', {})
        collection_cfg = CollectionConfig(
            macro_start=collection_data.get('macro_start', '2009-01-01'),
            macro_end=collection_data.get('macro_end', '2023-12-31'),
            crypto_start=collection_data.get('crypto_start', '2020-01-01'),
            crypto_end=collection_data.get('crypto_end', '2023-12-31'),
            nostr_active_from=collection_data.get('nostr_active_from', '2022-12-01'),
        )

        return cls(
            universe=UniverseConfig(**data.get('universe', {})),
            collection=collection_cfg,
            binance=binance_cfg,
            nostr=nostr_cfg,
            gdelt=gdelt_cfg,
            fred=fred_cfg,
            yfinance=yfinance_cfg,
            alpha=alpha_cfg,
            text_processing=TextProcessingConfig(**data.get('text_processing', {})),
            backtest=BacktestConfig(**data.get('backtest', {})),
            rl=rl_cfg,
        )

    def to_yaml(self, path: Path):
        """Save configuration to YAML file"""
        data = {
            'universe': {
                'top_n': self.universe.top_n,
                'min_volume_usd': self.universe.min_volume_usd,
                'rebalance_hour_utc': self.universe.rebalance_hour_utc,
            },
            'binance': {
                'interval': self.binance.interval,
                'partition': self.binance.partition,
            },
            'nostr': {
                'relay': self.nostr.relay,
                'kinds': self.nostr.kinds,
                'partition': self.nostr.partition,
            },
            'gdelt': {
                'partition': self.gdelt.partition,
                'scrape_batch_size': self.gdelt.scrape_batch_size,
                'finbert_batch_size': self.gdelt.finbert_batch_size,
            },
            'alpha': {
                'alpha_101_dir': str(self.alpha.alpha_101_dir),
                # 'alpha_191_dir' removed - only using Alpha 101
                'skip_alphas': self.alpha.skip_alphas,
            },
            'text_processing': {
                'aggregation_window': self.text_processing.aggregation_window,
                'cryptobert_model': self.text_processing.cryptobert_model,
                'finbert_model': self.text_processing.finbert_model,
                'batch_size': self.text_processing.batch_size,
            },
            'backtest': {
                'granularity': self.backtest.granularity,
                'train_start': self.backtest.train_start,
                'train_end': self.backtest.train_end,
                'val_start': self.backtest.val_start,
                'val_end': self.backtest.val_end,
                'test_start': self.backtest.test_start,
                'test_end': self.backtest.test_end,
            },
        }

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Default configuration instance
config = Config()
