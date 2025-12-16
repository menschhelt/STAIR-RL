# STAIR-RL: Semantic-Token Augmented Investment RL

> **Research in Progress**: This project is currently under active development as part of academic research. Results and methodology may be updated.

STAIR-RL is a reinforcement learning framework for cryptocurrency portfolio management that integrates multimodal data (price, news, social media, macroeconomic indicators) with a two-stage transfer learning approach.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STAIR-RL Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Binance   │  │    GDELT    │  │    Nostr    │  │  FRED/Macro │        │
│  │  Futures    │  │    News     │  │   Social    │  │    Data     │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         ▼                ▼                ▼                ▼               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Alpha 101  │  │   FinBERT   │  │ CryptoBERT  │  │    Macro    │        │
│  │   Factors   │  │  Embedding  │  │  Embedding  │  │  Features   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                                   │                                        │
│                                   ▼                                        │
│                    ┌──────────────────────────────┐                        │
│                    │   Hierarchical State Builder │                        │
│                    │   - Asset-level features     │                        │
│                    │   - Market-level features    │                        │
│                    │   - Text projections         │                        │
│                    └──────────────┬───────────────┘                        │
│                                   │                                        │
│         ┌─────────────────────────┼─────────────────────────┐              │
│         │                         │                         │              │
│         ▼                         ▼                         ▼              │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐         │
│  │   Phase 1   │          │   Phase 2   │          │  Backtest   │         │
│  │   CQL-SAC   │────▶     │  PPO-CVaR   │────▶     │   Engine    │         │
│  │  (Offline)  │          │  (Online)   │          │             │         │
│  └─────────────┘          └─────────────┘          └─────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Two-Stage Transfer Learning**: CQL-SAC for offline pre-training → PPO-CVaR for online fine-tuning
- **Multimodal State Representation**: Combines numerical (Alpha 101), textual (FinBERT/CryptoBERT), and macro features
- **Risk-Aware Optimization**: CVaR constraint with adaptive Lagrangian multiplier
- **Dynamic Universe Selection**: Daily rebalancing based on 24h quote volume (Top 20 by liquidity)

## Project Structure

```
stair-local/
├── agents/                 # RL agent implementations
│   ├── cql_sac.py         # Phase 1: Conservative Q-Learning with SAC
│   ├── ppo_cvar.py        # Phase 2: PPO with CVaR constraint
│   ├── networks.py        # Neural network architectures
│   └── hierarchical_state_builder.py
│
├── collectors/            # Data collection pipelines
│   ├── binance_futures.py # Binance USDT-M Perpetual Futures
│   ├── gdelt_local.py     # GDELT news data
│   ├── nostr_local.py     # Nostr social media data
│   └── fred_local.py      # FRED macroeconomic data
│
├── features/              # Feature engineering
│   ├── alpha_calculator.py
│   ├── cross_sectional_calculator.py
│   ├── fama_french.py     # Crypto-adapted Fama-French factors
│   └── macro_loader.py
│
├── environments/          # Trading environment
│   ├── trading_env.py     # Gym-compatible environment
│   └── position_sizer.py  # Kelly-criterion position sizing
│
├── backtesting/           # Backtesting framework
│   ├── engine.py
│   └── metrics.py
│
├── benchmarks/            # Benchmark strategies
│   ├── equal_weight.py
│   ├── markowitz.py       # Mean-Variance Optimization
│   └── fingpt_mvo.py      # FinGPT-enhanced MVO
│
├── scripts/               # Execution scripts
│   ├── run_training.py    # Main training script
│   ├── run_backtest.py    # Backtesting script
│   └── run_benchmark.py   # Benchmark comparison
│
└── config/
    └── settings.py        # Configuration management
```

## Data Pipeline

### 1. Price Data (Binance Futures)
- **Source**: Binance USDT-M Perpetual Futures
- **Collection**: Via `freqtrade` framework
- **Format**: 5-minute OHLCV candles
- **Period**: 2020-01 ~ 2025-12
- **Processing**: Alpha 101 factors with cross-sectional normalization

### 2. News Data (GDELT)
- **Source**: GDELT Global Knowledge Graph
- **Collection**: Weekly batch processing
- **Embedding**: FinBERT (768-dim → 64-dim projection)
- **Period**: 2020-01 ~ 2025-12

### 3. Social Data (Nostr)
- **Source**: Nostr decentralized social network
- **Collection**: Multi-relay aggregation
- **Embedding**: CryptoBERT (768-dim → 64-dim projection)
- **Period**: 2020-01 ~ 2025-12

### 4. Macroeconomic Data (FRED)
- **Source**: Federal Reserve Economic Data
- **Indicators**: Interest rates, inflation, VIX, DXY, etc.
- **Period**: 2020-01 ~ 2025-12

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/stair-local.git
cd stair-local

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Phase 1: CQL-SAC (Offline Pre-training)
python scripts/run_training.py --phase 1 --config config/training_config.yaml

# Phase 2: PPO-CVaR (Online Fine-tuning)
python scripts/run_training.py --phase 2 --resume checkpoints/phase1/cql_sac_final.pt
```

### Backtesting

```bash
python scripts/run_backtest.py \
  --model checkpoints/phase2/ppo_cvar_final.pt \
  --start 2025-01-01 \
  --end 2025-11-30 \
  --initial-nav 100000 \
  --output backtest_results/
```

### Benchmark Comparison

```bash
python scripts/run_benchmark.py \
  --strategy all \
  --start 2025-01-01 \
  --end 2025-11-30
```

## Data Availability

The preprocessed dataset used in this research is available for download:

**Google Drive**: [Download Dataset (~116GB)](https://drive.google.com/file/d/1vkXEy_cydDG5UwECX7WZFwl4USXZQdQJ/view?usp=sharing)

### Dataset Contents

| Directory | Size | Description |
|-----------|------|-------------|
| `binance/` | 2.6 GB | 72 monthly Parquet files (2020.01-2025.12), 5min OHLCV |
| `gdelt/` | 1.5 GB | 306 weekly Parquet files, news data |
| `nostr/` | 240 MB | 256 weekly Parquet files, social data |
| `macro/` | 4.8 MB | 406 monthly files, macroeconomic indicators |
| `embeddings/` | 354 MB | Pre-computed FinBERT/CryptoBERT embeddings (H5) |
| `features/` | 111 GB | Alpha 101 cache (raw + normalized) |
| `universe/` | 420 KB | Daily universe history |

## Hardware Requirements

- **GPU**: NVIDIA A100 40GB (or equivalent with 8GB+ VRAM)
- **RAM**: 32GB+ recommended
- **Storage**: 200GB+ for full dataset

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{stair-rl2024,
  title={STAIR-RL: Semantic-Token Augmented Investment Reinforcement Learning},
  author={},
  year={2024},
  note={Research in progress}
}
```

## License

This project is for research purposes only. Commercial use is not permitted without explicit permission.

## Acknowledgments

- Alpha 101 formulas based on Kakushadze (2016) "101 Formulaic Alphas"
- FinBERT model from ProsusAI
- CryptoBERT model from kk08/CryptoBERT
