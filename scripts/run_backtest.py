#!/usr/bin/env python3
"""
Backtest Script.

Runs backtest on a trained STAIR-RL model:
- Load trained model checkpoint
- Run on test period (2024.01 - present)
- Calculate performance metrics
- Generate report

Usage:
    python scripts/run_backtest.py --model checkpoints/ppo_cvar_final.pt
    python scripts/run_backtest.py --model checkpoints/ppo_cvar_final.pt --start 2024-01-01 --end 2024-06-30
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config, DATA_DIR, BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_backtest(
    model_path: Path,
    config: Config,
    start_date: str,
    end_date: str,
    initial_nav: float,
    output_dir: Path,
    device: torch.device,
):
    """
    Run backtest with trained model.

    Args:
        model_path: Path to trained model checkpoint
        config: Configuration
        start_date: Backtest start date
        end_date: Backtest end date
        initial_nav: Initial portfolio value
        output_dir: Directory for outputs
        device: Torch device
    """
    from agents.ppo_cvar import PPOCVaRAgent
    from environments.trading_env import TradingEnv, EnvConfig
    from backtesting.data_loader import BacktestDataLoader
    from backtesting.engine import BacktestEngine
    from backtesting.metrics import compute_all_metrics

    logger.info("=" * 60)
    logger.info("STAIR-RL Backtest")
    logger.info(f"Model: {model_path}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Initial NAV: ${initial_nav:,.0f}")
    logger.info("=" * 60)

    # Load test data
    logger.info("Loading test data...")
    data_loader = BacktestDataLoader(
        data_dir=DATA_DIR,
        feature_dir=DATA_DIR / 'features',
    )

    test_data = data_loader.load_period(
        start_date=start_date,
        end_date=end_date,
    )
    logger.info(f"Test data loaded: {len(test_data)} rows")

    if len(test_data) == 0:
        logger.error("No test data available")
        return None

    # Create environment
    env_config = EnvConfig(
        n_assets=config.universe.top_n,
        target_leverage=config.rl.target_leverage,
        transaction_cost_rate=config.backtest.taker_fee + config.backtest.slippage,
        initial_nav=initial_nav,
    )

    env = TradingEnv(
        data=test_data,
        config=env_config,
    )

    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Load agent
    logger.info(f"Loading model from {model_path}...")
    agent = PPOCVaRAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )
    agent.load(model_path)
    agent.eval()  # Set to evaluation mode

    # Run backtest
    logger.info("Running backtest...")

    nav_history = []
    returns_history = []
    weights_history = []
    actions_history = []

    state, info = env.reset()
    done = False
    step = 0

    while not done:
        # Get action from policy (no exploration)
        with torch.no_grad():
            action = agent.get_action_deterministic(state)

        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record history
        nav_history.append({
            'timestamp': info.get('timestamp'),
            'nav': info.get('nav', initial_nav),
            'step': step,
        })

        returns_history.append({
            'timestamp': info.get('timestamp'),
            'return': info.get('portfolio_return', 0),
            'step': step,
        })

        weights = info.get('weights', np.zeros(action_dim))
        weight_dict = {'timestamp': info.get('timestamp'), 'step': step}
        for i, w in enumerate(weights):
            weight_dict[f'w_{i}'] = w
        weights_history.append(weight_dict)

        actions_history.append({
            'timestamp': info.get('timestamp'),
            'action': action.tolist() if hasattr(action, 'tolist') else action,
            'step': step,
        })

        state = next_state
        step += 1

        if step % 1000 == 0:
            current_nav = nav_history[-1]['nav']
            logger.info(f"Step {step}: NAV = ${current_nav:,.0f}")

    # Convert to DataFrames
    nav_df = pd.DataFrame(nav_history)
    if 'timestamp' in nav_df.columns:
        nav_df = nav_df.set_index('timestamp')

    returns_df = pd.DataFrame(returns_history)
    if 'timestamp' in returns_df.columns:
        returns_df = returns_df.set_index('timestamp')

    weights_df = pd.DataFrame(weights_history)
    if 'timestamp' in weights_df.columns:
        weights_df = weights_df.set_index('timestamp')

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_all_metrics(
        nav_series=nav_df['nav'],
        returns_series=returns_df['return'],
        initial_nav=initial_nav,
        periods_per_year=252 * 24 * 12 if config.backtest.granularity == '5m' else 252,
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Return: {metrics['total_return']*100:.2f}%")
    logger.info(f"Annual Return: {metrics['annual_return']*100:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    logger.info(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    logger.info(f"CVaR (95%): {metrics['cvar_95']*100:.2f}%")
    logger.info(f"Volatility: {metrics['volatility']*100:.2f}%")
    logger.info(f"Total Turnover: {metrics.get('total_turnover', 0):.2f}")
    logger.info("=" * 60)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {metrics_path}")

    # Save time series
    nav_df.to_parquet(output_dir / 'nav_series.parquet')
    returns_df.to_parquet(output_dir / 'returns_series.parquet')
    weights_df.to_parquet(output_dir / 'weights_history.parquet')
    logger.info(f"Time series saved to {output_dir}")

    # Generate report
    generate_report(metrics, nav_df, returns_df, output_dir)

    return metrics


def generate_report(metrics: dict, nav_df: pd.DataFrame, returns_df: pd.DataFrame, output_dir: Path):
    """Generate markdown report."""
    report_path = output_dir / 'backtest_report.md'

    lines = [
        "# STAIR-RL Backtest Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",

        "## Performance Summary\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Return | {metrics['total_return']*100:.2f}% |",
        f"| Annual Return | {metrics['annual_return']*100:.2f}% |",
        f"| Sharpe Ratio | {metrics['sharpe_ratio']:.3f} |",
        f"| Sortino Ratio | {metrics['sortino_ratio']:.3f} |",
        f"| Max Drawdown | {metrics['max_drawdown']*100:.2f}% |",
        f"| Calmar Ratio | {metrics['calmar_ratio']:.3f} |",
        f"| CVaR (95%) | {metrics['cvar_95']*100:.2f}% |",
        f"| Volatility | {metrics['volatility']*100:.2f}% |",
        "",

        "## NAV Statistics\n",
        f"- Starting NAV: ${nav_df['nav'].iloc[0]:,.0f}",
        f"- Ending NAV: ${nav_df['nav'].iloc[-1]:,.0f}",
        f"- Peak NAV: ${nav_df['nav'].max():,.0f}",
        f"- Trough NAV: ${nav_df['nav'].min():,.0f}",
        "",

        "## Return Distribution\n",
        f"- Mean Daily Return: {returns_df['return'].mean()*100:.4f}%",
        f"- Std Daily Return: {returns_df['return'].std()*100:.4f}%",
        f"- Skewness: {returns_df['return'].skew():.3f}",
        f"- Kurtosis: {returns_df['return'].kurtosis():.3f}",
        f"- Best Day: {returns_df['return'].max()*100:.2f}%",
        f"- Worst Day: {returns_df['return'].min()*100:.2f}%",
        "",
    ]

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run backtest with trained STAIR-RL model'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--start', type=str, default=None,
        help='Start date (YYYY-MM-DD), defaults to test_start from config'
    )
    parser.add_argument(
        '--end', type=str, default=None,
        help='End date (YYYY-MM-DD), defaults to today'
    )
    parser.add_argument(
        '--initial-nav', type=float, default=100000.0,
        help='Initial portfolio value'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU ID to use'
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        config = Config.from_yaml(Path(args.config))
    else:
        config = Config()

    # Set dates
    start_date = args.start or config.backtest.test_start
    end_date = args.end or datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = BASE_DIR / 'backtest_results' / f'run_{timestamp}'

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    # Run backtest
    run_backtest(
        model_path=Path(args.model),
        config=config,
        start_date=start_date,
        end_date=end_date,
        initial_nav=args.initial_nav,
        output_dir=output_dir,
        device=device,
    )


if __name__ == '__main__':
    main()
