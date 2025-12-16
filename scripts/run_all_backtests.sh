#!/bin/bash
# ============================================
# STAIR-RL Backtest Execution Script
# ============================================
#
# 학습 완료 후 이 스크립트를 실행하여 백테스트를 수행합니다.
#
# Usage:
#   chmod +x scripts/run_all_backtests.sh
#   ./scripts/run_all_backtests.sh
#
# ============================================

set -e

STAIR_DIR="/home/work/data/RL/stair-local"
cd "$STAIR_DIR"

# 테스트 기간: 2025년 1월 - 2025년 11월
TEST_START="2025-01-01"
TEST_END="2025-11-30"
INITIAL_NAV=100000

echo "============================================"
echo "STAIR-RL Backtest Suite"
echo "Test Period: $TEST_START to $TEST_END"
echo "Initial NAV: \$${INITIAL_NAV}"
echo "============================================"
echo ""

# ============================================
# 1. Transfer Learning Model (Phase 2) Backtest
# ============================================

TRANSFER_MODEL="checkpoints/phase2_warmup_fixed/phase2/ppo_cvar_final.pt"
TRANSFER_OUTPUT="backtest_results/transfer_learning"

if [ -f "$TRANSFER_MODEL" ]; then
    echo "[1/2] Running Transfer Learning Model Backtest..."
    python scripts/run_backtest.py \
        --model "$TRANSFER_MODEL" \
        --start "$TEST_START" \
        --end "$TEST_END" \
        --initial-nav $INITIAL_NAV \
        --output "$TRANSFER_OUTPUT" \
        --gpu 0
    echo "Transfer Learning backtest complete. Results: $TRANSFER_OUTPUT"
else
    echo "[1/2] SKIPPED: Transfer Learning model not found ($TRANSFER_MODEL)"
    echo "       Run Phase 2 training to completion first."

    # 가장 최신 checkpoint로 대체 시도
    LATEST_CKPT=$(ls -t checkpoints/phase2_warmup_fixed/phase2/ppo_cvar_step_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "       Latest checkpoint available: $LATEST_CKPT"
        echo "       To run with latest checkpoint:"
        echo "         python scripts/run_backtest.py --model $LATEST_CKPT --start $TEST_START --end $TEST_END --output $TRANSFER_OUTPUT"
    fi
fi

echo ""

# ============================================
# 2. PPO-Only Baseline Backtest
# ============================================

PPO_ONLY_MODEL="checkpoints/*/ppo_only/ppo_only_final.pt"
PPO_ONLY_OUTPUT="backtest_results/ppo_only"

# Find PPO-only model (could be in any checkpoint directory)
PPO_ONLY_FOUND=$(ls $PPO_ONLY_MODEL 2>/dev/null | head -1)

if [ -n "$PPO_ONLY_FOUND" ]; then
    echo "[2/2] Running PPO-Only Baseline Backtest..."
    python scripts/run_backtest.py \
        --model "$PPO_ONLY_FOUND" \
        --start "$TEST_START" \
        --end "$TEST_END" \
        --initial-nav $INITIAL_NAV \
        --output "$PPO_ONLY_OUTPUT" \
        --gpu 0
    echo "PPO-Only backtest complete. Results: $PPO_ONLY_OUTPUT"
else
    echo "[2/2] SKIPPED: PPO-Only model not found"
    echo "       Run PPO-only training first:"
    echo "         python scripts/run_training.py --ppo-only --steps 500000"
fi

echo ""
echo "============================================"
echo "Backtest Suite Complete"
echo "============================================"

# ============================================
# 결과 비교 (둘 다 완료된 경우)
# ============================================

if [ -f "$TRANSFER_OUTPUT/metrics.json" ] && [ -f "$PPO_ONLY_OUTPUT/metrics.json" ]; then
    echo ""
    echo "=== Performance Comparison ==="
    echo ""
    echo "Transfer Learning Model:"
    cat "$TRANSFER_OUTPUT/metrics.json" | python -c "import sys,json; d=json.load(sys.stdin); print(f'  Sharpe: {d.get(\"sharpe_ratio\", \"N/A\"):.3f}, Annual Return: {d.get(\"annual_return\", 0)*100:.2f}%, Max DD: {d.get(\"max_drawdown\", 0)*100:.2f}%')"
    echo ""
    echo "PPO-Only Baseline:"
    cat "$PPO_ONLY_OUTPUT/metrics.json" | python -c "import sys,json; d=json.load(sys.stdin); print(f'  Sharpe: {d.get(\"sharpe_ratio\", \"N/A\"):.3f}, Annual Return: {d.get(\"annual_return\", 0)*100:.2f}%, Max DD: {d.get(\"max_drawdown\", 0)*100:.2f}%')"
fi
