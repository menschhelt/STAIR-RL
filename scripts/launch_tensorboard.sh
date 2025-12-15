#!/bin/bash
# Launch TensorBoard for STAIR-RL training monitoring
#
# Usage:
#   ./scripts/launch_tensorboard.sh [checkpoint_dir]
#
# Example:
#   ./scripts/launch_tensorboard.sh checkpoints/run_20250115_143022
#
# If no directory specified, uses latest checkpoint directory

set -e

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default checkpoint directory
CHECKPOINT_DIR="${1:-$PROJECT_DIR/checkpoints}"

# Find tensorboard logs
if [ -d "$CHECKPOINT_DIR/phase1/tensorboard" ] || [ -d "$CHECKPOINT_DIR/phase2/tensorboard" ]; then
    echo "===================="
    echo "Launching TensorBoard"
    echo "===================="
    echo ""
    echo "Checkpoint directory: $CHECKPOINT_DIR"
    echo ""

    # Check if both phase1 and phase2 exist
    if [ -d "$CHECKPOINT_DIR/phase1/tensorboard" ] && [ -d "$CHECKPOINT_DIR/phase2/tensorboard" ]; then
        echo "Monitoring both Phase 1 (CQL-SAC) and Phase 2 (PPO-CVaR)"
        tensorboard --logdir_spec \
            phase1:$CHECKPOINT_DIR/phase1/tensorboard,phase2:$CHECKPOINT_DIR/phase2/tensorboard \
            --port 6006 --bind_all
    elif [ -d "$CHECKPOINT_DIR/phase1/tensorboard" ]; then
        echo "Monitoring Phase 1 (CQL-SAC)"
        tensorboard --logdir $CHECKPOINT_DIR/phase1/tensorboard \
            --port 6006 --bind_all
    else
        echo "Monitoring Phase 2 (PPO-CVaR)"
        tensorboard --logdir $CHECKPOINT_DIR/phase2/tensorboard \
            --port 6006 --bind_all
    fi
else
    echo "Error: No TensorBoard logs found in $CHECKPOINT_DIR"
    echo ""
    echo "Expected directories:"
    echo "  - $CHECKPOINT_DIR/phase1/tensorboard"
    echo "  - $CHECKPOINT_DIR/phase2/tensorboard"
    echo ""
    echo "Make sure you've run training first:"
    echo "  python scripts/run_training.py --phase 1 --steps 500000"
    exit 1
fi
