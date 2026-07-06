#!/bin/bash
# Thin wrapper around compute_advantages_ensemble.py that:
#   1. picks up #GPUs automatically (or honours --nproc N)
#   2. picks a free master port
#   3. filters the noisy `[libdav1d @ 0x..] libdav1d 0.9.2` chatter from
#      stderr at the SHELL level — bulletproof, doesn't depend on the
#      Python-side fd-2 redirect installed inside the script.
#
# Usage:
#   bash run_compute_advantages_ensemble.sh                                # default config, all GPUs
#   bash run_compute_advantages_ensemble.sh --nproc 4                      # 4 GPUs
#   bash run_compute_advantages_ensemble.sh CONFIG_NAME [--nproc N] [HYDRA_OVERRIDES...]
#
# Examples:
#   bash run_compute_advantages_ensemble.sh
#   bash run_compute_advantages_ensemble.sh compute_advantages_ensemble --nproc 8
#   bash run_compute_advantages_ensemble.sh --nproc 1 advantage.batch_size=16

set -e

source switch_env openpi 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH=$(dirname $(dirname $(dirname $(dirname $(dirname "$SCRIPT_DIR")))))
export OFFLINE_RL_CONFIG="${REPO_PATH}/examples/offline_rl/config"
export PYTHONPATH=${REPO_PATH}:${PYTHONPATH}
cd "$SCRIPT_DIR"

# First positional arg = config name (unless it starts with --).
CONFIG_NAME="steam_compute_advantages_ensemble"
if [ $# -gt 0 ] && [[ "$1" != --* ]] && [[ "$1" != *=* ]]; then
    CONFIG_NAME="$1"
    shift
fi

NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
OVERRIDES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --nproc)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        *)
            OVERRIDES+=("$1")
            shift
            ;;
    esac
done

if [ "$NPROC_PER_NODE" -lt 1 ]; then
    NPROC_PER_NODE=1
fi

# Pick a free master port.
MASTER_PORT=${MASTER_PORT:-29500}
while netstat -tuln 2>/dev/null | grep -q ":$MASTER_PORT " || ss -tuln 2>/dev/null | grep -q ":$MASTER_PORT "; do
    MASTER_PORT=$((MASTER_PORT + 1))
    if [ $MASTER_PORT -gt 30000 ]; then
        MASTER_PORT=29500
        break
    fi
done

echo "=========================================="
echo "Ensemble advantage computation"
echo "=========================================="
echo "  Config:       $CONFIG_NAME"
echo "  GPUs:         $NPROC_PER_NODE"
echo "  Master port:  $MASTER_PORT"
if [ ${#OVERRIDES[@]} -gt 0 ]; then
    echo "  Overrides:    ${OVERRIDES[*]}"
fi
echo ""

# Drop libdav1d / libav lines from stderr at the shell level.  We use
# process substitution `2> >(grep -v ... >&2)` instead of a pipe so the
# program's exit code is preserved (a pipe would mask it with grep's exit).
if [ "$NPROC_PER_NODE" -eq 1 ]; then
    python compute_advantages_ensemble.py \
        --config-path "${OFFLINE_RL_CONFIG}" \
        --config-name "$CONFIG_NAME" \
        "${OVERRIDES[@]}" \
        2> >(grep -v --line-buffered -E "libdav1d" >&2)
else
    torchrun \
        --nproc_per_node="$NPROC_PER_NODE" \
        --master_port="$MASTER_PORT" \
        compute_advantages_ensemble.py \
        --config-path "${OFFLINE_RL_CONFIG}" \
        --config-name "$CONFIG_NAME" \
        "${OVERRIDES[@]}" \
        2> >(grep -v --line-buffered -E "libdav1d" >&2)
fi

echo ""
echo "=========================================="
echo "Done."
echo "=========================================="
