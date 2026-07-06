#! /bin/bash
set -x

tabs 4

CONFIG=$1
shift 1 2>/dev/null || true

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

E2E_CONFIG="${REPO_PATH}/tests/e2e_tests/offline_rl"

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

MASTER_PORT=${MASTER_PORT:-29500}
while netstat -tuln 2>/dev/null | grep -q ":$MASTER_PORT " || ss -tuln 2>/dev/null | grep -q ":$MASTER_PORT "; do
    MASTER_PORT=$((MASTER_PORT + 1))
    if [ $MASTER_PORT -gt 30000 ]; then
        MASTER_PORT=29500
        break
    fi
done

SCRIPT="${REPO_PATH}/examples/offline_rl/advantage_labeling/steam/process/compute_advantages_ensemble.py"

if [ "$NPROC_PER_NODE" -eq 1 ]; then
    python "${SCRIPT}" \
        --config-path "${E2E_CONFIG}" \
        --config-name "${CONFIG}" \
        "${OVERRIDES[@]}"
else
    torchrun \
        --nproc_per_node="$NPROC_PER_NODE" \
        --master_port="$MASTER_PORT" \
        "${SCRIPT}" \
        --config-path "${E2E_CONFIG}" \
        --config-name "${CONFIG}" \
        "${OVERRIDES[@]}"
fi
