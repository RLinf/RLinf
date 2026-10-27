#! /bin/bash
#SBATCH --job-name=supermimic
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

# Multi-node SuperMimic on Slurm. Each allocated node runs one Ray node; the
# driver runs on the first node and places one SuperMimic rank per GPU across
# all nodes. Partition, account, node count and GPU count come from your
# site's sbatch options, not from this script.
#
# Submit from the RLinf repo root with the environment already activated
# (sbatch exports it to the job). log_path must be on storage every node
# can write, because each rank writes its own checkpoint shard:
#
#   sbatch --nodes=2 --gpus-per-node=8 --partition=<partition> \
#       examples/embodiment/slurm_supermimic.sh supermimic_lafan1 \
#       env.motion_path=/shared/lafan1_g1 runner.logger.log_path=/shared/runs/lafan1
#
# Optional environment: GPUS_PER_NODE (default SLURM_GPUS_ON_NODE),
# RAY_PORT (default 6379), HEAD_IP (default the first address of the first
# node), RAY_HEAD_WAIT (seconds between the head and worker starts, default
# 10), RLINF_TIMEOUT (collective timeout in minutes), REPO_PATH (default the
# submit directory). NCCL interface variables such as NCCL_SOCKET_IFNAME
# are site specific; export them before sbatch when your network needs them.

set -euo pipefail

CONFIG_NAME=${1:?usage: sbatch [options] slurm_supermimic.sh <config-name> [overrides...]}
shift

REPO_PATH=${REPO_PATH:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -f "${REPO_PATH}/examples/embodiment/train_supermimic.py" ]; then
    echo "REPO_PATH=${REPO_PATH} is not an RLinf checkout; submit from the repo root or set REPO_PATH." >&2
    exit 1
fi
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}
if [ -z "${GPUS_PER_NODE}" ]; then
    echo "Set GPUS_PER_NODE or submit with --gpus-per-node." >&2
    exit 1
fi
RAY_PORT=${RAY_PORT:-6379}

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
NNODES=${#NODES[@]}
HEAD_NODE=${NODES[0]}
HEAD_IP=${HEAD_IP:-$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname --ip-address | awk '{print $1}')}
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
echo "nodes=${NNODES} gpus_per_node=${GPUS_PER_NODE} world_size=${WORLD_SIZE} head=${HEAD_NODE} (${HEAD_IP}:${RAY_PORT})"

cleanup() {
    # The blocking `ray start` steps still hold every node's task slot, so the
    # stop step must overlap them or it would wait behind them.
    srun --overlap --nodes="${NNODES}" --ntasks="${NNODES}" ray stop >/dev/null 2>&1 || true
    kill $(jobs -p) >/dev/null 2>&1 || true
}
trap cleanup EXIT

srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    env RLINF_NODE_RANK=0 \
    ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \
    --num-gpus="${GPUS_PER_NODE}" --block &
sleep "${RAY_HEAD_WAIT:-10}"
for ((i = 1; i < NNODES; i++)); do
    srun --nodes=1 --ntasks=1 -w "${NODES[$i]}" \
        env RLINF_NODE_RANK="${i}" \
        ray start --address="${HEAD_IP}:${RAY_PORT}" \
        --num-gpus="${GPUS_PER_NODE}" --block &
done

# RLinf's Cluster waits until cluster.num_nodes Ray nodes are alive.
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
bash "${REPO_PATH}/examples/embodiment/run_supermimic.sh" "${CONFIG_NAME}" \
    cluster.num_nodes="${NNODES}" \
    cluster.component_placement.supermimic="0-$((WORLD_SIZE - 1))" \
    "$@"
