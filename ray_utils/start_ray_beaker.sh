#!/bin/bash
#
# start_ray_docker.sh — Launch Ray cluster: local head + Beaker GPU workers via Gantry.
#
# Topology (from yam_ppo_openpi_topreward.yaml):
#   Node 0 — Local desktop (Ray head + rollout)
#   Node 1 — YAM robot controller (env worker)
#   Node 2 — Beaker GPU (actor training)
#   Node 3 — Beaker GPU (VLM planner / TOPReward)
#
# This script:
#   1. Starts Ray head on the local machine (node 0)
#   2. Submits two Beaker jobs (actor + vlm) that join the cluster over Tailscale
#
# Usage:
#   bash ray_utils/start_ray_docker.sh [OPTIONS] -- [EXTRA_GANTRY_ARGS...]

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash ray_utils/start_ray_docker.sh [OPTIONS] -- [EXTRA_GANTRY_ARGS...]

Start local Ray head and submit Beaker GPU worker jobs over Tailscale.

Options:
  --image IMAGE           Beaker image (default: shiruic/shirui-torch2.8.0_cuda12.8)
  --workspace WS          Beaker workspace (default: ai2/molmo-act)
  --cluster CLUSTER       Beaker cluster (repeatable)
  --budget BUDGET         Beaker budget account
  --priority PRIORITY     Job priority (low|normal|high|urgent)
  --name NAME             Experiment name prefix (-actor / -vlm suffix added)
  --venv NAME             Venv to activate inside the container (e.g. openpi)
  --install CMD           Install command to run inside the container
  --weka MOUNT            Weka bucket mount (repeatable, e.g. bucket:/mount)
  --env KEY=VALUE         Extra env var (repeatable)
  --env-secret K=V        Beaker secret env var (repeatable)
  --ray-head-ip IP        Head node Tailscale IP (default: auto-detect from tailscale0)
  --ray-port PORT         Ray port (default: 29500)
  --show-logs             Stream Beaker logs to stdout
  --allow-dirty           Allow submitting with a dirty git working directory
  --only-actor            Only submit the actor job
  --only-vlm              Only submit the VLM planner job
  --dry-run               Print commands without executing
  --help                  Show this help message

Extra gantry arguments can be passed after '--'.

Example:
  bash ray_utils/start_ray_docker.sh \
    --cluster ai2/ceres --budget ai2/molmo-act \
    --weka oe-training-default:/mount/weka \
    --env HF_HOME=/mount/weka/shiruic/hf_cache \
    --env-secret HF_TOKEN=hf_token_shirui \
    --priority urgent --allow-dirty
USAGE
    exit 0
}

# --- Defaults ---
BEAKER_IMAGE="shiruic/shirui-torch2.8.0_cuda12.8"
VENV=""
RAY_HEAD_IP=""
RAY_PORT=29500
WORKSPACE="ai2/molmo-act"
CLUSTERS=("ai2/ceres")
BUDGET=""
EXP_NAME=""
PRIORITY="urgent"
INSTALL_CMD="uv pip install -e '.[open]'"
WEKA_MOUNTS=("oe-training-default:/mount/weka")
EXTRA_ENVS=("HF_HOME=/mount/weka/shiruic/hf_cache")
ENV_SECRETS=("HF_TOKEN=hf_token_shirui")
SHOW_LOGS=""
DRY_RUN=""
ALLOW_DIRTY=""
ONLY_ACTOR=""
ONLY_VLM=""
EXTRA_GANTRY_ARGS=()

# Fixed for this topology
ACTOR_NODE_RANK=2
VLM_NODE_RANK=3
COMM_NET_DEVICES="tailscale0"
GPUS=1
SHARED_MEMORY="64GiB"

# --- Parse CLI args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)           usage ;;
        --image)          BEAKER_IMAGE="$2"; shift 2 ;;
        --workspace)      WORKSPACE="$2"; shift 2 ;;
        --cluster)        CLUSTERS+=("$2"); shift 2 ;;
        --budget)         BUDGET="$2"; shift 2 ;;
        --priority)       PRIORITY="$2"; shift 2 ;;
        --name)           EXP_NAME="$2"; shift 2 ;;
        --venv)           VENV="$2"; shift 2 ;;
        --install)        INSTALL_CMD="$2"; shift 2 ;;
        --weka)           WEKA_MOUNTS+=("$2"); shift 2 ;;
        --env)            EXTRA_ENVS+=("$2"); shift 2 ;;
        --env-secret)     ENV_SECRETS+=("$2"); shift 2 ;;
        --ray-head-ip)    RAY_HEAD_IP="$2"; shift 2 ;;
        --ray-port)       RAY_PORT="$2"; shift 2 ;;
        --show-logs)      SHOW_LOGS="true"; shift ;;
        --allow-dirty)    ALLOW_DIRTY="true"; shift ;;
        --only-actor)     ONLY_ACTOR="true"; shift ;;
        --only-vlm)       ONLY_VLM="true"; shift ;;
        --dry-run)        DRY_RUN="true"; shift ;;
        --)               shift; EXTRA_GANTRY_ARGS+=("$@"); break ;;
        *)                echo "Unknown option: $1"; usage ;;
    esac
done

# --- Validate ---
if [ -n "$ONLY_ACTOR" ] && [ -n "$ONLY_VLM" ]; then
    echo "Error: --only-actor and --only-vlm are mutually exclusive"
    exit 1
fi

if [ -z "$RAY_HEAD_IP" ]; then
    RAY_HEAD_IP=$(ip -4 addr show tailscale0 2>/dev/null | grep -oP 'inet \K[\d.]+' || true)
    if [ -z "$RAY_HEAD_IP" ]; then
        echo "Error: --ray-head-ip not set and could not detect tailscale0 IP on this machine"
        exit 1
    fi
    echo "Detected local Tailscale IP as Ray head: ${RAY_HEAD_IP}"
fi

# --- Helper: build entrypoint command for a given node rank ---
build_entrypoint() {
    local node_rank="$1"
    local cmd=""

    if [ -n "$VENV" ]; then
        cmd+="source switch_env ${VENV} && "
    fi

    cmd+="export RLINF_NODE_RANK=${node_rank} && "
    cmd+="export RLINF_COMM_NET_DEVICES=${COMM_NET_DEVICES} && "
    cmd+="NODE_IP=\$(tailscale ip -4 2>/dev/null || ip -4 addr show ${COMM_NET_DEVICES} 2>/dev/null | grep -oP 'inet \\\\K[\\\\d.]+' || hostname -I | awk '{print \\\$1}') && "
    cmd+="echo \"Local Tailscale IP: \${NODE_IP}\" && "
    cmd+="echo \"Connecting to Ray head at ${RAY_HEAD_IP}:${RAY_PORT}\" && "
    cmd+="ray start --address=${RAY_HEAD_IP}:${RAY_PORT} --node-ip-address=\${NODE_IP} --block"

    echo "$cmd"
}

# --- Helper: submit a Beaker job ---
submit_job() {
    local job_label="$1"
    local node_rank="$2"
    local entrypoint
    entrypoint=$(build_entrypoint "$node_rank")

    local gantry_args=(
        "gantry" "run" "--yes" "--no-python"
        "--gpus" "${GPUS}"
        "--shared-memory" "${SHARED_MEMORY}"
        "--host-networking"
        "--beaker-image" "$BEAKER_IMAGE"
        "--workspace" "$WORKSPACE"
    )

    [ -n "$EXP_NAME" ]    && gantry_args+=("--name" "${EXP_NAME}-${job_label}")
    [ -n "$BUDGET" ]       && gantry_args+=("--budget" "$BUDGET")
    [ -n "$PRIORITY" ]     && gantry_args+=("--priority" "$PRIORITY")
    [ -n "$INSTALL_CMD" ]  && gantry_args+=("--install" "$INSTALL_CMD")
    [ -n "$SHOW_LOGS" ]    && gantry_args+=("--show-logs")
    [ -n "$ALLOW_DIRTY" ]  && gantry_args+=("--allow-dirty")

    for cluster in "${CLUSTERS[@]+"${CLUSTERS[@]}"}"; do
        gantry_args+=("--cluster" "$cluster")
    done
    for weka in "${WEKA_MOUNTS[@]+"${WEKA_MOUNTS[@]}"}"; do
        gantry_args+=("--weka" "$weka")
    done

    gantry_args+=("--env" "RLINF_NODE_RANK=${node_rank}")
    gantry_args+=("--env" "RLINF_COMM_NET_DEVICES=${COMM_NET_DEVICES}")
    for env_var in "${EXTRA_ENVS[@]+"${EXTRA_ENVS[@]}"}"; do
        gantry_args+=("--env" "$env_var")
    done
    for secret in "${ENV_SECRETS[@]+"${ENV_SECRETS[@]}"}"; do
        gantry_args+=("--env-secret" "$secret")
    done

    for arg in "${EXTRA_GANTRY_ARGS[@]+"${EXTRA_GANTRY_ARGS[@]}"}"; do
        gantry_args+=("$arg")
    done

    gantry_args+=("--" "bash" "-c" "${entrypoint}")

    echo "--- Submitting ${job_label} job (node rank ${node_rank}) ---"

    if [ "$DRY_RUN" = "true" ]; then
        echo "[dry-run] Would execute:"
        printf '  %s\n' "${gantry_args[@]}"
        echo ""
    else
        "${gantry_args[@]}" &
    fi
}

# --- Start local Ray head (node 0) ---
echo "=== RLinf Ray Cluster ==="
echo "Ray head: ${RAY_HEAD_IP}:${RAY_PORT}"
echo ""

RAY_HEAD_ARGS=("--head" "--port=${RAY_PORT}" "--node-ip-address=${RAY_HEAD_IP}")

echo "--- Starting local Ray head node (node rank 0) ---"
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES="${COMM_NET_DEVICES}"
if [ "$DRY_RUN" = "true" ]; then
    echo "[dry-run] Would execute: ray stop && ray start ${RAY_HEAD_ARGS[*]}"
    echo ""
else
    ray stop 2>/dev/null || true
    ray start "${RAY_HEAD_ARGS[@]}"
fi

# --- Start local YAM worker (node 1) ---
echo "--- Starting local YAM worker (node rank 1) ---"
export RLINF_NODE_RANK=1
if [ "$DRY_RUN" = "true" ]; then
    echo "[dry-run] Would execute: ray start --address=${RAY_HEAD_IP}:${RAY_PORT} --node-ip-address=${RAY_HEAD_IP}"
    echo ""
else
    ray start "--address=${RAY_HEAD_IP}:${RAY_PORT}" "--node-ip-address=${RAY_HEAD_IP}"
fi

# --- Submit Beaker jobs ---
if [ -z "$ONLY_VLM" ]; then
    submit_job "actor" "$ACTOR_NODE_RANK"
fi

if [ -z "$ONLY_ACTOR" ]; then
    submit_job "vlm" "$VLM_NODE_RANK"
fi

if [ "$DRY_RUN" != "true" ]; then
    wait
fi
