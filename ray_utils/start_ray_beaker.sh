#!/bin/bash
#
# start_ray_beaker.sh — All-Beaker Ray cluster using multi-replica discovery.
#
# Topology:
#   Replica 0 — Ray head + env worker (RemoteEnv → gRPC to robot)
#   Replica 1..N — Ray GPU workers (actor, rollout)
#
# Ray head discovery uses BEAKER_LEADER_REPLICA_HOSTNAME (set automatically
# by Beaker for multi-replica experiments).
#
# The local robot machine runs RobotServer with a reverse SSH tunnel
# to the head node. Set ROBOT_SERVER_URL=localhost:50051 in the Beaker env.
#
# This script is the entrypoint for each Beaker replica. It detects its role
# from BEAKER_REPLICA_RANK and either starts as head or joins as worker.
#
# Standalone usage (for submitting via gantry):
#   bash ray_utils/start_ray_beaker.sh [OPTIONS] -- [EXTRA_GANTRY_ARGS...]
#
# As Beaker entrypoint (called inside the container):
#   bash ray_utils/start_ray_beaker.sh --entrypoint --train-cmd "python ..."

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash ray_utils/start_ray_beaker.sh [OPTIONS] -- [EXTRA_GANTRY_ARGS...]

All-Beaker Ray cluster launcher (multi-replica pattern).

Submission mode (run from your local machine to submit Beaker jobs):
  --image IMAGE           Beaker image (default: shiruic/shirui-torch2.8.0_cuda12.8)
  --workspace WS          Beaker workspace (default: ai2/molmo-act)
  --cluster CLUSTER       Beaker cluster (repeatable)
  --budget BUDGET         Beaker budget account
  --priority PRIORITY     Job priority (low|normal|high|urgent)
  --name NAME             Experiment name
  --replicas N            Number of replicas (default: 3)
  --gpus N                GPUs per replica (default: 1)
  --venv NAME             Venv to activate inside the container
  --install CMD           Install command inside the container
  --weka MOUNT            Weka bucket mount (repeatable)
  --env KEY=VALUE         Extra env var (repeatable)
  --env-secret K=V        Beaker secret env var (repeatable)
  --train-cmd CMD         Training command to run on head node
  --ray-port PORT         Ray port (default: 6379)
  --show-logs             Stream Beaker logs
  --allow-dirty           Allow dirty git working directory
  --dry-run               Print commands without executing
  --help                  Show this help

Entrypoint mode (called inside Beaker container):
  --entrypoint            Run as Beaker replica entrypoint
  --train-cmd CMD         Training command (head only)
  --ray-port PORT         Ray port (default: 6379)
  --install CMD           Install command to run before starting Ray
  --venv NAME             Venv to activate

Extra gantry arguments can be passed after '--'.
USAGE
    exit 0
}

# --- Defaults ---
BEAKER_IMAGE="shiruic/shirui-torch2.8.0_cuda12.8"
WORKSPACE="ai2/molmo-act"
CLUSTERS=("ai2/ceres")
BUDGET=""
EXP_NAME="rlinf-remote"
PRIORITY="urgent"
REPLICAS=3
GPUS=1
VENV=""
INSTALL_CMD="uv pip install -e '.[open]'"
WEKA_MOUNTS=("oe-training-default:/mount/weka")
EXTRA_ENVS=("HF_HOME=/mount/weka/shiruic/hf_cache")
ENV_SECRETS=("HF_TOKEN=hf_token_shirui")
TRAIN_CMD=""
RAY_PORT=6379
SHOW_LOGS=""
DRY_RUN=""
ALLOW_DIRTY=""
ENTRYPOINT_MODE=""
EXTRA_GANTRY_ARGS=()
_CLUSTERS_SET="" _WEKA_SET="" _ENVS_SET="" _SECRETS_SET=""

# --- Parse CLI args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)           usage ;;
        --entrypoint)     ENTRYPOINT_MODE="true"; shift ;;
        --image)          BEAKER_IMAGE="$2"; shift 2 ;;
        --workspace)      WORKSPACE="$2"; shift 2 ;;
        --cluster)        if [ -z "$_CLUSTERS_SET" ]; then CLUSTERS=(); _CLUSTERS_SET=1; fi; CLUSTERS+=("$2"); shift 2 ;;
        --budget)         BUDGET="$2"; shift 2 ;;
        --priority)       PRIORITY="$2"; shift 2 ;;
        --name)           EXP_NAME="$2"; shift 2 ;;
        --replicas)       REPLICAS="$2"; shift 2 ;;
        --gpus)           GPUS="$2"; shift 2 ;;
        --venv)           VENV="$2"; shift 2 ;;
        --install)        INSTALL_CMD="$2"; shift 2 ;;
        --weka)           if [ -z "$_WEKA_SET" ]; then WEKA_MOUNTS=(); _WEKA_SET=1; fi; WEKA_MOUNTS+=("$2"); shift 2 ;;
        --env)            if [ -z "$_ENVS_SET" ]; then EXTRA_ENVS=(); _ENVS_SET=1; fi; EXTRA_ENVS+=("$2"); shift 2 ;;
        --env-secret)     if [ -z "$_SECRETS_SET" ]; then ENV_SECRETS=(); _SECRETS_SET=1; fi; ENV_SECRETS+=("$2"); shift 2 ;;
        --train-cmd)      TRAIN_CMD="$2"; shift 2 ;;
        --ray-port)       RAY_PORT="$2"; shift 2 ;;
        --show-logs)      SHOW_LOGS="true"; shift ;;
        --allow-dirty)    ALLOW_DIRTY="true"; shift ;;
        --dry-run)        DRY_RUN="true"; shift ;;
        --)               shift; EXTRA_GANTRY_ARGS+=("$@"); break ;;
        *)                echo "Unknown option: $1"; usage ;;
    esac
done


# ==========================================================================
# Entrypoint mode — runs inside the Beaker container
# ==========================================================================
if [ "$ENTRYPOINT_MODE" = "true" ]; then
    REPLICA_RANK="${BEAKER_REPLICA_RANK:-0}"
    export RLINF_NODE_RANK="$REPLICA_RANK"

    echo "=== Beaker Replica ${REPLICA_RANK} ==="

    # Optional venv activation
    if [ -n "$VENV" ]; then
        echo "Activating venv: ${VENV}"
        source switch_env "${VENV}" || true
    fi

    # Optional install
    if [ -n "$INSTALL_CMD" ]; then
        echo "Running install: ${INSTALL_CMD}"
        eval "$INSTALL_CMD"
    fi

    # Activate venv if present (uv sync creates .venv)
    if [ -f ".venv/bin/activate" ]; then
        echo "Activating .venv in $(pwd)"
        source .venv/bin/activate
    else
        echo "Warning: .venv/bin/activate not found in $(pwd)"
    fi

    if [ "$REPLICA_RANK" = "0" ]; then
        # --- Head node ---
        echo "Starting Ray head on port ${RAY_PORT}"
        ray start --head --port="${RAY_PORT}"

        if [ -n "$TRAIN_CMD" ]; then
            echo "Running training command: ${TRAIN_CMD}"
            eval "$TRAIN_CMD"
        else
            echo "No --train-cmd specified; head node idle. Blocking..."
            tail -f /dev/null
        fi
    else
        # --- Worker node ---
        # Resolve head hostname to IP
        HEAD_HOST="${BEAKER_LEADER_REPLICA_HOSTNAME:-}"
        if [ -z "$HEAD_HOST" ]; then
            echo "Error: BEAKER_LEADER_REPLICA_HOSTNAME not set. Not running in multi-replica mode?"
            exit 1
        fi

        echo "Resolving head hostname: ${HEAD_HOST}"
        HEAD_IP=""
        for i in $(seq 1 60); do
            HEAD_IP=$(getent hosts "$HEAD_HOST" 2>/dev/null | awk '{print $1}' | head -1) || true
            if [ -n "$HEAD_IP" ]; then
                break
            fi
            echo "  Waiting for head DNS resolution... (attempt ${i}/60)"
            sleep 5
        done

        if [ -z "$HEAD_IP" ]; then
            echo "Error: Could not resolve ${HEAD_HOST} after 5 minutes"
            exit 1
        fi

        echo "Head IP: ${HEAD_IP}"
        echo "Connecting to Ray head at ${HEAD_IP}:${RAY_PORT}"

        # Wait for Ray head to be ready
        for i in $(seq 1 30); do
            if ray start --address="${HEAD_IP}:${RAY_PORT}" 2>/dev/null; then
                echo "Successfully joined Ray cluster"
                break
            fi
            if [ "$i" = "30" ]; then
                echo "Error: Could not connect to Ray head after 30 attempts"
                exit 1
            fi
            echo "  Ray head not ready, retrying... (attempt ${i}/30)"
            sleep 10
        done

        # Block until head disconnects or this process is killed
        echo "Worker running. Monitoring Ray connection..."
        while ray status >/dev/null 2>&1; do
            sleep 30
        done
        echo "Ray head disconnected or cluster shut down. Exiting."
    fi

    exit 0
fi


# ==========================================================================
# Submission mode — run from local machine to submit Beaker experiment
# ==========================================================================
echo "=== RLinf All-Beaker Ray Cluster ==="
echo "Replicas: ${REPLICAS}"
echo "GPUs/replica: ${GPUS}"
echo ""

# Build the entrypoint command that runs inside each replica
ENTRYPOINT_CMD="bash ray_utils/start_ray_beaker.sh --entrypoint --ray-port ${RAY_PORT}"
[ -n "$VENV" ] && ENTRYPOINT_CMD+=" --venv ${VENV}"
[ -n "$INSTALL_CMD" ] && ENTRYPOINT_CMD+=" --install '${INSTALL_CMD}'"
[ -n "$TRAIN_CMD" ] && ENTRYPOINT_CMD+=" --train-cmd '${TRAIN_CMD}'"

gantry_args=(
    "gantry" "run" "--yes" "--no-python"
    "--replicas" "${REPLICAS}"
    "--gpus" "${GPUS}"
    "--host-networking"
    "--beaker-image" "$BEAKER_IMAGE"
    "--workspace" "$WORKSPACE"
)

[ -n "$EXP_NAME" ]    && gantry_args+=("--name" "$EXP_NAME")
[ -n "$BUDGET" ]       && gantry_args+=("--budget" "$BUDGET")
[ -n "$PRIORITY" ]     && gantry_args+=("--priority" "$PRIORITY")
[ -n "$SHOW_LOGS" ]    && gantry_args+=("--show-logs")
[ -n "$ALLOW_DIRTY" ]  && gantry_args+=("--allow-dirty")

for cluster in "${CLUSTERS[@]+"${CLUSTERS[@]}"}"; do
    gantry_args+=("--cluster" "$cluster")
done
for weka in "${WEKA_MOUNTS[@]+"${WEKA_MOUNTS[@]}"}"; do
    gantry_args+=("--weka" "$weka")
done
for env_var in "${EXTRA_ENVS[@]+"${EXTRA_ENVS[@]}"}"; do
    gantry_args+=("--env" "$env_var")
done
for secret in "${ENV_SECRETS[@]+"${ENV_SECRETS[@]}"}"; do
    gantry_args+=("--env-secret" "$secret")
done
for arg in "${EXTRA_GANTRY_ARGS[@]+"${EXTRA_GANTRY_ARGS[@]}"}"; do
    gantry_args+=("$arg")
done

gantry_args+=("--" "bash" "-c" "${ENTRYPOINT_CMD}")

echo "--- Submitting Beaker experiment ---"
if [ "$DRY_RUN" = "true" ]; then
    echo "[dry-run] Would execute:"
    printf '  %s\n' "${gantry_args[@]}"
    echo ""
else
    "${gantry_args[@]}"
fi
