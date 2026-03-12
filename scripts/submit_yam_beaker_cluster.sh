#!/bin/bash
#
# submit_yam_beaker_cluster.sh — Submit a Beaker job that starts Ray head and idles.
#
# This is the first half of a desktop-driven training workflow:
#   1. This script submits a Beaker job that starts Ray head with GPUs and waits.
#   2. The desktop joins the Ray cluster via join_beaker_cluster.sh and runs training.
#
# The env worker runs directly on the desktop with YAMEnv — no gRPC, no SSH tunnel,
# no RemoteEnv. The Beaker node only provides GPUs for actor/rollout workers.
#
# Supported configs (both use TOPReward, both need 3 GPUs):
#   yam_ppo_openpi           — 3 GPUs (actor + rollout + VLM TOPReward on Beaker)
#   yam_ppo_openpi_topreward — 3 GPUs (actor + rollout + VLM TOPReward + subtask planning)
#
# Prerequisites:
#   gantry installed: pip install beaker-gantry
#
# Usage:
#   bash scripts/submit_yam_beaker_cluster.sh [OPTIONS]

set -euo pipefail

# --- Defaults ---
CONFIG_NAME="yam_ppo_openpi"
EXP_NAME=""
GPUS=0  # 0 = auto-detect based on config
CLUSTER="ai2/ceres-cirrascale"
BUDGET=""
PRIORITY="urgent"
DRY_RUN=""
SHOW_LOGS=""
ALLOW_DIRTY=""

BEAKER_IMAGE="shiruic/shirui-torch2.8.0_cuda12.8"
WORKSPACE="ai2/molmo-act"
WEKA_MOUNT="oe-training-default:/weka/oe-training-default"
INSTALL_CMD="uv sync --extra embodied"
RAY_PORT=6379

usage() {
    cat <<'EOF'
Usage: bash scripts/submit_yam_beaker_cluster.sh [OPTIONS]

Submit a Beaker job that starts Ray head with GPUs and idles, waiting for a
desktop worker to join and run training via join_beaker_cluster.sh.

Supported configs:
  yam_ppo_openpi             3 GPUs (actor + rollout + VLM TOPReward)
  yam_ppo_openpi_topreward   3 GPUs (actor + rollout + VLM TOPReward + subtask planning)

Options:
  --config NAME         Hydra config name for GPU auto-detection (default: yam_ppo_openpi)
  --gpus N              GPUs (0 = auto based on config)
  --name NAME           Experiment name (default: rlinf-cluster-<config>)
  --cluster CLUSTER     Beaker cluster (default: ai2/ceres-cirrascale)
  --budget BUDGET       Beaker budget account
  --priority PRIORITY   Job priority (default: urgent)
  --show-logs           Stream Beaker logs after submission
  --allow-dirty         Allow dirty git working directory
  --dry-run             Print command without executing
  --help                Show this help

After submission:
  1. Check Beaker logs for '=== Tailscale IP ===' to get the container IP
  2. Join the cluster from your desktop:
       bash scripts/join_beaker_cluster.sh \
           --head-ip <tailscale-ip> \
           --config yam_ppo_openpi \
           --model-path /path/to/checkpoint \
           --task "pick and place"
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)         usage ;;
        --config)       CONFIG_NAME="$2"; shift 2 ;;
        --gpus)         GPUS="$2"; shift 2 ;;
        --name)         EXP_NAME="$2"; shift 2 ;;
        --cluster)      CLUSTER="$2"; shift 2 ;;
        --budget)       BUDGET="$2"; shift 2 ;;
        --priority)     PRIORITY="$2"; shift 2 ;;
        --show-logs)    SHOW_LOGS="true"; shift ;;
        --allow-dirty)  ALLOW_DIRTY="true"; shift ;;
        --dry-run)      DRY_RUN="true"; shift ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$EXP_NAME" ]; then
    EXP_NAME="rlinf-cluster-${CONFIG_NAME}"
fi

# --- Auto-detect GPU count from config ---
case "$CONFIG_NAME" in
    *topreward*|*staged*|yam_ppo_openpi)
        # All YAM configs use TOPReward and need 3 GPUs.
        [ "$GPUS" -eq 0 ] && GPUS=3
        ;;
    *)
        [ "$GPUS" -eq 0 ] && GPUS=2
        ;;
esac

# --- Build the entrypoint ---
# 1. Install and start Tailscale (userspace networking for containers).
# 2. Print Tailscale IP so user can join the cluster from the desktop.
# 3. Start Ray head and idle (no --train-cmd → start_ray_beaker.sh blocks).
INSTALL_CMD_B64=$(echo "$INSTALL_CMD" | base64 | tr -d '\n')

ENTRYPOINT_CMD="curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg -o /usr/share/keyrings/tailscale-archive-keyring.gpg"
ENTRYPOINT_CMD+=" && echo 'deb [signed-by=/usr/share/keyrings/tailscale-archive-keyring.gpg] https://pkgs.tailscale.com/stable/ubuntu jammy main' > /etc/apt/sources.list.d/tailscale.list"
ENTRYPOINT_CMD+=" && (apt-get update || true) && apt-get install -y tailscale"
ENTRYPOINT_CMD+=" && (nohup tailscaled --tun=userspace-networking --state=mem: > /dev/null 2>&1 &)"
ENTRYPOINT_CMD+=" && sleep 2"
ENTRYPOINT_CMD+=" && tailscale up --authkey=\${TAILSCALE_AUTHKEY} --hostname=beaker-\${BEAKER_REPLICA_RANK:-0} --accept-routes"
ENTRYPOINT_CMD+=" && echo '=== Tailscale IP ===' && tailscale ip -4 && echo '=================='"
ENTRYPOINT_CMD+=" && TAILSCALE_NODE_IP=\$(tailscale ip -4)"
ENTRYPOINT_CMD+=" && if ip addr add \${TAILSCALE_NODE_IP}/32 dev lo 2>/dev/null; then echo '=== lo alias added: Ray will advertise Tailscale IP ==='; RAY_NODE_IP_ARG=\"--node-ip \${TAILSCALE_NODE_IP}\"; else echo '=== WARNING: ip addr add failed (no CAP_NET_ADMIN) — Ray will use internal IP ==='; RAY_NODE_IP_ARG=''; fi"
ENTRYPOINT_CMD+=" && INSTALL_CMD_DECODED=\$(echo ${INSTALL_CMD_B64} | base64 -d)"
ENTRYPOINT_CMD+=" && bash ray_utils/start_ray_beaker.sh"
ENTRYPOINT_CMD+=" --entrypoint"
ENTRYPOINT_CMD+=" --ray-port ${RAY_PORT}"
ENTRYPOINT_CMD+=" \${RAY_NODE_IP_ARG}"
ENTRYPOINT_CMD+=" --install \"\${INSTALL_CMD_DECODED}\""

# --- Build gantry command ---
gantry_args=(
    gantry run --yes --no-python
    --replicas 1
    --gpus "${GPUS}"
    --host-networking
    --beaker-image "${BEAKER_IMAGE}"
    --workspace "${WORKSPACE}"
    --cluster "${CLUSTER}"
    --name "${EXP_NAME}"
    --priority "${PRIORITY}"
    --weka "${WEKA_MOUNT}"
    --env "HF_HOME=/weka/oe-training-default/shiruic/hf_cache"
    --env "EMBODIED_PATH=examples/embodiment"
    --env-secret "HF_TOKEN=hf_token_shirui"
    --env-secret "TAILSCALE_AUTHKEY=tailscale_authkey_shirui"
)

[ -n "$BUDGET" ]      && gantry_args+=("--budget" "$BUDGET")
[ -n "$SHOW_LOGS" ]   && gantry_args+=("--show-logs")
[ -n "$ALLOW_DIRTY" ] && gantry_args+=("--allow-dirty")

gantry_args+=("--" "bash" "-c" "${ENTRYPOINT_CMD}")

echo "=== Submit Beaker Ray Cluster (GPU idle) ==="
echo "Config:       ${CONFIG_NAME}"
echo "GPUs:         ${GPUS}"
echo "Cluster:      ${CLUSTER}"
echo ""
echo "The Beaker container will start Ray head and idle."
echo "No training command is sent — training runs from the desktop."
echo ""
echo "After job starts:"
echo "  1. Check logs for '=== Tailscale IP ===' to get the container IP"
echo "  2. Join from desktop:"
echo "       bash scripts/join_beaker_cluster.sh \\"
echo "           --head-ip <tailscale-ip> \\"
echo "           --config ${CONFIG_NAME} \\"
echo "           --model-path /path/to/checkpoint \\"
echo "           --task \"pick and place\""
echo ""

if [ "$DRY_RUN" = "true" ]; then
    echo "[dry-run] Would execute:"
    printf '  %s\n' "${gantry_args[@]}"
else
    "${gantry_args[@]}"
fi
