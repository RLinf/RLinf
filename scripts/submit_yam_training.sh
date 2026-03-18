#!/bin/bash
#
# submit_yam_training.sh — Submit YAM training to Beaker.
#
# Supports two configs (both use TOPReward, both need 3 GPUs):
#   yam_ppo_openpi                  — PPO + π₀.5 + TOPReward (no subtask planning)
#   yam_ppo_openpi_topreward        — PPO + π₀.5 + TOPReward + subtask planning
#
# Topology (both configs, single Beaker node):
#   GPU 0 — actor (FSDP training)
#   GPU 1 — rollout (inference)
#   GPU 2 — VLM planner (Qwen3-VL-8B, TOPReward scoring; + subtask for topreward variant)
#   CPU   — RemoteEnv (gRPC to robot server via SSH tunnel)
#
# The robot server runs on the local desktop with a reverse SSH tunnel
# to the Beaker head node.
#
# Prerequisites:
#   1. Robot server + reverse SSH tunnel running on desktop:
#        bash scripts/start_robot_server.sh --config examples/embodiment/config/env/yam.yaml \
#            --remote-host <beaker-head-tailscale-ip> [--dummy]
#   2. gantry installed: pip install beaker-gantry
#
# Usage:
#   bash scripts/submit_yam_training.sh [OPTIONS] [-- HYDRA_OVERRIDES...]

set -euo pipefail

# --- Defaults ---
CONFIG_NAME="yam_ppo_openpi"
MODEL_PATH=""
TASK_DESC="pick and place"
EXP_NAME=""
REPLICAS=1
GPUS=0  # 0 = auto-detect based on config
CLUSTER="ai2/ceres-cirrascale"
BUDGET=""
PRIORITY="urgent"
DRY_RUN=""
SHOW_LOGS=""
ALLOW_DIRTY=""
EXTRA_OVERRIDES=()

INTERACTIVE=""
REPO_DIR="/weka/oe-training-default/shiruic/RLinf"

BEAKER_IMAGE="shiruic/shirui-torch2.8.0_cuda12.8"
WORKSPACE="ai2/molmo-act"
WEKA_MOUNT="oe-training-default:/weka/oe-training-default"
INSTALL_CMD="bash requirements/install.sh embodied --model openpi --env remote"
RAY_PORT=6379

usage() {
    cat <<'EOF'
Usage: bash scripts/submit_yam_training.sh [OPTIONS] [-- HYDRA_OVERRIDES...]

Submit YAM training to Beaker with automatic component placement.

Supported configs (both use TOPReward, both require 3 GPUs):
  yam_ppo_openpi                  3 GPUs — TOPReward only, no subtask planning
  yam_ppo_openpi_topreward        3 GPUs — TOPReward + VLM subtask planning

Options:
  --config NAME         Hydra config name (default: yam_ppo_openpi)
  --model-path PATH     Path to model checkpoint (local or HuggingFace ID)
  --task DESC           Task description (default: "pick and place")
  --name NAME           Experiment name (default: rlinf-<config>)
  --replicas N          Number of Beaker replicas (default: 1)
  --gpus N              GPUs per replica (0 = auto based on config)
  --cluster CLUSTER     Beaker cluster (default: ai2/ceres-cirrascale)
  --workspace WORKSPACE Beaker workspace (default: ai2/molmo-act)
  --budget BUDGET       Beaker budget account
  --priority PRIORITY   Job priority (default: urgent)
  --interactive         Create an interactive Beaker session with Ray head started and no training.
                        Attach from the cluster with: beaker session attach <session-id>
  --show-logs           Stream Beaker logs after submission
  --allow-dirty         Allow dirty git working directory
  --dry-run             Print command without executing
  --help                Show this help

Extra Hydra overrides can be passed after '--':
  bash scripts/submit_yam_training.sh --model-path thomas0829/folding_towel_pi05 -- algorithm.update_epoch=2

After submission:
  1. Check Beaker logs for the head node's Tailscale IP
  2. Start robot server with reverse SSH tunnel to that IP:
       bash scripts/start_robot_server.sh --config .../yam.yaml \
           --remote-host <head-tailscale-ip> [--dummy]
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)         usage ;;
        --config)       CONFIG_NAME="$2"; shift 2 ;;
        --model-path)   MODEL_PATH="$2"; shift 2 ;;
        --task)         TASK_DESC="$2"; shift 2 ;;
        --name)         EXP_NAME="$2"; shift 2 ;;
        --replicas)     REPLICAS="$2"; shift 2 ;;
        --gpus)         GPUS="$2"; shift 2 ;;
        --cluster)      CLUSTER="$2"; shift 2 ;;
        --workspace)    WORKSPACE="$2"; shift 2 ;;
        --budget)       BUDGET="$2"; shift 2 ;;
        --priority)     PRIORITY="$2"; shift 2 ;;
        --show-logs)    SHOW_LOGS="true"; shift ;;
        --allow-dirty)  ALLOW_DIRTY="true"; shift ;;
        --interactive)  INTERACTIVE="true"; shift ;;
        --dry-run)      DRY_RUN="true"; shift ;;
        --)             shift; EXTRA_OVERRIDES=("$@"); break ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$EXP_NAME" ]; then
    if [ -n "$INTERACTIVE" ]; then
        EXP_NAME="rlinf-${CONFIG_NAME}-interactive"
    else
        EXP_NAME="rlinf-${CONFIG_NAME}"
    fi
fi

# --- Detect config type and set GPU count / entry point ---
IS_TOPREWARD=false
ENTRY_SCRIPT="train_embodied_agent.py"

case "$CONFIG_NAME" in
    *topreward*|*staged*|yam_ppo_openpi)
        # All YAM configs use TOPReward → 3 GPUs, staged entry point.
        # yam_ppo_openpi uses TOPReward with subtask_interval=0 (no subtask planning).
        # yam_ppo_openpi_topreward also enables subtask planning (subtask_interval=3).
        IS_TOPREWARD=true
        ENTRY_SCRIPT="train_embodied_agent_staged.py"
        [ "$GPUS" -eq 0 ] && GPUS=3
        ;;
    *)
        [ "$GPUS" -eq 0 ] && GPUS=2
        ;;
esac

if [ -n "$INTERACTIVE" ]; then
    # ----------------------------------------------------------------
    # Interactive mode: same startup flow as submit_yam_beaker_cluster.sh,
    # but submitted as a Beaker session instead of a Beaker job.
    # ----------------------------------------------------------------

    INSTALL_CMD_B64=$(echo "$INSTALL_CMD" | base64 | tr -d '\n')

    ENTRYPOINT_CMD="curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg -o /usr/share/keyrings/tailscale-archive-keyring.gpg"
    ENTRYPOINT_CMD+=" && echo 'deb [signed-by=/usr/share/keyrings/tailscale-archive-keyring.gpg] https://pkgs.tailscale.com/stable/ubuntu jammy main' > /etc/apt/sources.list.d/tailscale.list"
    ENTRYPOINT_CMD+=" && (apt-get update || true) && apt-get install -y tailscale"
    ENTRYPOINT_CMD+=" && (nohup tailscaled --tun=userspace-networking --state=mem: > /dev/null 2>&1 &)"
    ENTRYPOINT_CMD+=" && sleep 2"
    ENTRYPOINT_CMD+=" && tailscale up --authkey=\"\${TAILSCALE_AUTHKEY}\" --advertise-tags=tag:robo-beaker --hostname=beaker-\${BEAKER_REPLICA_RANK:-0} --accept-routes"
    ENTRYPOINT_CMD+=" && echo '=== Tailscale IP ===' && tailscale ip -4 && echo '=================='"
    ENTRYPOINT_CMD+=" && TAILSCALE_NODE_IP=\$(tailscale ip -4)"
    ENTRYPOINT_CMD+=" && if ip addr add \${TAILSCALE_NODE_IP}/32 dev lo 2>/dev/null; then echo '=== lo alias added: Ray will advertise Tailscale IP ==='; RAY_NODE_IP_ARG=\"--node-ip \${TAILSCALE_NODE_IP}\"; else echo '=== WARNING: ip addr add failed (no CAP_NET_ADMIN) — Ray will use internal IP ==='; RAY_NODE_IP_ARG=''; fi"
    ENTRYPOINT_CMD+=" && mkdir -p /root/.ssh && chmod 700 /root/.ssh"
    ENTRYPOINT_CMD+=" && ssh-keygen -t ed25519 -f /root/.ssh/container_key -N '' -C beaker-container"
    ENTRYPOINT_CMD+=" && chmod 600 /root/.ssh/container_key"
    ENTRYPOINT_CMD+=" && echo '=== Container SSH Public Key (add to your desktop ~/.ssh/authorized_keys) ==='"
    ENTRYPOINT_CMD+=" && cat /root/.ssh/container_key.pub"
    ENTRYPOINT_CMD+=" && echo '=================='"
    ENTRYPOINT_CMD+=" && INSTALL_CMD_DECODED=\$(echo ${INSTALL_CMD_B64} | base64 -d)"
    ENTRYPOINT_CMD+=" && export RAY_health_check_period_ms=3600000"
    ENTRYPOINT_CMD+=" && cd ${REPO_DIR}"
    ENTRYPOINT_CMD+=" && bash ray_utils/start_ray_beaker.sh"
    ENTRYPOINT_CMD+=" --entrypoint"
    ENTRYPOINT_CMD+=" --interactive-shell"
    ENTRYPOINT_CMD+=" --ray-port ${RAY_PORT}"
    ENTRYPOINT_CMD+=" \${RAY_NODE_IP_ARG}"
    ENTRYPOINT_CMD+=" --install \"\${INSTALL_CMD_DECODED}\""

    session_args=(
        beaker session create
        --remote --bare
        --cluster "${CLUSTER}"
        --gpus "${GPUS}"
        --workspace "${WORKSPACE}"
        --priority "${PRIORITY}"
        --name "${EXP_NAME}"
        --mount "src=weka,ref=oe-training-default,dst=/weka/oe-training-default"
        --image "beaker://${BEAKER_IMAGE}"
        --workdir "${REPO_DIR}"
        --env "HF_HOME=/weka/oe-training-default/shiruic/hf_cache"
        --env "EMBODIED_PATH=examples/embodiment"
        --env "RAY_health_check_failure_threshold=10"
        --env "RAY_health_check_timeout_ms=30000"
        --secret-env "HF_TOKEN=hf_token_shirui"
        --secret-env "TAILSCALE_AUTHKEY=tailscale_authkey_shirui"
        --
        bash -c "${ENTRYPOINT_CMD}"
    )

    [ -n "$BUDGET" ] && session_args+=("--budget" "$BUDGET")

    echo "=== Submit Interactive Beaker Session ==="
    echo "Config:       ${CONFIG_NAME}"
    echo "GPUs:         ${GPUS}"
    echo "Cluster:      ${CLUSTER}"
    echo "Workspace:    ${WORKSPACE}"
    echo "Priority:     ${PRIORITY}"
    echo "Image:        ${BEAKER_IMAGE}"
    echo "Repo:         ${REPO_DIR}"
    echo ""

    if [ "$DRY_RUN" = "true" ]; then
        echo "[dry-run] Would execute:"
        printf '  %s\n' "${session_args[@]}"
    else
        "${session_args[@]}"
        echo ""
        echo "Setup running in background (Tailscale → install → Ray head → interactive shell)."
        echo "Check session logs, then attach from the cluster:"
        echo "  beaker session attach <session-id>"
    fi

else
    # ----------------------------------------------------------------
    # Training mode: existing gantry submission (unchanged)
    # ----------------------------------------------------------------

    # --- Build the training command (runs on head node only) ---
    TRAIN_CMD="python examples/embodiment/${ENTRY_SCRIPT}"
    TRAIN_CMD+=" --config-name ${CONFIG_NAME}"
    TRAIN_CMD+=" cluster.num_nodes=${REPLICAS}"

    # For single replica the placement is baked into the config:
    #   yam_ppo_openpi:           actor:0  rollout:1
    #   yam_ppo_openpi_topreward: actor:0  rollout:1  VLM:2 (heuristic: max(0,1)+1)
    # For multiple replicas, distribute actor + rollout across node ranks.
    if [ "$REPLICAS" -gt 1 ]; then
        LAST_RANK=$((REPLICAS - 1))
        ALL_RANKS=$(seq -s, 0 "$LAST_RANK")
        TRAIN_CMD+=" 'cluster.component_placement.actor.placement=0-${LAST_RANK}'"
        TRAIN_CMD+=" 'cluster.component_placement.rollout.placement=0-${LAST_RANK}'"
        if [ "$IS_TOPREWARD" = true ]; then
            # WARNING: Multi-replica placement for VLM configs is NOT fully tested.
            TRAIN_CMD+=" 'cluster.node_groups=[{label: gpu, node_ranks: \"${ALL_RANKS}\"}, {label: beaker_vlm, node_ranks: 0}]'"
        else
            TRAIN_CMD+=" 'cluster.node_groups=[{label: gpu, node_ranks: \"${ALL_RANKS}\"}]'"
        fi
    fi

    if [ -n "$MODEL_PATH" ]; then
        TRAIN_CMD+=" actor.model.model_path=${MODEL_PATH}"
        TRAIN_CMD+=" rollout.model.model_path=${MODEL_PATH}"
    fi

    TRAIN_CMD+=" 'env.train.task_description=${TASK_DESC}'"
    TRAIN_CMD+=" 'env.eval.task_description=${TASK_DESC}'"

    for override in "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"; do
        TRAIN_CMD+=" ${override}"
    done

    # --- Build the entrypoint ---
    # 1. Install and start Tailscale (userspace networking for containers).
    # 2. Print Tailscale IP so user can set up reverse SSH tunnel.
    # 3. Uses start_ray_beaker.sh entrypoint mode for Ray head/worker setup.
    #
    # The train command is base64-encoded to avoid nested quoting issues
    # (it contains single-quoted Hydra overrides that break inside bash -c).
    TRAIN_CMD_B64=$(echo "$TRAIN_CMD" | base64 | tr -d '\n')
    INSTALL_CMD_B64=$(echo "$INSTALL_CMD" | base64 | tr -d '\n')

    ENTRYPOINT_CMD="curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg -o /usr/share/keyrings/tailscale-archive-keyring.gpg"
    ENTRYPOINT_CMD+=" && echo 'deb [signed-by=/usr/share/keyrings/tailscale-archive-keyring.gpg] https://pkgs.tailscale.com/stable/ubuntu jammy main' > /etc/apt/sources.list.d/tailscale.list"
    ENTRYPOINT_CMD+=" && (apt-get update || true) && apt-get install -y tailscale"
    ENTRYPOINT_CMD+=" && (nohup tailscaled --tun=userspace-networking --state=mem: > /dev/null 2>&1 &)"
    ENTRYPOINT_CMD+=" && sleep 2"
    ENTRYPOINT_CMD+=" && tailscale up --authkey=\${TAILSCALE_AUTHKEY} --hostname=beaker-\${BEAKER_REPLICA_RANK:-0} --accept-routes"
    ENTRYPOINT_CMD+=" && echo '=== Tailscale IP ===' && tailscale ip -4 && echo '=================='"
    ENTRYPOINT_CMD+=" && TAILSCALE_NODE_IP=\$(tailscale ip -4)"
    ENTRYPOINT_CMD+=" && mkdir -p /root/.ssh && chmod 700 /root/.ssh"
    ENTRYPOINT_CMD+=" && ssh-keygen -t ed25519 -f /root/.ssh/container_key -N '' -C beaker-container"
    ENTRYPOINT_CMD+=" && chmod 600 /root/.ssh/container_key"
    ENTRYPOINT_CMD+=" && echo '=== Container SSH Public Key (add to your desktop ~/.ssh/authorized_keys) ==='"
    ENTRYPOINT_CMD+=" && cat /root/.ssh/container_key.pub"
    ENTRYPOINT_CMD+=" && echo '=================='"
    ENTRYPOINT_CMD+=" && TRAIN_CMD_DECODED=\$(echo ${TRAIN_CMD_B64} | base64 -d)"
    ENTRYPOINT_CMD+=" && INSTALL_CMD_DECODED=\$(echo ${INSTALL_CMD_B64} | base64 -d)"
    ENTRYPOINT_CMD+=" && bash ray_utils/start_ray_beaker.sh"
    ENTRYPOINT_CMD+=" --entrypoint"
    ENTRYPOINT_CMD+=" --ray-port ${RAY_PORT}"
    ENTRYPOINT_CMD+=" --install \"\${INSTALL_CMD_DECODED}\""
    ENTRYPOINT_CMD+=" --train-cmd \"\${TRAIN_CMD_DECODED}\""

    # --- Build gantry command ---
    gantry_args=(
        gantry run --yes --no-python
        --replicas "${REPLICAS}"
        --gpus "${GPUS}"
        --host-networking
        --beaker-image "${BEAKER_IMAGE}"
        --workspace "${WORKSPACE}"
        --cluster "${CLUSTER}"
        --name "${EXP_NAME}"
        --priority "${PRIORITY}"
        --weka "${WEKA_MOUNT}"
        --env "HF_HOME=/weka/oe-training-default/shiruic/hf_cache"
        --env "ROBOT_SERVER_URL=localhost:50051"
        --env "EMBODIED_PATH=examples/embodiment"
        --env-secret "HF_TOKEN=hf_token_shirui"
        --env-secret "TAILSCALE_AUTHKEY=tailscale_authkey_shirui"
    )

    [ -n "$BUDGET" ]      && gantry_args+=("--budget" "$BUDGET")
    [ -n "$SHOW_LOGS" ]   && gantry_args+=("--show-logs")
    [ -n "$ALLOW_DIRTY" ] && gantry_args+=("--allow-dirty")

    gantry_args+=("--" "bash" "-c" "${ENTRYPOINT_CMD}")

    echo "=== Submit YAM Training to Beaker ==="
    echo "Config:       ${CONFIG_NAME}"
    echo "Entry point:  ${ENTRY_SCRIPT}"
    echo "Model:        ${MODEL_PATH:-<not set>}"
    echo "Task:         ${TASK_DESC}"
    echo "Replicas:     ${REPLICAS}"
    echo "GPUs/node:    ${GPUS}"
    echo "Cluster:      ${CLUSTER}"
    echo "Workspace:    ${WORKSPACE}"
    echo "TOPReward:    ${IS_TOPREWARD}"
    echo ""
    echo "Training command (head node):"
    echo "  ${TRAIN_CMD}"
    echo ""
    echo "After job starts:"
    echo "  1. Check logs for '=== Tailscale IP ===' to get the head node IP"
    echo "  2. Start robot server: bash scripts/start_robot_server.sh \\"
    echo "       --config examples/embodiment/config/env/yam.yaml \\"
    echo "       --remote-host <tailscale-ip> [--dummy]"
    echo ""

    if [ "$DRY_RUN" = "true" ]; then
        echo "[dry-run] Would execute:"
        printf '  %s\n' "${gantry_args[@]}"
    else
        "${gantry_args[@]}"
    fi

fi
