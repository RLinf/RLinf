#!/usr/bin/env bash
# GenieSim + RLinf Docker launcher.
#
# Environment variables (set before running, or edit defaults below):
#   RLINF_REPO    — absolute path to the RLinf repository on the host
#   GENIESIM_REPO — absolute path to the GenieSim (genie_sim) repository on the host
#
# Container mount layout:
#   RLINF_REPO    -> /geniesim/RLinf
#   GENIESIM_REPO -> /geniesim/main

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RLINF_REPO="${RLINF_REPO:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
GENIESIM_REPO="${GENIESIM_REPO:-$(cd "${RLINF_REPO}/../genie_sim" && pwd)}"

IMAGE="${IMAGE:-geniesim-rlinf-train:latest}"

show_help() {
    cat <<EOF
Usage: bash $(basename "$0") <command> [args...]

Environment variables:
  RLINF_REPO      Path to RLinf repo    (default: ${RLINF_REPO})
  GENIESIM_REPO   Path to GenieSim repo (default: ${GENIESIM_REPO})
  IMAGE           Docker image           (default: ${IMAGE})

Commands:
  collect [--num-demos N] [--step-hz HZ] [extra args...]
      Collect SpaceMouse demos (default: 50 demos, 10 Hz)

  train [extra hydra overrides...]
      Start SAC training

  convert [extra args...]
      Convert demo pkl files to replay buffer

  shell
      Drop into an interactive bash shell inside the container

  <any command>
      Run arbitrary command inside the container

Examples:
  bash $(basename "$0") collect --num-demos 50
  bash $(basename "$0") train
  bash $(basename "$0") train algorithm.gamma=0.97
  bash $(basename "$0") convert --recompute-reward
  bash $(basename "$0") shell
EOF
}

docker_run() {
    local name="${1}"
    shift
    bash "${SCRIPT_DIR}/cleanup_stale.sh" "${GENIESIM_REPO}" 2>/dev/null || true
    docker run -it --rm --name "$name" \
        --network host --ipc host --gpus all --privileged \
        -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data \
        -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg \
        -v "${GENIESIM_REPO}":/geniesim/main \
        -v "${RLINF_REPO}":/geniesim/RLinf \
        -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache \
        -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov \
        -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip \
        -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache \
        -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache \
        -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs \
        -v ~/docker/isaac-sim/config:/root/.nvidia-omniverse/config \
        -v ~/docker/isaac-sim/data/documents:/root/Documents \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v "${XAUTHORITY:-$HOME/.Xauthority}":/root/.Xauthority:ro \
        -e DISPLAY="$DISPLAY" \
        -e XAUTHORITY=/root/.Xauthority \
        -e GENIESIM_CONTAINER=1 \
        "$IMAGE" \
        /geniesim/scripts/run_rlinf.sh \
        "$@"
}

if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

CMD="$1"
shift

case "$CMD" in
    collect)
        NUM_DEMOS=50
        STEP_HZ=10
        EXTRA_ARGS=()
        while [ $# -gt 0 ]; do
            case "$1" in
                --num-demos) NUM_DEMOS="$2"; shift 2 ;;
                --step-hz)   STEP_HZ="$2";  shift 2 ;;
                *)           EXTRA_ARGS+=("$1"); shift ;;
            esac
        done
        docker_run geniesim_collect \
            python3 examples/embodiment/collect_sim_data.py \
                --config examples/embodiment/config/env/geniesim_place_workpiece.yaml \
                --save-dir /geniesim/main/sac_demo \
                --num-demos "$NUM_DEMOS" \
                --step-hz "$STEP_HZ" \
                "${EXTRA_ARGS[@]}"
        ;;

    train)
        docker_run geniesim_train \
            python3 examples/embodiment/train_embodied_agent.py \
                --config-name geniesim_sac_spacemouse \
                "$@"
        ;;

    convert)
        docker_run geniesim_convert \
            python3 examples/embodiment/convert_demos_to_buffer.py \
                --demo-dir /geniesim/main/sac_demo \
                --output-dir /geniesim/main/sac_demo_buffer \
                "$@"
        ;;

    shell)
        docker_run geniesim_shell bash "$@"
        ;;

    help|--help|-h)
        show_help
        ;;

    *)
        docker_run geniesim_run "$CMD" "$@"
        ;;
esac
