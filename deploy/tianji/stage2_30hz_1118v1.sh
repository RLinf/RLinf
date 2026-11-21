#!/usr/bin/env bash
# Generalized Car 1118v1 / sub6 deployment & DAgger collection entry.
# Parameterized via site.env (TIANJI_CAR_NUM_SUBTASKS, TIANJI_CAR_BANNER, TIANJI_MODEL_DIR, etc.).
set -euo pipefail
umask 077

launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rlinf_root="${RLINF_ROOT:-$(cd -- "$launcher/../.." && pwd)}"
export RLINF_ROOT="$rlinf_root"

mode="${1:---run}"
if [[ -n "${TIANJI_SITE_ENV:-}" && -f "${TIANJI_SITE_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "$TIANJI_SITE_ENV"
elif [[ -f "$launcher/site.env" ]]; then
  # shellcheck disable=SC1091
  source "$launcher/site.env"
elif [[ -f "$rlinf_root/site.env" ]]; then
  # shellcheck disable=SC1091
  source "$rlinf_root/site.env"
elif [[ -f "$HOME/.config/tianji/site.env" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.config/tianji/site.env"
fi

num_subtasks="${TIANJI_CAR_NUM_SUBTASKS:-15}"
if [[ -z "${TIANJI_CAR_BANNER:-}" ]]; then
  if [[ "$num_subtasks" == "6" ]]; then
    banner="Stage2-30HZ-sub6-1118v1"
  else
    banner="Stage2-30HZ-1118v1"
  fi
else
  banner="$TIANJI_CAR_BANNER"
fi
export TIANJI_CAR_NUM_SUBTASKS="$num_subtasks"
export TIANJI_CAR_BANNER="$banner"

cache="${TIANJI_CAR_CACHE:-${TIANJI_MODEL_DIR:-$rlinf_root/models/${banner}_step_30000}}"
if [[ "$mode" == --cache ]]; then
  exec python3 "$launcher/car_1118_assets.py" stage "$cache" "${@:2}"
fi

case "$mode" in
  --verify|--install|--run|--dagger) ;;
  *)
    echo "Usage: $0 [--cache [--source NAS_ROOT]|--verify|--install|--run|--dagger]" >&2
    exit 2
    ;;
esac

cpu="${TIANJI_CPU_PYTHON:-${RLINF_PYTHON:-}}"
gpu="${TIANJI_GPU_PYTHON:-${RLINF_PYTHON:-}}"
[[ -n "$cpu" && -n "$gpu" ]] && ([[ -x "$cpu" ]] || command -v "$cpu" >/dev/null 2>&1) && ([[ -x "$gpu" ]] || command -v "$gpu" >/dev/null 2>&1) || {
  echo "Configure RLINF_PYTHON (or TIANJI_CPU_PYTHON and TIANJI_GPU_PYTHON) in site.env." >&2
  exit 2
}

cache="$(realpath -e -- "$cache")"
identity="$(python3 "$launcher/car_1118_assets.py" verify "$cache")"
export TIANJI_CPU_PYTHON="$cpu" TIANJI_GPU_PYTHON="$gpu"
export TIANJI_CAR_1118=1 TIANJI_ACTION_SPACE=joint TIANJI_CONTROL_HZ=30
export TIANJI_ACTION_HORIZON=60 TIANJI_RT_SCHEDULE=1 TIANJI_ROW_SCALE=1 TIANJI_DENSE_ROWS=0
export TIANJI_DEPLOY_ACTOR_UPDATES=0 TIANJI_PI05_SERVICE=0 TIANJI_TACTILE_REQUIRED=0
export TIANJI_RLT_STAGE1_MODEL_PATH="$cache"
export TIANJI_RLT_STAGE1_NORM_STATS_PATH="$cache/norm_stats.json"
export TIANJI_SUBTASK_PROMPTS="$cache/prompts.json"
unset TIANJI_STAGE2_PREFETCH_AT_OVERRIDE TIANJI_STAGE_TIME_SCALE TIANJI_SHADOW_RUNTIME \
  TIANJI_STAGE2_SHADOW_VARIANT TIANJI_R5B_MONITOR_CONTROL TIANJI_SHADOW_EPISODE_REVIEW

# shellcheck disable=SC1091
source "$launcher/tianji_local.env"
runtime_dir="${TIANJI_RUNTIME_DIR:-$rlinf_root/.runtime/tianji}"
mkdir -p "$runtime_dir"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    "$TIANJI_CPU_PYTHON" "$launcher/select_stage1_gpu.py" "${TIANJI_STAGE1_GPU:-auto}")"
  export CUDA_VISIBLE_DEVICES
fi
# shellcheck disable=SC1091
source "$launcher/tianji_cuda_env.sh"

if [[ "$mode" == --verify ]]; then
  "$PYTHON_BIN" "$launcher/check_car_1118_hil.py"
  "$PYTHON_BIN" "$launcher/check_car_1118_model.py" > "$runtime_dir/car-model-check.tmp"
  printf '%s\n' "$identity" > "$runtime_dir/car-model-verified.tmp"
  mv "$runtime_dir/car-model-check.tmp" "$runtime_dir/car-model-check.json"
  mv "$runtime_dir/car-model-verified.tmp" "$runtime_dir/car-model-verified.json"
  cat "$runtime_dir/car-model-check.json"
  exit 0
fi

[[ -f "$runtime_dir/car-model-verified.json" ]] &&
  [[ "$(cat "$runtime_dir/car-model-verified.json")" == "$identity" ]] || {
    echo "Run --verify successfully with these model assets first." >&2
    exit 2
  }

if [[ "$mode" == --install ]]; then
  infra_python="${LOOPINFRA_PYTHON:-$(cat "$LOOPINFRA_ROOT/.loopinfra-python")}"
  env -u PYTHONPATH "$infra_python" -B "$launcher/inspect_loopinfra_local.py" "$LOOPINFRA_ROOT" --speed trial45 |
    "$TIANJI_CPU_PYTHON" -c '
import json
import os
import sys
inspection = json.load(sys.stdin)
for key, name in (("pose_config_id", "TIANJI_CAR_POSE_CONFIG_ID"),
                  ("reset_pose_id", "TIANJI_CAR_RESET_POSE_ID")):
    if inspection[key] != os.environ.get(name):
        sys.exit(f"Installed collector {key} does not match site.env")
'
  "$TIANJI_CPU_PYTHON" "$launcher/install_car_1118_desktop.py" "$rlinf_root"
  exit 0
fi

if ss -Hltn "( sport = :$TIANJI_HIL_PORT )" | grep -q .; then
  echo "HIL port $TIANJI_HIL_PORT is already in use. End the running Stage2 session before switching." >&2
  exit 2
fi

collector_args=()
if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  collector_args+=(--no-browser)
fi
bash "$launcher/run_tianji_wuji_collector.sh" --disable-xense "${collector_args[@]}"
"$TIANJI_CPU_PYTHON" "$launcher/check_car_1118_collector.py"

mkdir -p "$TIANJI_RUNS_DIR"
hil_mode="inference"
exp_name="$banner"
if [[ "$mode" == --dagger ]]; then
  hil_mode="dagger"
  exp_name="${banner}-dagger"
fi
TIANJI_RLT_OUTPUT="$(mktemp -d "$TIANJI_RUNS_DIR/${exp_name}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
export TIANJI_RLT_OUTPUT

python3 - "$TIANJI_RLT_OUTPUT" "$hil_mode" <<'PY'
import json
import os
from pathlib import Path
import sys
keys = (
    "TIANJI_RLT_STAGE1_MODEL_PATH",
    "TIANJI_RLT_STAGE1_NORM_STATS_PATH",
    "TIANJI_SUBTASK_PROMPTS",
    "TIANJI_CAR_NUM_SUBTASKS",
    "TIANJI_CAR_BANNER",
    "TIANJI_CONTROL_HZ",
    "TIANJI_ACTION_HORIZON",
    "TIANJI_STAGE2_CHUNK",
    "TIANJI_STAGE2_PREFETCH_AT",
    "RLT_REF_OFFSET",
    "CUDA_VISIBLE_DEVICES",
    "LOOPINFRA_URL",
    "TIANJI_HIL_PORT",
)
settings = {key: os.getenv(key) for key in keys}
settings["hil_mode"] = sys.argv[2]
settings["model"] = json.loads((Path(settings[keys[0]]) / "identity.json").read_text())
(Path(sys.argv[1]) / "deployment.json").write_text(json.dumps(settings, indent=2) + "\n")
PY

RAY_TMPDIR="$(mktemp -d /tmp/tjcar.XXXXXX)"
export RAY_TMPDIR
echo "${exp_name} HIL: http://127.0.0.1:$TIANJI_HIL_PORT/"
echo "Run archive: $TIANJI_RLT_OUTPUT"
exec "$PYTHON_BIN" -u "$rlinf_root/evaluations/eval_embodied_agent.py" \
  --config-path "$EMBODIED_PATH/config" --config-name tianji2_joint_stage1 \
  hydra.run.dir="$TIANJI_RLT_OUTPUT" runner.logger.experiment_name="$exp_name" \
  runner.deploy_actor_updates=false rollout.early_policy_recv=false \
  ++env.eval.override_cfg.episode_archive=true \
  ++env.eval.override_cfg.episode_archive_image_hz=15 \
  ++env.eval.override_cfg.tactile_enabled=false actor.model.tactile_enabled=false \
  env.eval.override_cfg.hil_mode="$hil_mode" \
  env.eval.override_cfg.recorder_dir="$TIANJI_RLT_OUTPUT/hil"
