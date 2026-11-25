#!/usr/bin/env bash
# Plug-and-play Sim2Real toy-car inference / check / DAgger launcher.
# Supports both pure-SFT (model.safetensors, use_rlt=False) and RLT Stage1
# checkpoints in 16D joint space at 30 Hz (tianji_stage1_joint_h60), with a
# single global prompt or 1/4/12/24 subtask prompts (prompts.json).
set -euo pipefail
umask 077

DEFAULT_CAR_PROMPT="Install four wheels and screws on a wheel-less toy vehicle using the toy drill."

launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rlinf_root="${RLINF_ROOT:-$(cd -- "$launcher/../.." && pwd)}"
export RLINF_ROOT="$rlinf_root"

usage() {
  cat <<'EOF'
Usage: run_tianji_sim2real_car.sh --model <dir> (--check | --read-only | --run | --dagger) [options]

Required mode (pick one):
  --check                   Offline GPU smoke test (no LoopInfra or robot motion)
  --read-only               Live LoopInfra cameras/joints + GPU inference, read_only=true (NO motion)
  --run                     Real 30 Hz 16D joint execution on Tianji via :8879 HIL panel
  --dagger                  Real 30 Hz execution + DAgger episode archive (hil_mode=dagger)

Options:
  --model <dir>             Checkpoint directory (default: $TIANJI_MODEL_DIR or $TIANJI_RLT_STAGE1_MODEL_PATH)
  --norm-stats <path>       Path to norm_stats.json (default: <model>/norm_stats.json)
  --prompts <prompts.json>  Optional prompts.json (1..32 prompts; default: <model>/prompts.json if present)
  --prompt "<text>"         Single global prompt override (default when no prompts.json exists:
                            "Install four wheels and screws on a wheel-less toy vehicle using the toy drill.")
  --config-name <name>      OpenPI config: tianji_stage1_joint_h60 (default) or tianji_stage1_joint
  --use-rlt <auto|true|false>
                            Checkpoint RLT mode (default: auto-detected from checkpoint keys)
  --gpu <id>                Pin CUDA GPU index (e.g. 0)
  --fast <0|1|2>            Fast inference mode (0=eager, 1=CUDA graph, 2=fused+graph)
  --iters <N>               Benchmark iterations for --check (default: 20)
  --blend-steps <N>         Control-side chunk Smoothstep blend steps (default: 10)
  --max-vel-rad-s <rad/s>   Per-step arm joint velocity clamp in rad/s (default: 0.7854; 0 disables)
  --rtc-guidance            Enable RLinf RTC guidance instead of control-side Smoothstep blend
  --no-rtc-guidance         Disable RLinf RTC guidance (default for unguided Smoothstep blend)
  --max-chunks <N>          Stop rollout after N chunks
  -h, --help                Show this help and exit
  -- [hydra_overrides...]   Extra Hydra overrides passed to eval_embodied_agent.py
EOF
}

mode=""
model_arg=""
norm_stats_arg=""
prompts_arg=""
prompt_arg=""
prompts_explicit=0
prompt_explicit=0
config_name="${TIANJI_OPENPI_CONFIG:-tianji_stage1_joint_h60}"
use_rlt="${TIANJI_USE_RLT:-auto}"
gpu_arg=""
rtc_guidance=0
fast_mode="${TIANJI_FAST_INFER:-}"
check_iters="20"
blend_steps="${TIANJI_CHUNK_BLEND_STEPS:-10}"
max_vel_rad_s="${TIANJI_MAX_STEP_VEL_RAD_S:-0.7854}"
extra_overrides=()

while (( $# )); do
  case "$1" in
    --check|--read-only|--run|--dagger)
      if [[ -n "$mode" && "$mode" != "$1" ]]; then
        echo "Specify only one mode (--check, --read-only, --run, --dagger), got $mode and $1" >&2
        exit 2
      fi
      mode="$1"
      shift
      ;;
    --model) model_arg="${2:?--model requires a directory}"; shift 2 ;;
    --model=*) model_arg="${1#*=}"; shift ;;
    --norm-stats) norm_stats_arg="${2:?--norm-stats requires a path}"; shift 2 ;;
    --norm-stats=*) norm_stats_arg="${1#*=}"; shift ;;
    --prompts) prompts_arg="${2:?--prompts requires a file}"; prompts_explicit=1; shift 2 ;;
    --prompts=*) prompts_arg="${1#*=}"; prompts_explicit=1; shift ;;
    --prompt) prompt_arg="${2:?--prompt requires text}"; prompt_explicit=1; shift 2 ;;
    --prompt=*) prompt_arg="${1#*=}"; prompt_explicit=1; shift ;;
    --config-name) config_name="${2:?--config-name requires a value}"; shift 2 ;;
    --config-name=*) config_name="${1#*=}"; shift ;;
    --use-rlt) use_rlt="${2:?--use-rlt requires auto|true|false}"; shift 2 ;;
    --use-rlt=*) use_rlt="${1#*=}"; shift ;;
    --gpu) gpu_arg="${2:?--gpu requires a GPU id}"; shift 2 ;;
    --gpu=*) gpu_arg="${1#*=}"; shift ;;
    --fast) fast_mode="${2:?--fast requires 0|1|2}"; shift 2 ;;
    --fast=*) fast_mode="${1#*=}"; shift ;;
    --iters)
      if [[ ! "${2:-}" =~ ^[1-9][0-9]*$ ]]; then
        echo "--iters requires a positive integer" >&2
        exit 2
      fi
      check_iters="$2"
      shift 2
      ;;
    --blend-steps)
      if [[ ! "${2:-}" =~ ^[0-9]+$ ]]; then
        echo "--blend-steps requires a non-negative integer" >&2
        exit 2
      fi
      blend_steps="$2"
      shift 2
      ;;
    --max-vel-rad-s) max_vel_rad_s="${2:?--max-vel-rad-s requires a number}"; shift 2 ;;
    --max-vel-rad-s=*) max_vel_rad_s="${1#*=}"; shift ;;
    --rtc-guidance) rtc_guidance=1; shift ;;
    --no-rtc-guidance) rtc_guidance=0; shift ;;
    --max-chunks)
      if [[ ! "${2:-}" =~ ^[0-9]+$ ]]; then
        echo "--max-chunks requires a non-negative integer" >&2
        exit 2
      fi
      max_chunks="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_overrides+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1 (pass Hydra overrides after --)" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$mode" ]]; then
  echo "A mode flag is required (--check, --read-only, --run, or --dagger)." >&2
  usage >&2
  exit 2
fi

if (( prompts_explicit && prompt_explicit )); then
  echo "Pass either --prompts <prompts.json> or --prompt \"<text>\", not both." >&2
  exit 2
fi

# Source site.env first so machine paths load before Sim2Real overrides.
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

model_dir="${model_arg:-${TIANJI_RLT_STAGE1_MODEL_PATH:-${TIANJI_MODEL_DIR:-}}}"
if [[ -z "$model_dir" ]]; then
  echo "Model checkpoint directory is required (--model <dir> or TIANJI_MODEL_DIR)." >&2
  exit 2
fi
model_dir="$(realpath -e -- "$model_dir")"

norm_stats="${norm_stats_arg:-${model_dir}/norm_stats.json}"
norm_stats="$(realpath -e -- "$norm_stats")"

if (( prompts_explicit )); then
  prompts_file="$(realpath -e -- "$prompts_arg")"
  task_prompt="$DEFAULT_CAR_PROMPT"
elif (( prompt_explicit )); then
  prompts_file=""
  task_prompt="$prompt_arg"
elif [[ -f "$model_dir/prompts.json" ]]; then
  prompts_file="$(realpath -e -- "$model_dir/prompts.json")"
  task_prompt="$DEFAULT_CAR_PROMPT"
else
  prompts_file=""
  task_prompt="$DEFAULT_CAR_PROMPT"
fi

if [[ ! -f "$model_dir/model.safetensors" && \
      ! -f "$model_dir/full_weights.pt" && \
      ! -f "$model_dir/model_state_dict/full_weights.pt" && \
      ! -f "$model_dir/actor/model_state_dict/full_weights.pt" ]]; then
  echo "No model.safetensors or full_weights.pt found under $model_dir" >&2
  exit 2
fi

unset TIANJI_CAR_1118 TIANJI_CAR_NUM_SUBTASKS TIANJI_CAR_BANNER \
  TIANJI_STAGE2_PREFETCH_AT_OVERRIDE TIANJI_STAGE_TIME_SCALE \
  TIANJI_SHADOW_RUNTIME TIANJI_STAGE2_SHADOW_VARIANT \
  TIANJI_R5B_MONITOR_CONTROL TIANJI_SHADOW_EPISODE_REVIEW

export TIANJI_ACTION_SPACE=joint
export TIANJI_CONTROL_HZ=30
export TIANJI_ACTION_HORIZON=60
export TIANJI_OPENPI_CONFIG="$config_name"
export TIANJI_STAGE1_VLA_ONLY=1
export TIANJI_MAX_SUBTASKS=32
export TIANJI_PI05_SERVICE=0
export TIANJI_TACTILE_REQUIRED=0
export TIANJI_ROW_SCALE=1
export TIANJI_DENSE_ROWS=0
export TIANJI_DEPLOY_ACTOR_UPDATES=0
export TIANJI_CHUNK_BLEND_STEPS="$blend_steps"
export TIANJI_MAX_STEP_VEL_RAD_S="$max_vel_rad_s"
export TIANJI_CHUNK_DIAGNOSTICS="${TIANJI_CHUNK_DIAGNOSTICS:-1}"
if [[ -n "$fast_mode" ]]; then
  export TIANJI_FAST_INFER="$fast_mode"
fi
export TIANJI_RLT_STAGE1_MODEL_PATH="$model_dir"
export TIANJI_RLT_STAGE1_NORM_STATS_PATH="$norm_stats"
export TIANJI_SUBTASK_PROMPTS="$prompts_file"
export TIANJI_TASK_PROMPT="$task_prompt"

if (( ! rtc_guidance )); then
  extra_overrides=(++env.eval.override_cfg.rtc_guidance=false "${extra_overrides[@]}")
fi

case "$use_rlt" in
  auto) unset TIANJI_USE_RLT ;;
  true|1) export TIANJI_USE_RLT=true ;;
  false|0) export TIANJI_USE_RLT=false ;;
  *)
    echo "Unsupported --use-rlt value '$use_rlt'; expected auto, true, or false." >&2
    exit 2
    ;;
esac

# Bootstrap timetable & Python paths without letting site.env clobber CLI options.
TIANJI_SKIP_SITE_ENV=1
export TIANJI_SKIP_SITE_ENV
# shellcheck disable=SC1091
source "$launcher/tianji_local.env"
unset TIANJI_SKIP_SITE_ENV

if [[ -n "$gpu_arg" ]]; then
  export CUDA_VISIBLE_DEVICES="$gpu_arg"
  export TIANJI_STAGE1_GPU="$gpu_arg"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
      "$TIANJI_CPU_PYTHON" "$launcher/select_stage1_gpu.py" "${TIANJI_STAGE1_GPU:-auto}")"
    export CUDA_VISIBLE_DEVICES
  fi
fi

# shellcheck disable=SC1091
source "$launcher/tianji_cuda_env.sh"

if [[ -n "${max_chunks:-}" ]] && (( max_chunks > 0 )); then
  extra_overrides+=("env.eval.max_steps_per_rollout_epoch=$(( max_chunks * TIANJI_STAGE2_CHUNK ))")
fi

if [[ "$mode" == --check ]]; then
  check_args=(
    --model "$model_dir"
    --norm-stats "$norm_stats"
    --config-name "$config_name"
    --use-rlt "$use_rlt"
    --iters "$check_iters"
    --blend-steps "$blend_steps"
    --v-max "$max_vel_rad_s"
  )
  if [[ -n "$fast_mode" ]]; then
    check_args+=(--fast "$fast_mode")
  fi
  if [[ -n "$prompts_file" ]]; then
    check_args+=(--prompts "$prompts_file")
  else
    check_args+=(--prompt "$task_prompt")
  fi
  exec "$PYTHON_BIN" -u "$launcher/check_sim2real_car_model.py" "${check_args[@]}"
fi

if ss -Hltn "( sport = :$TIANJI_HIL_PORT )" 2>/dev/null | grep -q .; then
  echo "HIL port $TIANJI_HIL_PORT is already in use. Stop the running session first." >&2
  exit 2
fi

mkdir -p "$TIANJI_RUNS_DIR"
exp_name="sim2real_car_${mode#--}"
TIANJI_RLT_OUTPUT="${TIANJI_STAGE1_OUTPUT:-$(mktemp -d "$TIANJI_RUNS_DIR/${exp_name}_$(date +%Y%m%d_%H%M%S)_XXXXXX")}"
export TIANJI_RLT_OUTPUT
mkdir -p "$TIANJI_RLT_OUTPUT"
RAY_TMPDIR="$(mktemp -d "${TIANJI_RUNTIME_DIR}/ray_sim2real.XXXXXX")"
export RAY_TMPDIR

if [[ "$mode" == --read-only ]]; then
  "$TIANJI_CPU_PYTHON" - <<'PY'
import json
import os
import urllib.request

url = os.environ.get("LOOPINFRA_URL", "http://127.0.0.1:8780").rstrip("/") + "/api/status"
token = os.environ.get("LOOPINFRA_TOKEN")
req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token} if token else {})
try:
    with urllib.request.urlopen(req, timeout=3) as resp:
        data = json.load(resp)
except Exception as exc:
    raise SystemExit(f"LoopInfra not reachable at {url}: {exc}")
print(f"LoopInfra connected={data.get('connected')} stale={data.get('stale')} mode={(data.get('robot_control') or {}).get('mode')}")
PY
  echo "Sim2Real Car READ-ONLY HIL: http://127.0.0.1:$TIANJI_HIL_PORT/ (no robot motion)"
  echo "Run archive: $TIANJI_RLT_OUTPUT"
  exec "$PYTHON_BIN" -u "$rlinf_root/evaluations/eval_embodied_agent.py" \
    --config-path "$EMBODIED_PATH/config" --config-name tianji2_joint_stage1 \
    hydra.run.dir="$TIANJI_RLT_OUTPUT" runner.logger.experiment_name="$exp_name" \
    runner.deploy_actor_updates=false rollout.early_policy_recv=false \
    env.eval.override_cfg.read_only=true \
    env.eval.override_cfg.reset_to_saved_pose=false \
    env.eval.override_cfg.reset_after_episode=false \
    env.eval.override_cfg.hil_mode=inference \
    ++env.eval.override_cfg.tactile_enabled=false actor.model.tactile_enabled=false \
    "${extra_overrides[@]}"
fi

# --run and --dagger: validate live collector before moving hardware.
bash "$launcher/check_tianji_local.sh" --mode stage1 --live

hil_mode="inference"
recorder_sub="transitions"
dagger_overrides=()
if [[ "$mode" == --dagger ]]; then
  hil_mode="dagger"
  recorder_sub="hil"
  dagger_overrides+=(
    ++env.eval.override_cfg.episode_archive=true
    ++env.eval.override_cfg.episode_archive_image_hz=15
  )
fi

echo "Sim2Real Car ($mode, hil_mode=$hil_mode) HIL: http://127.0.0.1:$TIANJI_HIL_PORT/"
echo "Model: $model_dir ($config_name)"
echo "Prompt: ${prompts_file:-$task_prompt}"
echo "Run archive: $TIANJI_RLT_OUTPUT"

exec "$PYTHON_BIN" -u "$rlinf_root/evaluations/eval_embodied_agent.py" \
  --config-path "$EMBODIED_PATH/config" --config-name tianji2_joint_stage1 \
  hydra.run.dir="$TIANJI_RLT_OUTPUT" runner.logger.experiment_name="$exp_name" \
  runner.deploy_actor_updates=false rollout.early_policy_recv=false \
  env.eval.override_cfg.hil_mode="$hil_mode" \
  env.eval.override_cfg.recorder_dir="$TIANJI_RLT_OUTPUT/$recorder_sub" \
  ++env.eval.override_cfg.tactile_enabled=false actor.model.tactile_enabled=false \
  "${dagger_overrides[@]}" \
  "${extra_overrides[@]}"
