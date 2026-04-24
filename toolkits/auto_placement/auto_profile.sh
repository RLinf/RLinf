#! /bin/bash
# auto profile (if needed) + auto placement for embodied training.
# Requires Ray to be running (e.g. ray start --head).

set -euo pipefail

CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../examples/embodiment" && pwd)"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export REPO_PATH
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export RUN_SCRIPT="${EMBODIED_PATH}/run_embodiment.sh"
export POSTPROCESS_SCRIPT="${REPO_PATH}/toolkits/auto_placement/profile_postprocess.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-}"
export OMNIGIBSON_DATASET_PATH="${OMNIGIBSON_DATASET_PATH:-${OMNIGIBSON_DATA_PATH}/behavior-1k-assets/}"
export OMNIGIBSON_KEY_PATH="${OMNIGIBSON_KEY_PATH:-${OMNIGIBSON_DATA_PATH}/omnigibson.key}"
export OMNIGIBSON_ASSET_PATH="${OMNIGIBSON_ASSET_PATH:-${OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets/}"
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

CONFIG_NAME="${1:-maniskill_ppo_openvlaoft_quickstart}"

ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"

LOG_DIR="${REPO_PATH}/logs/profile_logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
export LOG_DIR

# Compose config via Hydra (do NOT manually parse YAML) so `defaults:` is expanded.
BASE_ENV_NUM="$(
EMBODIED_PATH="${EMBODIED_PATH}" CONFIG_NAME="${CONFIG_NAME}" python - <<'PY'
import os

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

embodied_path = os.environ["EMBODIED_PATH"]
config_name = os.environ["CONFIG_NAME"]

with initialize_config_dir(
    version_base="1.1", config_dir=os.path.join(embodied_path, "config")
):
    cfg = compose(config_name=config_name)

env_num = OmegaConf.select(cfg, "env.train.total_num_envs")
if env_num is None:
    env_num = OmegaConf.select(cfg, "data.env_num")
if env_num is None:
    raise SystemExit(
        "Missing env.train.total_num_envs (and data.env_num fallback) in composed config."
    )
print(int(env_num))
PY
)"

# For profiling we only need a few env steps; keep this fixed and avoid depending
# on optional config fields (some configs omit the `data` section).
STEP_PER_ENV=3

# Always override group names to unique ones per run (avoid Ray actor name collisions).
ACTOR_GROUP_NAME="ActorGroup"
ROLLOUT_GROUP_NAME="RolloutGroup"
ENV_GROUP_NAME="EnvGroup"

ENV_NUM_TEST_LIST=(
  "${BASE_ENV_NUM}"
  "$((BASE_ENV_NUM * 2))"
#   "$((BASE_ENV_NUM * 3))"
#   "$((BASE_ENV_NUM * 4))"
)

echo "Profiling env nums: ${ENV_NUM_TEST_LIST[*]}" | tee -a "${MEGA_LOG_FILE}"

for ENV_NUM in "${ENV_NUM_TEST_LIST[@]}"; do
  echo "=== Profiling total_num_envs=${ENV_NUM} ===" | tee -a "${MEGA_LOG_FILE}"

  # Run embodied training via the standard entry script.
  LOG_DIR="${LOG_DIR}" bash "${RUN_SCRIPT}" "${CONFIG_NAME}" "${ROBOT_PLATFORM}" \
    "runner.logger.log_path=${LOG_DIR}" \
    "cluster.component_placement.actor=all" \
    "cluster.component_placement.rollout=all" \
    "cluster.component_placement.env=all" \
    "env.train.total_num_envs=${ENV_NUM}" \
    "env.train.group_size=1" \
    "runner.max_steps=${STEP_PER_ENV}" \
    "runner.save_interval=0" \
    "+runner.per_worker_log=true" \
    "rollout.pipeline_stage_num=1" \
    "actor.group_name=${ACTOR_GROUP_NAME}_env${ENV_NUM}" \
    "rollout.group_name=${ROLLOUT_GROUP_NAME}_env${ENV_NUM}" \
    "env.group_name=${ENV_GROUP_NAME}_env${ENV_NUM}" \
    2>&1 | tee -a "${MEGA_LOG_FILE}"

  # Write TensorBoard markers so extraction can segment runs.
  EMBODIED_PATH="${EMBODIED_PATH}" CONFIG_NAME="${CONFIG_NAME}" LOG_DIR="${LOG_DIR}" ENV_NUM="${ENV_NUM}" \
  python - <<'PY' 2>&1 | tee -a "${MEGA_LOG_FILE}"
import os

from hydra import compose, initialize_config_dir

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement

env_num = int(os.environ["ENV_NUM"])
log_dir = os.environ["LOG_DIR"]
embodied_path = os.environ["EMBODIED_PATH"]
config_name = os.environ["CONFIG_NAME"]

with initialize_config_dir(
    version_base="1.1", config_dir=os.path.join(embodied_path, "config")
):
    cfg = compose(
        config_name=config_name,
        overrides=[
            f"runner.logger.log_path={log_dir}",
            "cluster.component_placement.actor=all",
            "cluster.component_placement.rollout=all",
            "cluster.component_placement.env=all",
            f"env.train.total_num_envs={env_num}",
            "env.train.group_size=1",
            "+runner.per_worker_log=true",
            "rollout.pipeline_stage_num=1",
        ],
    )

cfg = validate_cfg(cfg)
placement = HybridComponentPlacement(cfg, Cluster())
env_world_size = int(placement.get_world_size("env"))
if env_world_size <= 0:
    raise SystemExit(f"Invalid env_world_size={env_world_size}")
if env_num % env_world_size != 0:
    raise SystemExit(
        f"total_num_envs ({env_num}) must be divisible by env_world_size ({env_world_size})"
    )

env_num_per_instance = env_num // env_world_size

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as exc:
    raise SystemExit(
        "tensorboard is required for marker logging. Try `pip install tensorboard`."
    ) from exc

tb_logdir = os.path.join(log_dir, "tensorboard", "all")
os.makedirs(tb_logdir, exist_ok=True)
writer = SummaryWriter(tb_logdir)
writer.add_scalar("profile/env_num_per_instance", int(env_num_per_instance), 0)
writer.add_scalar("profile/total_num_envs", int(env_num), 0)
writer.flush()
writer.close()
PY
done

# Allow TensorBoard events to flush to disk before extraction
sleep 2

EMBODIED_PATH="${EMBODIED_PATH}" CONFIG_NAME="${CONFIG_NAME}" LOG_DIR="${LOG_DIR}" \
python - <<'PY' 2>&1 | tee -a "${MEGA_LOG_FILE}"
import os
from pathlib import Path

import yaml


def _find_run_dirs(logdir: Path) -> list[Path]:
    event_prefix = "events.out.tfevents"
    run_dirs: list[Path] = []
    for root, _dirs, files in os.walk(logdir):
        if any(f.startswith(event_prefix) for f in files):
            run_dirs.append(Path(root))
    return sorted(set(run_dirs))


def _load_scalars(run_dir: Path) -> dict[str, list[tuple[float, float]]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        raise SystemExit(
            "tensorboard is required for extraction. Try `pip install tensorboard`."
        ) from exc

    ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    results: dict[str, list[tuple[float, float]]] = {}
    for tag in (
        "time/env/env_interact_step",
        "time/rollout/predict",
        "time/actor/run_training",
        "env/memory/util",
        "rollout/memory/util",
        "profile/env_num_per_instance",
        "profile/total_num_envs",
    ):
        if tag not in available:
            continue
        results[tag] = [(e.wall_time, e.value) for e in ea.Scalars(tag)]
    return results


def extract_from_tensorboard(log_dir: str) -> dict:
    logdir = Path(log_dir)
    if not logdir.exists():
        raise SystemExit(f"logdir not found: {logdir}")

    run_dirs = _find_run_dirs(logdir)
    if not run_dirs:
        print(f"# No TensorBoard event files found under: {logdir}")
        return {}

    env_time_profile_data: dict[int, float] = {}
    rollout_time_profile_data: dict[int, float] = {}
    actor_time_profile_data: dict[int, float] = {}
    env_memory_util_profile_data: dict[int, float] = {}
    rollout_memory_util_profile_data: dict[int, float] = {}

    def _vals_in_window(events, t_start, t_end, inclusive_end: bool = False):
        if inclusive_end:
            return [v for t, v in events if t_start < t <= t_end]
        return [v for t, v in events if t_start <= t < t_end]

    for run_dir in run_dirs:
        data = _load_scalars(run_dir)
        if not data or "profile/env_num_per_instance" not in data:
            continue

        env_events = sorted(data["profile/env_num_per_instance"], key=lambda x: x[0])
        total_env_events = sorted(
            data.get("profile/total_num_envs", []), key=lambda x: x[0]
        )
        env_cost_events = data.get("time/env/env_interact_step", [])
        rollout_cost_events = data.get("time/rollout/predict", [])
        actor_cost_events = data.get("time/actor/run_training", [])
        env_mem_util_events = data.get("env/memory/util", [])
        rollout_mem_util_events = data.get("rollout/memory/util", [])

        for i, (t_end, env_num_val) in enumerate(env_events):
            t_start = env_events[i - 1][0] if i > 0 else 0.0
            env_num_per_instance = int(env_num_val)

            # Pair by index, not by wall_time window: each run writes
            # profile/env_num_per_instance then profile/total_num_envs, so the
            # total marker always has a strictly later timestamp than the env
            # marker. Using (t_{i-1}, t_i] then misses that run's total (drops
            # the first actor segment) and attributes totals to the next
            # segment (off-by-one keys).
            if i < len(total_env_events):
                total_num_envs = int(total_env_events[i][1])
            else:
                total_env_vals = _vals_in_window(
                    total_env_events, t_start, t_end, inclusive_end=True
                )
                total_num_envs = (
                    int(total_env_vals[-1]) if total_env_vals else None
                )

            env_vals = _vals_in_window(env_cost_events, t_start, t_end, inclusive_end=True)
            rollout_vals = _vals_in_window(
                rollout_cost_events, t_start, t_end, inclusive_end=True
            )
            actor_vals = _vals_in_window(
                actor_cost_events, t_start, t_end, inclusive_end=True
            )
            env_mem_util_vals = _vals_in_window(
                env_mem_util_events, t_start, t_end, inclusive_end=True
            )
            rollout_mem_util_vals = _vals_in_window(
                rollout_mem_util_events, t_start, t_end, inclusive_end=True
            )

            if env_vals:
                env_time_profile_data[env_num_per_instance] = sum(env_vals) / len(env_vals)
            if rollout_vals:
                rollout_time_profile_data[env_num_per_instance] = sum(rollout_vals) / len(rollout_vals)
            if actor_vals and total_num_envs is not None:
                actor_time_profile_data[total_num_envs] = sum(actor_vals) / len(actor_vals)

            if env_mem_util_vals:
                env_memory_util_profile_data[env_num_per_instance] = sum(env_mem_util_vals) / len(env_mem_util_vals)
            if rollout_mem_util_vals:
                rollout_memory_util_profile_data[env_num_per_instance] = sum(rollout_mem_util_vals) / len(rollout_mem_util_vals)

    profile_time = {}
    if env_time_profile_data:
        profile_time["env_profile_time"] = env_time_profile_data
    if rollout_time_profile_data:
        profile_time["rollout_profile_time"] = rollout_time_profile_data
    if actor_time_profile_data:
        profile_time["actor_profile_time"] = actor_time_profile_data

    profile_memory = {}
    if env_memory_util_profile_data:
        profile_memory["env_profile_memory"] = env_memory_util_profile_data
    if rollout_memory_util_profile_data:
        profile_memory["rollout_profile_memory"] = rollout_memory_util_profile_data

    profile_data = {}
    if profile_time:
        profile_data["profile_time"] = profile_time
    if profile_memory:
        profile_data["profile_memory"] = profile_memory

    print(profile_data)
    return profile_data


def update_yaml_with_profile_data(config_path: str, config_name: str, profile_data: dict) -> None:
    original_config_path = f"{config_path}/{config_name}.yaml"
    new_config_path = original_config_path.replace(".yaml", "_profiled.yaml")

    with open(original_config_path, "r") as f:
        content = yaml.safe_load(f)

    profile_time = profile_data.get("profile_time", {})
    profile_memory = profile_data.get("profile_memory", {})
    if profile_time:
        if "profile_time" not in content:
            content["profile_time"] = {}
        content["profile_time"].update(profile_time)
    if profile_memory:
        if "profile_memory" not in content:
            content["profile_memory"] = {}
        content["profile_memory"].update(profile_memory)

    with open(new_config_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)

    print(f"Updated profile_time/profile_memory in new file: {new_config_path}")


embodied_path = os.environ["EMBODIED_PATH"]
config_name = os.environ["CONFIG_NAME"]
log_dir = os.environ["LOG_DIR"]

profile_data = extract_from_tensorboard(log_dir)
update_yaml_with_profile_data(f"{embodied_path}/config", config_name, profile_data)
PY


echo "Done. Profiled yaml written next to the original config (_profiled.yaml)." | tee -a "${MEGA_LOG_FILE}"
