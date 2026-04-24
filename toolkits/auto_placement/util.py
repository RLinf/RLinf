# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import yaml
from fitter import DataFitter
from omegaconf import DictConfig, OmegaConf, open_dict

_GLOBAL_CONFIG = None


def init_global_config(config, component_placement, cluster) -> None:
    if config.runner.task_type == "reasoning":
        init_global_config_reasoning(config, component_placement)
    else:
        init_global_config_env(config, component_placement, cluster)


def init_global_config_reasoning(config, component_placement) -> None:
    global _GLOBAL_CONFIG

    _GLOBAL_CONFIG = Namespace(
        task_type=config.runner.task_type,
        total_gpus=component_placement._cluster_num_gpus,
        group_size=config.algorithm.group_size,
        n_minibatches=config.algorithm.n_minibatches,
        rollout_batch_size=config.data.rollout_batch_size,
        seq_length=config.runner.seq_length,
        max_running_requests=config.rollout.max_running_requests,
        gpu_memory_utilization=config.rollout.gpu_memory_utilization,
        components_config={},
    )

    for component in component_placement._components:
        if component == "reward":
            continue
        instance_num = getattr(component_placement, f"{component}_dp_size")
        world_size = getattr(component_placement, f"{component}_world_size")
        model_parallel_size = world_size // instance_num

        _GLOBAL_CONFIG.components_config[component] = Namespace(
            model_parallel_size=model_parallel_size,
            max_world_size=world_size,
            collocated_cost_total=getattr(config.profile_data, f"{component}_cost"),
        )

    if "inference" not in component_placement._components:
        model_parallel_size = _GLOBAL_CONFIG.components_config[
            "actor"
        ].model_parallel_size
        world_size = _GLOBAL_CONFIG.components_config["actor"].max_world_size
        _GLOBAL_CONFIG.components_config["inference"] = Namespace(
            model_parallel_size=model_parallel_size,
            max_world_size=world_size,
            collocated_cost_total=getattr(config.profile_data, "inference_cost"),
        )


def init_global_config_env(config, component_placement, cluster) -> None:
    global _GLOBAL_CONFIG

    # Use actor_profile_time (total_num_envs -> time). Prefer `profile_time.*`,
    # fallback to legacy `profile_data.*` for backward compatibility.
    actor_profile_cfg = OmegaConf.select(config, "profile_time.actor_profile_time")
    if actor_profile_cfg is None:
        actor_profile_cfg = OmegaConf.select(config, "profile_data.actor_profile_data")
    if not isinstance(actor_profile_cfg, DictConfig) or actor_profile_cfg == {}:
        raise ValueError(
            "Missing profile_time.actor_profile_time (or legacy profile_data.actor_profile_data). "
            "Please run collect_profile to generate profiling data."
        )
    actor_profile_data = OmegaConf.to_container(actor_profile_cfg, resolve=True)
    actor_profile_data = {int(k): float(v) for k, v in actor_profile_data.items()}

    total_num_envs = OmegaConf.select(config, "env.train.total_num_envs")
    if total_num_envs is None:
        total_num_envs = OmegaConf.select(config, "data.env_num")
    if total_num_envs is None:
        raise ValueError(
            "Missing env.train.total_num_envs (and data.env_num fallback). "
            "Please set env.train.total_num_envs in config."
        )
    total_num_envs = int(total_num_envs)
    actor_cost_total = DataFitter(actor_profile_data).get_value(total_num_envs)

    _GLOBAL_CONFIG = Namespace(
        task_type=config.runner.task_type,
        total_gpus=cluster.num_accelerators,
        env_num=total_num_envs,
        pipeline_stage_num=config.rollout.pipeline_stage_num,
        profile_time=getattr(
            config, "profile_time", getattr(config, "profile_data", None)
        ),
        profile_memory=getattr(config, "profile_memory", None),
        rollout_batch_size=1,  # For actor node init
        group_size=1,  # For actor node init
        n_minibatches=1,  # For actor node init
        components_config={},
    )

    for component in component_placement._components:
        instance_num: int = component_placement.get_world_size(component)
        world_size: int = component_placement.get_world_size(component)
        model_parallel_size = (
            world_size // instance_num
        )  # For env and rollout in embodiment task, we set dp_size = world_size, so model_parallel_size = 1

        if component == "rollout":
            component = "env_rollout"
            _GLOBAL_CONFIG.components_config[component] = Namespace(
                model_parallel_size=model_parallel_size,
                max_world_size=world_size,
            )
        elif component == "env":
            _GLOBAL_CONFIG.components_config[component] = Namespace(
                model_parallel_size=model_parallel_size,
                max_world_size=world_size,
            )
        else:
            _GLOBAL_CONFIG.components_config[component] = Namespace(
                model_parallel_size=model_parallel_size,
                max_world_size=world_size,
                collocated_cost_total=(
                    actor_cost_total
                    if component == "actor"
                    else getattr(config.profile_data, f"{component}_cost")
                ),
            )


def get_global_config():
    global _GLOBAL_CONFIG
    assert _GLOBAL_CONFIG is not None, "Global config has not been set"
    return _GLOBAL_CONFIG


def get_valid_gpu_num_list(role: str) -> list[int]:
    """Get valid gpu num list for the component based on the constraints of batch and group size."""
    config = get_global_config()

    global_step_batch_size = config.rollout_batch_size * config.group_size
    assert global_step_batch_size % config.n_minibatches == 0, (
        f"global_step_batch_size={global_step_batch_size} must be divisible by train_iter={config.n_minibatches}"
    )
    trainer_iter_batch_size = global_step_batch_size // config.n_minibatches

    valid_dp_sizes = []

    model_parallel_size = config.components_config[role].model_parallel_size

    max_dp_size = config.total_gpus // model_parallel_size
    for dp_size in range(1, max_dp_size + 1):
        if trainer_iter_batch_size % (dp_size * config.group_size) == 0:
            valid_dp_sizes.append(dp_size)

    return [dp_size * model_parallel_size for dp_size in valid_dp_sizes]


def has_profile_data(cfg) -> bool:
    """Check if there is profile data in the config."""
    if (
        OmegaConf.select(cfg, "profile_time.actor_profile_time") is None
        and OmegaConf.select(cfg, "profile_data.actor_profile_data") is None
    ):
        return False
    env_profile = OmegaConf.select(cfg, "profile_time.env_profile_time")
    if env_profile is None:
        env_profile = OmegaConf.select(cfg, "profile_data.env_profile_data")
    if not isinstance(env_profile, DictConfig) or env_profile == {}:
        return False
    rollout_profile = OmegaConf.select(cfg, "profile_time.rollout_profile_time")
    if rollout_profile is None:
        rollout_profile = OmegaConf.select(cfg, "profile_data.rollout_profile_data")
    if not isinstance(rollout_profile, DictConfig) or rollout_profile == {}:
        return False
    return True


def has_target_env_num(cfg) -> bool:
    target_env_num = OmegaConf.select(cfg, "env.train.total_num_envs")
    if target_env_num is None:
        target_env_num = OmegaConf.select(cfg, "data.env_num")
    return target_env_num is not None


def modify_cfg_for_profiling(cfg, env_num):
    with open_dict(cfg):
        cfg.cluster.component_placement = {
            "actor": "all",
            "rollout": "all",
            "env": "all",
        }  # Ensure collocated mode for profiling
        cfg.env.train.total_num_envs = env_num
        cfg.runner.max_steps = (
            OmegaConf.select(
                cfg, "data.step_per_env", default=3
            )  # Run the specified number of steps for profiling
        )
        cfg.runner.logger.per_worker_log = True
        cfg.rollout.pipeline_stage_num = 1


def modify_cfg_for_profiling_with_group(cfg, env_num, run_idx: int) -> None:
    """Profiling config + unique group names to avoid Ray actor name collisions."""
    modify_cfg_for_profiling(cfg, env_num)
    base_actor_group_name = cfg.actor.group_name
    base_rollout_group_name = cfg.rollout.group_name
    base_env_group_name = cfg.env.group_name
    with open_dict(cfg):
        cfg.actor.group_name = f"{base_actor_group_name}_env{env_num}_run{run_idx}"
        cfg.rollout.group_name = f"{base_rollout_group_name}_env{env_num}_run{run_idx}"
        cfg.env.group_name = f"{base_env_group_name}_env{env_num}_run{run_idx}"


def update_yaml_with_profile_data(profile_data):
    parser = ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    args, _ = parser.parse_known_args()
    config_path = args.config_path
    config_name = args.config_name

    original_config_path = f"{config_path}/{config_name}.yaml"
    new_config_path = original_config_path.replace(".yaml", "_profiled.yaml")

    with open(original_config_path, "r") as f:
        content = yaml.safe_load(f)

    # New format: split time and memory.
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

    logging.info(f"Updated profile_time/profile_memory in new file: {new_config_path}")


def log_env_num_instavg_to_tensorboard(cfg, env_num_per_instance: int) -> None:
    """Log env_num_per_instance to TensorBoard"""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "tensorboard is required for env_num_per_instance logging. Try `pip install tensorboard`."
        ) from exc

    log_root = cfg.runner.logger.log_path
    log_suffix = "all" if cfg.runner.get("per_worker_log", False) else ""
    tb_logdir = os.path.join(log_root, "tensorboard", log_suffix)
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(tb_logdir)
    writer.add_scalar("profile/env_num_per_instance", env_num_per_instance, 0)
    writer.flush()
    writer.close()


def log_total_num_envs_to_tensorboard(cfg, total_num_envs: int) -> None:
    """Log total_num_envs (env.train.total_num_envs for this run) to TensorBoard."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "tensorboard is required for total_num_envs logging. Try `pip install tensorboard`."
        ) from exc

    log_root = cfg.runner.logger.log_path
    log_suffix = "all" if cfg.runner.get("per_worker_log", False) else ""
    tb_logdir = os.path.join(log_root, "tensorboard", log_suffix)
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(tb_logdir)
    writer.add_scalar("profile/total_num_envs", int(total_num_envs), 0)
    writer.flush()
    writer.close()


def _find_run_dirs(logdir: Path) -> list[Path]:
    event_prefix = "events.out.tfevents"
    run_dirs: list[Path] = []
    for root, _dirs, files in os.walk(logdir):
        if any(f.startswith(event_prefix) for f in files):
            run_dirs.append(Path(root))
    return sorted(set(run_dirs))


def _load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "tensorboard is required. Try `pip install tensorboard`."
        ) from exc

    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={"scalars": 0},
    )
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


def extract_from_tensorboard():
    log_dir_str = os.environ.get("LOG_DIR")

    if not log_dir_str:
        print("# Error: $LOG_DIR environment variable is not set.")
        return {}

    logdir = Path(log_dir_str)
    if not logdir.exists():
        print(f"# Error: logdir not found: {logdir}")
        return {}

    run_dirs = _find_run_dirs(logdir)
    if not run_dirs:
        print(f"# No TensorBoard event files found under: {logdir}")
        return {}

    # Time profile data
    env_time_profile_data: dict[int, float] = {}
    rollout_time_profile_data: dict[int, float] = {}
    actor_time_profile_data: dict[int, float] = {}
    # Memory profile data (%util)
    env_memory_util_profile_data: dict[int, float] = {}
    rollout_memory_util_profile_data: dict[int, float] = {}

    def _vals_in_window(events, t_start, t_end, inclusive_end: bool = False):
        """Return values with timestamps in (t_start, t_end] if inclusive_end else [t_start, t_end)."""
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

        # Profile markers: each run logs env_num_per_instance then total_num_envs
        # with consecutive wall times, so the total is always strictly after the
        # env marker. Segment i uses t_end = env marker i; a time window would
        # exclude that run's total and shift keys — pair by index instead.
        for i, (t_end, env_num_val) in enumerate(env_events):
            t_start = env_events[i - 1][0] if i > 0 else 0.0
            env_num_per_instance = int(env_num_val)
            if i < len(total_env_events):
                total_num_envs = int(total_env_events[i][1])
            else:
                total_env_vals = _vals_in_window(
                    total_env_events, t_start, t_end, inclusive_end=True
                )
                total_num_envs = int(total_env_vals[-1]) if total_env_vals else None
            env_vals = _vals_in_window(
                env_cost_events, t_start, t_end, inclusive_end=True
            )
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
                env_time_profile_data[env_num_per_instance] = sum(env_vals) / len(
                    env_vals
                )
            if rollout_vals:
                rollout_time_profile_data[env_num_per_instance] = sum(
                    rollout_vals
                ) / len(rollout_vals)
            if actor_vals and total_num_envs is not None:
                actor_time_profile_data[total_num_envs] = sum(actor_vals) / len(
                    actor_vals
                )

            if env_mem_util_vals:
                env_memory_util_profile_data[env_num_per_instance] = sum(
                    env_mem_util_vals
                ) / len(env_mem_util_vals)
            if rollout_mem_util_vals:
                rollout_memory_util_profile_data[env_num_per_instance] = sum(
                    rollout_mem_util_vals
                ) / len(rollout_mem_util_vals)

    profile_time: dict[str, dict[int, float]] = {}
    if env_time_profile_data:
        profile_time["env_profile_time"] = env_time_profile_data
    if rollout_time_profile_data:
        profile_time["rollout_profile_time"] = rollout_time_profile_data
    if actor_time_profile_data:
        profile_time["actor_profile_time"] = actor_time_profile_data

    profile_memory: dict[str, dict[int, float]] = {}
    if env_memory_util_profile_data:
        profile_memory["env_profile_memory"] = env_memory_util_profile_data
    if rollout_memory_util_profile_data:
        profile_memory["rollout_profile_memory"] = rollout_memory_util_profile_data

    profile_data: dict[str, dict] = {}
    if profile_time:
        profile_data["profile_time"] = profile_time
    if profile_memory:
        profile_data["profile_memory"] = profile_memory

    print(profile_data)

    return profile_data
