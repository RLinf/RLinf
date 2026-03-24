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

    _GLOBAL_CONFIG = Namespace(
        task_type=config.runner.task_type,
        total_gpus=cluster.num_accelerators,
        env_num=config.data.env_num,
        profile_data=config.profile_data,
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
                collocated_cost_total=getattr(config.profile_data, f"{component}_cost"),
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
    if OmegaConf.select(cfg, "profile_data.actor_cost") is None:
        return False
    env_profile = OmegaConf.select(cfg, "profile_data.env_profile_data")
    if not isinstance(env_profile, DictConfig) or env_profile == {}:
        return False
    rollout_profile = OmegaConf.select(cfg, "profile_data.rollout_profile_data")
    if not isinstance(rollout_profile, DictConfig) or rollout_profile == {}:
        return False
    return True


def has_target_env_num(cfg) -> bool:
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

    if "profile_data" not in content:
        content["profile_data"] = {}
    content["profile_data"].update(profile_data)

    with open(new_config_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)

    logging.info(f"Updated profile_data in new file: {new_config_path}")


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
        "time/env/interact",
        "time/rollout/predict",
        "time/actor/run_training",
        "profile/env_num_per_instance",
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

    env_profile_data: dict[int, float] = {}
    rollout_profile_data: dict[int, float] = {}
    actor_costs_all: list[float] = []

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
        env_cost_events = data.get("time/env/interact", [])
        rollout_cost_events = data.get("time/rollout/predict", [])
        actor_cost_events = data.get("time/actor/run_training", [])

        # Profile events are logged AFTER each run. So for event i at t_i, run i's metrics
        # have timestamps in (t_{i-1}, t_i]. Use inclusive_end to capture the last run.
        for i, (t_end, env_num_val) in enumerate(env_events):
            t_start = env_events[i - 1][0] if i > 0 else 0.0
            env_num = int(env_num_val)
            env_vals = _vals_in_window(
                env_cost_events, t_start, t_end, inclusive_end=True
            )
            rollout_vals = _vals_in_window(
                rollout_cost_events, t_start, t_end, inclusive_end=True
            )
            actor_vals = _vals_in_window(
                actor_cost_events, t_start, t_end, inclusive_end=True
            )

            if env_vals:
                env_profile_data[env_num] = sum(env_vals) / len(env_vals)
            if rollout_vals:
                rollout_profile_data[env_num] = sum(rollout_vals) / len(rollout_vals)
            if actor_vals:
                # collect per-env actor costs; later we select base env_num
                actor_costs_all.extend([(env_num, v) for v in actor_vals])

    profile_data = {}
    if env_profile_data:
        profile_data["env_profile_data"] = env_profile_data
    if rollout_profile_data:
        profile_data["rollout_profile_data"] = rollout_profile_data
    if actor_costs_all:
        base_env_num = min(env_profile_data.keys()) if env_profile_data else None
        if base_env_num is not None:
            base_actor_vals = [v for env, v in actor_costs_all if env == base_env_num]
            if base_actor_vals:
                profile_data["actor_cost"] = sum(base_actor_vals) / len(base_actor_vals)

    print(profile_data)

    return profile_data
