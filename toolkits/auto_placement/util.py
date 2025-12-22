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

from argparse import Namespace

_GLOBAL_CONFIG = None


def init_global_config(config, component_placement):
    if config.runner.task_type == "reasoning":
        init_global_config_reasoning(config, component_placement)
    else:
        init_global_config_env(config, component_placement)


def init_global_config_reasoning(config, component_placement):
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


def init_global_config_env(config, component_placement):
    global _GLOBAL_CONFIG

    _GLOBAL_CONFIG = Namespace(
        task_type=config.runner.task_type,
        total_gpus=component_placement._cluster_num_gpus,
        env_num=config.data.env_num,
        profile_data=config.profile_data,
        rollout_batch_size=1,  # For actor node init
        group_size=1,  # For actor node init
        n_minibatches=1,  # For actor node init
        components_config={},
    )

    for component in component_placement._components:
        instance_num = getattr(component_placement, f"{component}_dp_size")
        world_size = getattr(component_placement, f"{component}_world_size")
        model_parallel_size = world_size // instance_num

        if component == "rollout":
            component = "env_rollout"
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
