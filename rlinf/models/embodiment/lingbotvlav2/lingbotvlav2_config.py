# Copyright 2026 The RLinf Authors.
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

import torch
from omegaconf import DictConfig

from rlinf.utils.precision import torch_dtype_from_precision


def validate_actor_fsdp(actor_cfg: DictConfig) -> None:
    fsdp_cfg = actor_cfg.fsdp_config
    if fsdp_cfg.strategy != "fsdp2":
        raise ValueError(
            "Expected actor.fsdp_config.strategy=fsdp2 for LingBot-VLA V2, "
            f"got {fsdp_cfg.strategy!r}."
        )
    param_dtype = fsdp_cfg.mixed_precision.get("param_dtype", None)
    if (
        torch_dtype_from_precision(actor_cfg.model.precision) != torch.float32
        or torch_dtype_from_precision(param_dtype) != torch.bfloat16
    ):
        raise ValueError(
            "LingBot-VLA V2 PPO keeps trainable weights in fp32 and computes "
            "in bf16; expected actor.model.precision=fp32 and "
            "actor.fsdp_config.mixed_precision.param_dtype=bf16, got "
            f"{actor_cfg.model.precision!r} and {param_dtype!r}."
        )
    if fsdp_cfg.mixed_precision.get("cast_forward_inputs", True):
        raise ValueError(
            "Expected actor.fsdp_config.mixed_precision.cast_forward_inputs="
            "false for LingBot-VLA V2, which keeps its SDE samples in fp32."
        )


def validate_batch_shape(
    cfg: DictConfig, rollout_world_size: int, stage_num: int
) -> None:
    """Keep rollout scoring and actor replay at one MoE/GEMM batch shape.

    A bf16 grouped GEMM gives a different result at a different batch size, so
    both passes must split their batches into chunks of the same size.
    """
    fmb = cfg.actor.model.forward_micro_batch_size
    rollout_fmb = cfg.rollout.model.get("forward_micro_batch_size", fmb)
    if rollout_fmb != fmb:
        raise ValueError(
            "Expected rollout.model.forward_micro_batch_size to equal "
            f"actor.model.forward_micro_batch_size ({fmb}), got {rollout_fmb}."
        )
    if cfg.actor.micro_batch_size % fmb != 0:
        raise ValueError(
            f"Expected actor.micro_batch_size to be a multiple of "
            f"actor.model.forward_micro_batch_size ({fmb}), got "
            f"{cfg.actor.micro_batch_size}."
        )
    envs_per_stage = cfg.env.train.total_num_envs // rollout_world_size // stage_num
    if envs_per_stage % fmb != 0:
        raise ValueError(
            "Expected the envs per rollout worker and pipeline stage "
            "(env.train.total_num_envs / rollout workers / "
            "rollout.pipeline_stage_num) to be a multiple of "
            f"actor.model.forward_micro_batch_size ({fmb}), got {envs_per_stage}."
        )
