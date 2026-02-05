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
from typing import TYPE_CHECKING, Union

from omegaconf import DictConfig

if TYPE_CHECKING:
    from rlinf.workers.inference.fsdp_inference_worker import FSDPInference
    from rlinf.workers.inference.megatron_inference_worker import MegatronActorInference, MegatronCriticInference


def get_inference_backend_worker(
    cfg: DictConfig, role = 'actor',
) -> Union["FSDPInference", "MegatronActorInference"]:
    """Get the inference backend worker class based on the training backend.

    Args:
        cfg (DictConfig): Configuration for the inference task.

    Returns:
        Inference worker class.
    """
    assert role == "actor" or role == "critic", \
        "in get_inference_backend_worker: argument role can only be actor or critic, " \
        f"but now is {role}"
    training_backend = getattr(cfg, role).training_backend
    if training_backend == "megatron":
        if role == "actor":
            from rlinf.workers.inference.megatron_inference_worker import (
                MegatronActorInference,
            )

            return MegatronActorInference
        elif role == "critic":
            from rlinf.workers.inference.megatron_inference_worker import (
                MegatronCriticInference,
            )

            return MegatronCriticInference
        else:
            raise ValueError(f"Unknown role '{role}' for get_inference_backend_worker")

    elif training_backend == "fsdp":
        if role == "actor":
            from rlinf.workers.inference.fsdp_inference_worker import FSDPInference

            return FSDPInference

        elif role == "critic":
            raise ValueError(f"PPO for reasoning is not implemented for FSDP backend yet")
        else:
            raise ValueError(f"Unknown role '{role}' for get_inference_backend_worker")
    else:
        raise ValueError(
            f"Unsupported training backend for inference: {training_backend}"
        )
