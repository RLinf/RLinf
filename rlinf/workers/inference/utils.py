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
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from rlinf.utils.placement import ModelParallelComponentPlacement

if TYPE_CHECKING:
    from rlinf.workers.inference.fsdp_inference_worker import FSDPInference
    from rlinf.workers.inference.megatron_inference_worker import MegatronInference


def get_inference_backend_worker(
    cfg: DictConfig, placement: ModelParallelComponentPlacement
) -> "FSDPInference" | "MegatronInference":
    """Get the inference backend worker class based on the training backend.

    Args:
        cfg (DictConfig): Configuration for the inference task.
        placement (ComponentPlacement): Component placement information.

    Returns:
        Inference worker class.
    """
    training_backend = cfg.actor.training_backend
    if training_backend == "megatron":
        return MegatronInference(cfg, placement)
    elif training_backend == "fsdp":
        return FSDPInference(cfg, placement)
    else:
        raise ValueError(
            f"Unsupported training backend for inference: {training_backend}"
        )
