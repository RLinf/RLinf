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

from abc import ABC, abstractmethod
from typing import TypeVar

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    parallelize_module,
)

T = TypeVar("T", bound="TensorParallelizerBase")


class TensorParallelizerBase(ABC):
    _registry: dict[str, type["TensorParallelizerBase"]] = {}
    _by_model_type: dict[str, str] = {}

    @abstractmethod
    def _create_parallel_plan(
        self,
        hf_model: nn.Module,
    ) -> dict[str, ParallelStyle]: ...

    def parallelize(
        self,
        hf_model: nn.Module,
        tp_mesh: DeviceMesh,
    ) -> nn.Module:
        parallel_plan = self._create_parallel_plan(hf_model)
        return parallelize_module(hf_model, tp_mesh, parallel_plan)

    @classmethod
    def register(cls, key: str, *, model_type: str | None = None):
        """
        @TensorParallelizerBase.register("deepseek_qwen_1_5b", model_type="qwen2")
        class DeepSeekQwenParallelizer(TensorParallelizerBase):
            ...
        """

        def decorator(par_cls: type[T]) -> type[T]:
            if key in cls._registry:
                raise ValueError(f"Parallelizer key {key!r} already registered")
            cls._registry[key] = par_cls
            if model_type is not None:
                cls._by_model_type[model_type] = key
            return par_cls

        return decorator

    @classmethod
    def create(cls, key: str, **kwargs) -> "TensorParallelizerBase":
        try:
            par_cls = cls._registry[key]
        except KeyError:
            raise KeyError(f"No parallelizer registered under key {key!r}")
        return par_cls(**kwargs)

    @classmethod
    def create_for_model(
        cls,
        hf_model: nn.Module,
        **kwargs,
    ) -> "TensorParallelizerBase":
        config = getattr(hf_model, "config", None)
        model_type = getattr(config, "model_type", None)
        if model_type is None:
            raise ValueError("hf_model.config.model_type not found")

        try:
            key = cls._by_model_type[model_type]
        except KeyError:
            raise KeyError(f"No parallelizer registered for model_type {model_type!r}")
        return cls.create(key, **kwargs)


@TensorParallelizerBase.register("qwen2.5")
class QwenTensorParallelizer(TensorParallelizerBase):
    """
    Tensor parallelizer for Qwen2 models.
    """

    @classmethod
    def _create_parallel_plan(cls, hf_model: nn.Module) -> dict[str, ParallelStyle]:
        parallel_plan = {}
        parallel_plan["model.embed_tokens"] = RowwiseParallel(input_layouts=Replicate())
        parallel_plan["lm_head"] = ColwiseParallel()

        for layer_id, _layer in enumerate(hf_model.model.layers):
            p = f"model.layers.{layer_id}"
            parallel_plan[f"{p}.self_attn.q_proj"] = ColwiseParallel()
            parallel_plan[f"{p}.self_attn.k_proj"] = ColwiseParallel()
            parallel_plan[f"{p}.self_attn.v_proj"] = ColwiseParallel()
            parallel_plan[f"{p}.self_attn.o_proj"] = RowwiseParallel()

            parallel_plan[f"{p}.mlp.gate_proj"] = ColwiseParallel()
            parallel_plan[f"{p}.mlp.up_proj"] = ColwiseParallel()
            parallel_plan[f"{p}.mlp.down_proj"] = RowwiseParallel()

        return parallel_plan
