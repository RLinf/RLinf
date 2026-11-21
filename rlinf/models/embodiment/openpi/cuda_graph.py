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

"""Full-sampler CUDA graph and static input buffers for Pi0 evaluation."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_map

from rlinf.models.embodiment.openpi.modules.model import Observation
from rlinf.utils.cuda_graph import CUDAGraphManager, GraphCaptureSpec


class EvalCudaGraph:
    """Capture and replay an observation sampler with fixed input buffers."""

    def __init__(
        self,
        *,
        device: torch.device,
        sampler: Callable[..., torch.Tensor],
    ) -> None:
        self.sampler = sampler
        self.manager = CUDAGraphManager(device=device, create_graph_pool=False)
        self.capture_stream = torch.cuda.Stream(device=device)
        self.signature = None

    def clear(self) -> None:
        """Release captured GPU resources."""
        with torch.cuda.device(self.manager.device):
            self.manager.destroy()
        self.signature = None

    @staticmethod
    def _inputs(observation: Observation, noise: torch.Tensor) -> dict[str, Any]:
        return {
            "observation": {
                field.name: getattr(observation, field.name)
                for field in dataclasses.fields(observation)
            },
            "noise": noise,
        }

    def _input_signature(
        self, observation: Observation, noise: torch.Tensor, num_steps: int
    ) -> tuple:
        leaves, structure = tree_flatten(self._inputs(observation, noise))
        return (
            num_steps,
            structure,
            tuple(
                (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
                if isinstance(value, torch.Tensor)
                else value
                for value in leaves
            ),
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
        )

    def needs_capture(
        self, observation: Observation, noise: torch.Tensor, num_steps: int
    ) -> bool:
        """Check whether the input layout or sampling math needs a new graph."""
        return self.signature != self._input_signature(observation, noise, num_steps)

    def capture(
        self, observation: Observation, noise: torch.Tensor, num_steps: int
    ) -> None:
        """Capture the sampler using independent static input buffers."""
        signature = self._input_signature(observation, noise, num_steps)
        with torch.cuda.device(self.manager.device):
            static = tree_map(
                lambda value: (
                    value.detach().clone() if isinstance(value, torch.Tensor) else value
                ),
                self._inputs(observation, noise),
            )

            def sample(graph_inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
                return {
                    "actions": self.sampler(
                        Observation(**graph_inputs["observation"]),
                        num_steps=num_steps,
                        noise=graph_inputs["noise"],
                    )
                }

            current_stream = torch.cuda.current_stream(self.manager.device)
            self.capture_stream.wait_stream(current_stream)
            try:
                with (
                    torch.cuda.stream(self.capture_stream),
                    torch.autocast(
                        "cuda",
                        enabled=torch.is_autocast_enabled("cuda"),
                        dtype=torch.get_autocast_dtype("cuda"),
                        cache_enabled=False,
                    ),
                ):
                    self.manager.capture(
                        GraphCaptureSpec(
                            name="sample",
                            func=sample,
                            inputs=static,
                            external_inputs={"observation", "noise"},
                        ),
                        use_shared_pool=False,
                    )
            except Exception:
                self.clear()
                raise
            current_stream.wait_stream(self.capture_stream)
            self.signature = signature

    def replay(self, observation: Observation, noise: torch.Tensor) -> torch.Tensor:
        """Copy current inputs and return an independent sampled action tensor."""
        with torch.cuda.device(self.manager.device):
            outputs = self.manager.replay("sample", self._inputs(observation, noise))
            return outputs["actions"].clone()
