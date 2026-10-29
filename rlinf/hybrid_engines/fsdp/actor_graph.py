# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forward/backward graph replay for a resident, single-rank FSDP1 actor."""

from collections.abc import Callable

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from rlinf.scheduler import Worker
from rlinf.utils.logging import get_logger


def validate_actor_graph_cfg(actor_cfg: DictConfig) -> None:
    """Reject training modes that change captured parameter or gradient storage."""
    fsdp = actor_cfg.get("fsdp_config", {})
    if (
        fsdp.get("strategy") != "fsdp"
        or fsdp.get("sharding_strategy") != "no_shard"
        or not fsdp.get("use_orig_params", False)
    ):
        raise ValueError(
            "Actor CUDA Graph requires FSDP1 no_shard with use_orig_params=True"
        )
    if (
        fsdp.get("gradient_checkpointing", False)
        or OmegaConf.select(
            actor_cfg, "fsdp_config.amp_autocast.enabled", default=False
        )
        or OmegaConf.select(actor_cfg, "fsdp_config.grad_scaler.enabled", default=False)
        or OmegaConf.select(actor_cfg, "fsdp_config.mixed_precision.param_dtype")
        is not None
        or fsdp.get("mixed_precision_rules")
        or fsdp.get("cpu_offload", False)
    ):
        raise ValueError(
            "Actor CUDA Graph requires resident parameters without checkpointing, AMP or FSDP casting"
        )
    if OmegaConf.select(actor_cfg, "optim.critic_warmup_steps", default=0):
        raise ValueError("Actor CUDA Graph does not support critic-only warmup")


class FSDPActorGraph:
    """Capture the network and its backward; keep loss and optimizer outside.

    One owner serves serial forward/backward pairs with a fixed input layout.
    Gradients accumulate in persistent buffers: callers must zero them with
    ``set_to_none=False`` between optimizer steps. Clear before offloading or
    replacing parameters. FSDP sharding and multi-rank communication are not
    supported by this runner.
    """

    _STREAM_NAMES = (
        "_unshard_stream",
        "_pre_unshard_stream",
        "_post_backward_stream",
        "_all_reduce_stream",
        "_default_stream",
    )

    def __init__(self, model: nn.Module):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if Worker.torch_device_type != "cuda":
            raise ValueError("Actor CUDA Graph requires NVIDIA CUDA")
        if not isinstance(model, FSDP) or torch.distributed.get_world_size() != 1:
            raise ValueError("Actor CUDA Graph requires single-rank FSDP1")
        self.modules = FSDP.fsdp_modules(model)
        for module in self.modules:
            handle = module._handle
            if handle is not None and (
                handle.uses_sharded_strategy or handle._uses_param_mixed_precision
            ):
                raise ValueError(
                    "Actor CUDA Graph requires NO_SHARD without FSDP parameter casting"
                )
        self.model = model
        self.platform = Worker.torch_platform
        self.stream = self.platform.Stream()
        self.forward_graph = None
        self.backward_graph = None
        self.inputs = {}
        self.outputs = {}
        self.output_grads = {}
        self.parameters = ()
        self.gradients = ()
        self.signature = None
        self.pending = False
        self.captures = self.forward_replays = self.backward_replays = 0

    @staticmethod
    def _signature(inputs: dict[str, torch.Tensor], key: tuple) -> tuple:
        return key, tuple(
            (name, tuple(tensor.shape), tensor.dtype, tensor.device)
            for name, tensor in inputs.items()
        )

    def _join_streams(self) -> None:
        # Every FSDP side stream must enter and rejoin this capture.
        current = self.platform.current_stream()
        streams = {
            getattr(module, name, None)
            for module in self.modules
            for name in self._STREAM_NAMES
        }
        for stream in streams:
            if stream is not None and stream != current:
                stream.wait_stream(current)
                current.wait_stream(stream)

    def _capture(self, function: Callable, inputs: dict, signature: tuple) -> None:
        if any(t.device.type != "cuda" or t.requires_grad for t in inputs.values()):
            raise ValueError(
                "Actor graph inputs must be CUDA tensors without gradients"
            )
        if any(p.grad is not None for p in self.model.parameters()):
            raise ValueError("Capture the actor graph before accumulating gradients")
        self.inputs = {name: tensor.detach().clone() for name, tensor in inputs.items()}
        self.stream.wait_stream(self.platform.current_stream())
        with torch.random.fork_rng(devices=[self.platform.current_device()]):
            with self.platform.stream(self.stream):
                for _ in range(3):
                    outputs = function(self.inputs)
                    differentiable = tuple(
                        t for t in outputs.values() if t.requires_grad
                    )
                    torch.autograd.backward(
                        differentiable,
                        tuple(torch.ones_like(t) for t in differentiable),
                    )
                    self.model.zero_grad(set_to_none=False)
            self.platform.current_stream().wait_stream(self.stream)
            self.platform.synchronize()
            # A model may have initialized FSDP on the legacy default stream
            # while syncing weights. Use the capture stream for these callbacks.
            defaults = [
                (module, name, getattr(module, name))
                for module in self.modules
                for name in self._STREAM_NAMES
                if getattr(module, name, None) is not None
                and (
                    name == "_default_stream" or getattr(module, name).cuda_stream == 0
                )
            ]
            for module, name, _ in defaults:
                setattr(module, name, self.stream)
            try:
                self.forward_graph = self.platform.CUDAGraph()
                self.backward_graph = self.platform.CUDAGraph()
                with self.platform.graph(self.forward_graph, stream=self.stream):
                    self._join_streams()
                    self.outputs = function(self.inputs)
                    self._join_streams()
                self.output_grads = {
                    name: torch.zeros_like(tensor)
                    for name, tensor in self.outputs.items()
                    if tensor.requires_grad
                }
                with self.platform.graph(self.backward_graph, stream=self.stream):
                    self._join_streams()
                    torch.autograd.backward(
                        tuple(self.outputs[name] for name in self.output_grads),
                        tuple(self.output_grads.values()),
                    )
                    self._join_streams()
            finally:
                for module, name, stream in defaults:
                    setattr(module, name, stream)
        self.platform.current_stream().wait_stream(self.stream)
        self.model.zero_grad(set_to_none=False)
        self.parameters = tuple(p for p in self.model.parameters() if p.requires_grad)
        self.gradients = tuple(p.grad for p in self.parameters)
        if any(grad is None for grad in self.gradients):
            raise RuntimeError(
                "Actor graph must produce every trainable parameter gradient"
            )
        self.parameter_addresses = tuple(p.data_ptr() for p in self.parameters)
        self.signature = signature
        self.captures += 1
        get_logger().info("Captured actor forward/backward CUDA graphs")

    def forward(self, function: Callable, inputs: dict, *, key: tuple = ()) -> dict:
        """Replay a network forward and return independent leaves for the loss."""
        if self.pending:
            raise RuntimeError("Complete actor graph backward before another forward")
        signature = self._signature(inputs, key)
        if self.forward_graph is None:
            self._capture(function, inputs, signature)
        elif signature != self.signature:
            raise ValueError(
                "Actor CUDA Graph requires a fixed input layout within an update"
            )
        if tuple(p.data_ptr() for p in self.parameters) != self.parameter_addresses:
            raise RuntimeError(
                "Actor parameters moved; clear graphs before offload or reload"
            )
        if any(
            p.grad is None or p.grad.data_ptr() != grad.data_ptr()
            for p, grad in zip(self.parameters, self.gradients, strict=True)
        ):
            raise RuntimeError(
                "Actor graph gradients require zero_grad(set_to_none=False)"
            )
        for name, tensor in inputs.items():
            self.inputs[name].copy_(tensor)
        self.forward_graph.replay()
        self.forward_replays += 1
        self.pending = True
        return {
            name: tensor.detach().clone().requires_grad_(name in self.output_grads)
            for name, tensor in self.outputs.items()
        }

    def backward(self, outputs: dict) -> None:
        """Accumulate parameter gradients using gradients of the loss leaves."""
        if not self.pending:
            raise RuntimeError("Actor graph backward requires a preceding forward")
        for name, target in self.output_grads.items():
            gradient = outputs[name].grad
            target.zero_() if gradient is None else target.copy_(gradient)
        self.backward_graph.replay()
        self.backward_replays += 1
        self.pending = False

    def clear(self) -> None:
        """Release captured storage after the optimizer step and before offload."""
        self.platform.synchronize()
        self.forward_graph = self.backward_graph = None
        self.inputs.clear()
        self.outputs.clear()
        self.output_grads.clear()
        self.parameters = self.gradients = ()
        self.pending = False
        self.signature = None
