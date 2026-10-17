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

import gc
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch


def _graph_backend(device: torch.device):
    """Device-specific graph calls for CUDA and Ascend NPU.

    torch_npu mirrors the CUDA graph API (NPUGraph, the ``graph`` context
    manager, pool handles), so one manager serves both; NPU graphs do not
    expose ``register_generator_state``, so generator registration is a
    CUDA-only feature and callers must keep sampling outside the captured
    region on NPU.
    """
    if device.type == "npu":
        return _NpuGraphBackend()
    if device.type == "cuda":
        return _CudaGraphBackend()
    raise ValueError(f"Graph capture requires CUDA or NPU, got {device}.")


class _CudaGraphBackend:
    @staticmethod
    def new_graph():
        return torch.cuda.CUDAGraph()

    @staticmethod
    def graph_ctx(graph, pool):
        return torch.cuda.graph(graph, pool=pool)

    @staticmethod
    def pool_handle():
        return torch.cuda.graph_pool_handle()

    @staticmethod
    def synchronize():
        torch.cuda.synchronize()

    @staticmethod
    def empty_cache():
        torch.cuda.empty_cache()

    @staticmethod
    def default_generator(index):
        return torch.cuda.default_generators[index]

    supports_generators = True

    @staticmethod
    def is_device_tensor(tensor: torch.Tensor) -> bool:
        return tensor.is_cuda


class _NpuGraphBackend:
    @staticmethod
    def new_graph():
        return torch.npu.NPUGraph()

    @staticmethod
    def graph_ctx(graph, pool):
        return torch.npu.graph(graph, pool=pool)

    @staticmethod
    def pool_handle():
        return torch.npu.graph_pool_handle()

    @staticmethod
    def synchronize():
        torch.npu.synchronize()

    @staticmethod
    def empty_cache():
        torch.npu.empty_cache()

    @staticmethod
    def default_generator(index):
        return None

    supports_generators = False

    @staticmethod
    def is_device_tensor(tensor: torch.Tensor) -> bool:
        return tensor.device.type == "npu"


def graph_signature(tree: Any, prefix: str = "") -> tuple | None:
    """Hashable signature of a nested tree for rollout graph cache keys.

    Handles Mappings (including ``BatchFeature``), lists, tuples, tensors,
    and scalar leaves. Returns None when the tree contains other objects:
    their repr may embed values that change per call, so callers must keep
    such trees on the eager path instead of keying graphs by them.
    """
    if torch.is_tensor(tree):
        return (prefix, str(tree.device), tuple(tree.shape), str(tree.dtype))
    if isinstance(tree, str) or tree is None or isinstance(tree, (bool, int, float)):
        return (prefix, type(tree).__name__, tree)
    if isinstance(tree, Mapping):
        if not all(isinstance(key, str) for key in tree):
            return None
        parts = []
        for k in sorted(tree.keys(), key=str):
            sub = graph_signature(tree[k], f"{prefix}.{k}" if prefix else str(k))
            if sub is None:
                return None
            parts.append(sub)
        return ("mapping", tuple(parts))
    if isinstance(tree, (list, tuple)):
        parts = []
        for i, v in enumerate(tree):
            sub = graph_signature(v, f"{prefix}[{i}]")
            if sub is None:
                return None
            parts.append(sub)
        return (type(tree).__name__, tuple(parts))
    return None


def clone_tensor_tree(tree: Any) -> Any:
    """Clone tensor leaves, passing non-tensor structure through unchanged."""
    if torch.is_tensor(tree):
        return tree.detach().clone()
    if isinstance(tree, Mapping):
        return {k: clone_tensor_tree(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [clone_tensor_tree(v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(clone_tensor_tree(v) for v in tree)
    return tree


def build_kv_cache(
    batch_size: int,
    num_layers: int,
    num_kv_heads: int,
    seq_length: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build KV cache buffers: list[num_layers] of (K, V), each [B, H_kv, T, D]."""
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(num_layers):
        k = torch.empty(
            (batch_size, num_kv_heads, seq_length, head_dim), dtype=dtype, device=device
        )
        v = torch.empty(
            (batch_size, num_kv_heads, seq_length, head_dim), dtype=dtype, device=device
        )
        kv_cache.append((k, v))
    return kv_cache


def copy_tensor_tree(dst: Any, src: Any, *, name: str = "") -> None:
    """Update static tensor buffers without changing their shape or dtype.

    Mapping keys absent from ``src`` retain their existing values. Sequence
    lengths and non-tensor metadata must match the captured structure.
    """
    if torch.is_tensor(dst):
        if not torch.is_tensor(src):
            raise TypeError(f"{name}: dst is Tensor but src is {type(src)}")
        if dst.shape != src.shape:
            raise ValueError(
                f"{name}: shape mismatch dst={tuple(dst.shape)} src={tuple(src.shape)}"
            )
        if dst.dtype != src.dtype:
            raise ValueError(f"{name}: dtype mismatch dst={dst.dtype} src={src.dtype}")
        dst.copy_(src)
        return

    if isinstance(dst, Mapping):
        if not isinstance(src, Mapping):
            raise TypeError(f"{name}: dst is dict but src is {type(src)}")
        for k in dst.keys():
            if k not in src:
                continue
            copy_tensor_tree(dst[k], src[k], name=f"{name}.{k}" if name else str(k))
        return

    if isinstance(dst, (list, tuple)):
        if not isinstance(src, (list, tuple)):
            raise TypeError(f"{name}: dst is {type(dst)} but src is {type(src)}")
        if len(dst) != len(src):
            raise ValueError(f"{name}: len mismatch dst={len(dst)} src={len(src)}")
        for i in range(len(dst)):
            copy_tensor_tree(dst[i], src[i], name=f"{name}[{i}]")
        return

    if dst is None or isinstance(dst, (str, bool, int, float)):
        if type(dst) is not type(src) or dst != src:
            raise ValueError(f"{name}: static metadata changed from {dst!r} to {src!r}")
        return

    raise TypeError(f"{name}: unsupported dst type {type(dst)}")


def _assert_on_device_tree(
    tree: Any, *, name: str, is_device_tensor: Callable[[torch.Tensor], bool]
) -> None:
    if torch.is_tensor(tree):
        if not is_device_tensor(tree):
            raise ValueError(
                f"{name} must be an accelerator tensor, got device={tree.device}"
            )
        return
    if isinstance(tree, dict):
        for k, v in tree.items():
            _assert_on_device_tree(
                v, name=f"{name}.{k}", is_device_tensor=is_device_tensor
            )
        return
    if isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            _assert_on_device_tree(
                v, name=f"{name}[{i}]", is_device_tensor=is_device_tensor
            )
        return
    if tree is None:
        return
    return


@dataclass
class GraphCaptureSpec:
    """Specification for capturing a CUDA graph."""

    name: str
    func: Callable[[dict[str, Any]], dict[str, Any]]
    inputs: dict[str, Any]
    external_inputs: set[str] = field(default_factory=set)
    warmup_iters: int = 3
    register_default_cuda_generator: bool = False
    cuda_generators: tuple[torch.Generator, ...] = field(default_factory=tuple)


@dataclass
class GraphSpec:
    """Specification for a captured graph."""

    name: str
    graph: Any
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    external_inputs: set[str] = field(default_factory=set)


class CUDAGraphManager:
    """
    Manager for accelerator graphs (CUDA and Ascend NPU) with support for:
    - Single graph capture and replay
    - Shared memory pool across multiple graphs (graph_pool_handle)

    Note: Buffer offload/onload is NOT supported because graphs capture
    memory addresses - buffers must remain at the same device addresses.
    For model weights offload, handle it separately in the model.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        create_graph_pool: bool = True,
    ):
        self.device = device or torch.device("cuda")
        self.backend = _graph_backend(self.device)
        self.graphs: dict[str, GraphSpec] = {}
        # Shared memory pool for all graphs to optimize memory usage
        self.graph_pool = None
        if create_graph_pool:
            self.create_shared_pool()

    def create_shared_pool(self) -> None:
        self.graph_pool = self.backend.pool_handle()

    def _register_generators(
        self,
        graph: Any,
        spec: GraphCaptureSpec,
    ) -> None:
        if not self.backend.supports_generators:
            if spec.register_default_cuda_generator or spec.cuda_generators:
                raise RuntimeError(
                    "NPU graphs do not support registering generator state; "
                    "keep sampling outside the captured region."
                )
            return

        if not hasattr(graph, "register_generator_state"):
            if spec.register_default_cuda_generator or spec.cuda_generators:
                raise RuntimeError(
                    "Current PyTorch build does not support CUDAGraph.register_generator_state."
                )
            return

        if spec.register_default_cuda_generator:
            device_index = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            graph.register_generator_state(self.backend.default_generator(device_index))

        for generator in spec.cuda_generators:
            graph.register_generator_state(generator)

    def capture(
        self,
        spec: GraphCaptureSpec,
        use_shared_pool: bool = True,
    ) -> GraphSpec:
        """
        Capture a graph from the given specification.

        Args:
            spec: The capture specification containing function, inputs, outputs.
            use_shared_pool: Whether to use the shared memory pool for this graph.

        Returns:
            The captured GraphSpec.
        """

        _assert_on_device_tree(
            spec.inputs,
            name=f"{spec.name}.inputs",
            is_device_tensor=self.backend.is_device_tensor,
        )

        for _ in range(spec.warmup_iters):
            _ = spec.func(spec.inputs)

        self.backend.synchronize()

        graph = self.backend.new_graph()
        self._register_generators(graph, spec)

        # Use shared pool if requested for memory optimization
        pool = self.graph_pool if use_shared_pool else None

        with self.backend.graph_ctx(graph, pool):
            outputs = spec.func(spec.inputs)
        _assert_on_device_tree(
            outputs,
            name=f"{spec.name}.outputs",
            is_device_tensor=self.backend.is_device_tensor,
        )

        graph_spec = GraphSpec(
            name=spec.name,
            graph=graph,
            inputs=spec.inputs,
            outputs=outputs,
            external_inputs=spec.external_inputs,
        )

        self.graphs[spec.name] = graph_spec
        return graph_spec

    def replay(
        self,
        name: str,
        inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Replay a captured CUDA graph.

        Args:
            name: Name of the graph to replay.
            inputs: Optional new inputs to copy into the static buffers before replay.

        Returns:
            The output tensors from the graph.
        """
        graph_spec = self.graphs[name]

        if graph_spec.external_inputs and inputs is None:
            raise ValueError(
                f"Graph '{name}' has external inputs {graph_spec.external_inputs} but replay() got inputs=None."
            )

        for k in graph_spec.external_inputs:
            if k not in inputs:
                continue
            if k not in graph_spec.inputs:
                raise KeyError(
                    f"Graph '{name}': external input '{k}' not found in static input buffers."
                )
            copy_tensor_tree(graph_spec.inputs[k], inputs[k], name=f"{name}.inputs.{k}")

        graph_spec.graph.replay()
        return graph_spec.outputs

    def destroy(self, name: str | None = None) -> None:
        """
        Destroy captured graphs and the shared memory pool, releasing device memory.

        Args:
            name: Name of the graph to destroy. If None, destroy all graphs.
        """
        self.backend.synchronize()
        if name is not None:
            if name in self.graphs:
                del self.graphs[name]
        else:
            self.graphs.clear()
        self.graph_pool = None

        gc.collect()
        self.backend.empty_cache()

    def __contains__(self, name: str) -> bool:
        return name in self.graphs

    def __getitem__(self, name: str) -> GraphSpec:
        return self.graphs[name]
