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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch


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


def _copy_tree_(dst: Any, src: Any, *, name: str = "") -> None:
    """In-place copy src -> dst for nested structures of Tensors/dicts/lists/tuples."""
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

    if isinstance(dst, dict):
        if not isinstance(src, dict):
            raise TypeError(f"{name}: dst is dict but src is {type(src)}")
        for k in dst.keys():
            if k not in src:
                continue
            _copy_tree_(dst[k], src[k], name=f"{name}.{k}" if name else str(k))
        return

    if isinstance(dst, (list, tuple)):
        if not isinstance(src, (list, tuple)):
            raise TypeError(f"{name}: dst is {type(dst)} but src is {type(src)}")
        if len(dst) != len(src):
            raise ValueError(f"{name}: len mismatch dst={len(dst)} src={len(src)}")
        for i in range(len(dst)):
            _copy_tree_(dst[i], src[i], name=f"{name}[{i}]")
        return

    if dst is None:
        return

    raise TypeError(f"{name}: unsupported dst type {type(dst)}")


def _assert_on_cuda_tree(tree: Any, *, name: str) -> None:
    if torch.is_tensor(tree):
        if not tree.is_cuda:
            raise ValueError(f"{name} must be CUDA tensor, got device={tree.device}")
        return
    if isinstance(tree, dict):
        for k, v in tree.items():
            _assert_on_cuda_tree(v, name=f"{name}.{k}")
        return
    if isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            _assert_on_cuda_tree(v, name=f"{name}[{i}]")
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
    """Specification for a captured CUDA graph."""

    name: str
    graph: torch.cuda.CUDAGraph
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    external_inputs: set[str] = field(default_factory=set)


class CUDAGraphManager:
    """
    Manager for CUDA graphs with support for:
    - Single graph capture and replay
    - Shared memory pool across multiple graphs (graph_pool_handle)

    Note: Buffer offload/onload is NOT supported because CUDA graphs
    capture memory addresses - buffers must remain at the same GPU addresses.
    For model weights offload, handle it separately in the model.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        create_graph_pool: bool = True,
    ):
        self.device = device or torch.device("cuda")
        self.graphs: dict[str, GraphSpec] = {}
        # Shared memory pool for all graphs to optimize memory usage
        self.graph_pool = None
        if create_graph_pool:
            self.create_shared_pool()

    def create_shared_pool(self) -> None:
        self.graph_pool = torch.cuda.graph_pool_handle()

    def _register_generators(
        self,
        graph: torch.cuda.CUDAGraph,
        spec: GraphCaptureSpec,
    ) -> None:
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
            graph.register_generator_state(torch.cuda.default_generators[device_index])

        for generator in spec.cuda_generators:
            graph.register_generator_state(generator)

    def capture(
        self,
        spec: GraphCaptureSpec,
        use_shared_pool: bool = True,
    ) -> GraphSpec:
        """
        Capture a CUDA graph from the given specification.

        Args:
            spec: The capture specification containing function, inputs, outputs.
            use_shared_pool: Whether to use the shared memory pool for this graph.

        Returns:
            The captured GraphSpec.
        """

        _assert_on_cuda_tree(spec.inputs, name=f"{spec.name}.inputs")

        for _ in range(spec.warmup_iters):
            _ = spec.func(spec.inputs)

        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        self._register_generators(graph, spec)

        # Use shared pool if requested for memory optimization
        pool = self.graph_pool if use_shared_pool else None

        with torch.cuda.graph(graph, pool=pool):
            outputs = spec.func(spec.inputs)
        _assert_on_cuda_tree(outputs, name=f"{spec.name}.outputs")

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
            _copy_tree_(graph_spec.inputs[k], inputs[k], name=f"{name}.inputs.{k}")

        graph_spec.graph.replay()
        return graph_spec.outputs

    def destroy(self, name: str | None = None) -> None:
        """
        Destroy captured CUDA graphs and shared memory pool, releasing GPU memory.

        Args:
            name: Name of the graph to destroy. If None, destroy all graphs.
        """
        torch.cuda.synchronize()
        if name is not None:
            if name in self.graphs:
                del self.graphs[name]
        else:
            self.graphs.clear()
        self.graph_pool = None

        gc.collect()
        torch.cuda.empty_cache()

    def __contains__(self, name: str) -> bool:
        return name in self.graphs

    def __getitem__(self, name: str) -> GraphSpec:
        return self.graphs[name]
