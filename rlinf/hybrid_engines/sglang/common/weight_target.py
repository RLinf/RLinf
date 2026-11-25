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

"""Validate and map actor buckets before loading a SGLang serving module."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rlinf.scheduler import Worker


def receive_hf_weights(
    worker: Worker,
    actor_group_name: str,
    actor_rank: int,
    load_bucket: Callable[[dict], None],
) -> int | None:
    """Receive named actor weight buckets and load them in order.

    Prefetch the next bucket while loading the current one.
    Returns the policy version when supplied by the actor. Legacy single-bucket
    senders without ``bucket_length`` remain supported.
    """
    bucket = worker.recv(src_group_name=actor_group_name, src_rank=actor_rank)
    count = int(bucket.pop("bucket_length", 1))
    version = bucket.pop("weight_version", None)
    if count < 1:
        raise ValueError(f"Expected positive weight bucket count, got {count}")
    for index in range(count):
        pending = (
            worker.recv(
                src_group_name=actor_group_name, src_rank=actor_rank, async_op=True
            )
            if index + 1 < count
            else None
        )
        load_bucket(bucket)
        bucket = pending.wait() if pending is not None else None
    return int(version) if version is not None else None


class SGLangWeightTarget(torch.nn.Module):
    """Apply actor buckets to a serving module using a model-provided name map.

    ``name_mapper`` returns a module-local serving key, or None for actor state
    that is not served. ``checkpoint_dir`` contains that module's safetensors.
    Validation is shared; model adapters own the mapping.
    """

    def __init__(
        self,
        load_weights: Callable[[str, list[tuple[str, torch.Tensor]]], None],
        *,
        checkpoint_dir: str | Path,
        target_module: str,
        name_mapper: Callable[[str], str | None],
    ) -> None:
        super().__init__()
        from safetensors import safe_open

        self.load_weights = load_weights
        self.target_module = target_module
        self.name_mapper = name_mapper
        self.shapes: dict[str, tuple[int, ...]] = {}
        self._full_sync = False
        self._received_names: set[str] = set()
        self._patch_state: dict[str, torch.Tensor] | None = None
        for path in sorted(Path(checkpoint_dir).glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as reader:
                for name in reader.keys():
                    if name in self.shapes:
                        raise ValueError(f"Duplicate serving checkpoint tensor: {name}")
                    self.shapes[name] = tuple(reader.get_slice(name).get_shape())
        if not self.shapes:
            raise ValueError(f"No serving safetensors found in {checkpoint_dir}")

    def bind_patch_model(self, module: torch.nn.Module) -> None:
        """Expose checkpoint-named views into resident, unsharded serving weights.

        Fused linear projections are exposed as nonoverlapping slices, so sparse
        updates preserve serving storage and do not require a second model copy.
        """
        from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
        from torch.distributed.tensor import DTensor

        mappers = [
            get_param_names_mapping(mapping)
            for attr in ("lora_param_names_mapping", "param_names_mapping")
            if (mapping := getattr(module, attr, None))
        ]
        parameters = dict(module.named_parameters())
        parameters.update(module.named_buffers())
        views = {}
        for name, shape in self.shapes.items():
            mapped, shard = name, None
            if name not in parameters:
                for mapper in mappers:
                    mapped, index, _ = mapper(mapped)
                    if index is not None:
                        shard = index
            # SGLang maps checkpoint-only, unused weights to an empty name.
            if not mapped:
                continue
            if mapped not in parameters:
                raise ValueError(
                    f"No live serving parameter for patch key {name} -> {mapped}"
                )
            parameter = parameters[mapped]
            if isinstance(parameter, DTensor) or parameter.device.type == "meta":
                raise ValueError(
                    f"Patch target must be resident and unsharded: {mapped}"
                )
            value = parameter.detach()
            if shard is not None:
                owner = module.get_submodule(mapped.rsplit(".", 1)[0])
                sizes = getattr(owner, "output_sizes", None)
                dim = getattr(parameter, "output_dim", None)
                if sizes is None or dim is None or not isinstance(shard, int):
                    raise ValueError(
                        f"Unsupported fused patch parameter: {name} -> {mapped}"
                    )
                value = value.narrow(dim, sum(sizes[:shard]), sizes[shard])
            if tuple(value.shape) != shape or not value.is_contiguous():
                raise ValueError(
                    f"Patch target {name} requires a contiguous full-shape view "
                    f"{shape}, got {tuple(value.shape)}"
                )
            views[name] = value
        if not views:
            raise ValueError("No serving parameters are available for patch sync")
        self._patch_state = views

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Return writable serving views for PatchWeightSyncer."""
        if self._patch_state is None:
            raise RuntimeError("Bind a serving module before initializing patch sync")
        if destination is None:
            destination = OrderedDict()
        for name, value in self._patch_state.items():
            destination[prefix + name] = value if keep_vars else value.detach()
        return destination

    def begin_sync(self, *, full: bool) -> None:
        """Start a transfer; full transfers must cover the entire serving module."""
        self._full_sync = full
        self._received_names.clear()

    def finish_sync(self) -> None:
        """Validate coverage before the worker makes the new weights available."""
        if self._full_sync and self._received_names != self.shapes.keys():
            missing = self.shapes.keys() - self._received_names
            raise ValueError(
                f"Full {self.target_module} sync is missing {len(missing)} weights: {sorted(missing)[:8]}"
            )
        self._received_names.clear()
        self._full_sync = False

    def load_checkpoint(self, checkpoint: str, bucket_size: int = 536870912) -> None:
        """Load the complete serving state from an RLinf checkpoint for evaluation."""
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        selected = {
            name: tensor
            for name, tensor in state.items()
            if self.name_mapper(name) is not None
        }
        mapped = [self.name_mapper(name) for name in selected]
        if set(mapped) != self.shapes.keys() or len(mapped) != len(set(mapped)):
            raise ValueError(
                f"runner.ckpt_path must contain the complete {self.target_module} serving state with unique mapped names"
            )
        self.begin_sync(full=True)
        bucket, size = {}, 0
        for key, value in selected.items():
            bucket[key] = value
            size += value.numel() * value.element_size()
            if size >= bucket_size:
                self.load_state_dict(bucket)
                bucket, size = {}, 0
        if bucket:
            self.load_state_dict(bucket)
        self.finish_sync()

    def load_state_dict(
        self, state_dict: Mapping[str, torch.Tensor], strict=False, assign=False
    ) -> None:
        """Validate and load one bucket; propagate partial loading failures."""
        del strict, assign
        named = []
        names = set()
        for name, tensor in state_dict.items():
            key = self.name_mapper(name)
            if key is None or key not in self.shapes:
                raise ValueError(
                    f"Actor tensor has no serving counterpart: {name} -> {key}"
                )
            if key in names or (self._full_sync and key in self._received_names):
                raise ValueError(f"Duplicate serving tensor in weight sync: {key}")
            names.add(key)
            shape = self.shapes[key]
            if shape == () and tensor.shape == (1,):
                tensor = tensor.reshape(())
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"Serving weight shape mismatch for {key}: {tuple(tensor.shape)} != {shape}"
                )
            named.append((key, tensor.detach()))
        if not named:
            raise ValueError("Received an empty serving weight bucket")
        self.load_weights(self.target_module, named)
        self._received_names.update(names)
