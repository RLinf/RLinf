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

"""Validate V2 checkpoints against the constructed policy and dormant heads."""

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

from safetensors import safe_open
from safetensors.torch import load_model
from torch import nn
from transformers.modeling_utils import load_sharded_checkpoint


def _checkpoint_shapes(weights: Path) -> dict[str, tuple[int, ...]]:
    index_path = weights / "model.safetensors.index.json"
    weight_map = (
        json.loads(index_path.read_text())["weight_map"]
        if index_path.is_file()
        else None
    )
    if weight_map is not None:
        shard_keys: dict[str, set[str]] = {}
        for key, shard in weight_map.items():
            shard_keys.setdefault(shard, set()).add(key)
        shards = set(shard_keys)
    else:
        shard_keys = {}
        shards = {"model.safetensors"}

    shapes: dict[str, tuple[int, ...]] = {}
    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(
                safe_open(str(weights / shard), framework="pt", device="cpu")
            )
            for shard in shards
        }
        for shard, handle in handles.items():
            actual_keys = set(handle.keys())
            expected_keys = shard_keys[shard] if weight_map is not None else actual_keys
            if actual_keys != expected_keys:
                raise RuntimeError(
                    f"Safetensors index does not match shard {shard}: "
                    f"missing={sorted(expected_keys - actual_keys)} "
                    f"unindexed={sorted(actual_keys - expected_keys)}"
                )
            for key in actual_keys:
                shapes[key] = tuple(handle.get_slice(key).get_shape())
    return shapes


def dormant_alignment_state_shapes(
    alignment_model: nn.Module, action_model: nn.Module
) -> dict[str, tuple[int, ...]]:
    """Return state entries created only when upstream alignment heads are enabled."""
    action_keys = set(action_model.state_dict())
    return {
        name: tuple(tensor.shape)
        for name, tensor in alignment_model.state_dict().items()
        if name not in action_keys
    }


def load_vla_checkpoint(
    model: nn.Module,
    weights: Path,
    enable_alignment_heads: bool,
    alignment_model_config: Any,
    build_alignment_model: Callable[[Any], nn.Module],
) -> None:
    """Load a V2 checkpoint after exact key and shape validation."""
    checkpoint_shapes = _checkpoint_shapes(weights)
    model_shapes = {
        key: tuple(value.shape) for key, value in model.state_dict().items()
    }
    model_keys = set(model_shapes)
    checkpoint_keys = set(checkpoint_shapes)
    missing = model_keys - checkpoint_keys
    extra = checkpoint_keys - model_keys
    shape_mismatches = {
        key: (model_shapes[key], checkpoint_shapes[key])
        for key in model_keys & checkpoint_keys
        if model_shapes[key] != checkpoint_shapes[key]
    }
    if missing or shape_mismatches:
        raise RuntimeError(
            "Checkpoint does not match the constructed V2 policy: "
            f"missing={sorted(missing)} shape_mismatches={shape_mismatches}"
        )

    if extra:
        if enable_alignment_heads:
            raise RuntimeError(f"Unexpected checkpoint keys: {sorted(extra)}")
        alignment_model = build_alignment_model(alignment_model_config)
        allowed_shapes = dormant_alignment_state_shapes(alignment_model, model)
        del alignment_model
        if extra != set(allowed_shapes):
            raise RuntimeError(
                "Unexpected checkpoint keys outside the constructed alignment modules: "
                f"extra={sorted(extra)} allowed={sorted(allowed_shapes)}"
            )
        extra_shape_mismatches = {
            key: (allowed_shapes[key], checkpoint_shapes[key])
            for key in extra
            if allowed_shapes[key] != checkpoint_shapes[key]
        }
        if extra_shape_mismatches:
            raise RuntimeError(
                f"Dormant alignment checkpoint shape mismatch: {extra_shape_mismatches}"
            )

    index_path = weights / "model.safetensors.index.json"
    if index_path.is_file():
        result = load_sharded_checkpoint(
            model, str(weights), strict=not bool(extra), prefer_safe=True
        )
        loaded_missing = set(result.missing_keys)
        loaded_extra = set(result.unexpected_keys)
    else:
        loaded_missing, loaded_extra = load_model(
            model, str(weights / "model.safetensors"), strict=not bool(extra)
        )
        loaded_missing = set(loaded_missing)
        loaded_extra = set(loaded_extra)
    if loaded_missing or loaded_extra != extra:
        raise RuntimeError(
            "Checkpoint load integrity failure: "
            f"missing={sorted(loaded_missing)} unexpected={sorted(loaded_extra)} "
            f"expected_extra={sorted(extra)}"
        )
