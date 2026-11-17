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

"""Explicit pi0.5 target discovery and adaptation over model-independent numerics."""

from __future__ import annotations

from torch import nn

from rlinf.models.embodiment.openpi.modules.gemma import FeedForward
from rlinf.quantization.backends.base import QuantizationBackend
from rlinf.quantization.config import TargetManifestEntry
from rlinf.quantization.layers.linear import FakeQuantLinear, QuantizedLinear
from rlinf.quantization.models.openpi.actor import (
    FakeQuantFeedForward,
    FakeQuantPatchConv2d,
    FakeQuantVisionMultiheadAttention,
)
from rlinf.quantization.models.openpi.rollout import (
    QuantizedFeedForward,
    QuantizedPatchConv2d,
    QuantizedVisionMultiheadAttention,
)
from rlinf.quantization.registry import QuantizationAdapter

_ACTION_PROJECTIONS = frozenset(
    {
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
    }
)

_ACTOR_PROJECTIONS = (
    FakeQuantLinear,
    FakeQuantPatchConv2d,
    FakeQuantFeedForward,
    FakeQuantVisionMultiheadAttention,
)


_ROLLOUT_PROJECTIONS = (
    QuantizedLinear,
    QuantizedPatchConv2d,
    QuantizedFeedForward,
    QuantizedVisionMultiheadAttention,
)


def discover_pi05_targets(model: nn.Module) -> tuple[TargetManifestEntry, ...]:
    """List candidate projections, including paths outside nn.Linear.forward."""
    modules = dict(model.named_modules())
    required = {"llm", "img", "img.stem", "img.head", *_ACTION_PROJECTIONS}
    missing = sorted(required - modules.keys())
    if missing:
        raise ValueError(f"π0.5 target modules are missing: {missing}")
    if not isinstance(modules["img.stem"], nn.Conv2d):
        raise ValueError("π0.5 img.stem must be a Conv2d patch embedding")
    if getattr(modules["img"], "pool_type", "none") != "none":
        raise ValueError("π0.5 QAT vision adapter requires pool_type='none'")

    targets: list[TargetManifestEntry] = []
    for name, module in modules.items():
        if name == "img.stem":
            targets.append(
                TargetManifestEntry(
                    path=name,
                    weight_key=f"{name}.weight",
                    weight_shape=tuple(module.weight.shape),
                    kind="patch_conv",
                )
            )
        elif isinstance(module, nn.MultiheadAttention) and name.startswith("img."):
            if module.in_proj_weight is None:
                raise ValueError(f"π0.5 {name} has no packed in_proj_weight")
            targets.extend(
                (
                    TargetManifestEntry(
                        path=f"{name}.in_proj_weight",
                        weight_key=f"{name}.in_proj_weight",
                        weight_shape=tuple(module.in_proj_weight.shape),
                        kind="mha_qkv",
                    ),
                    TargetManifestEntry(
                        path=f"{name}.out_proj",
                        weight_key=f"{name}.out_proj.weight",
                        weight_shape=tuple(module.out_proj.weight.shape),
                        kind="mha_out_proj",
                    ),
                )
            )
        elif isinstance(module, FeedForward) and name.startswith("llm."):
            targets.extend(
                (
                    TargetManifestEntry(
                        path=f"{name}.w_gating",
                        weight_key=f"{name}.w_gating",
                        weight_shape=tuple(module.w_gating.shape),
                        kind="functional_gating",
                    ),
                    TargetManifestEntry(
                        path=f"{name}.w_linear",
                        weight_key=f"{name}.w_linear",
                        weight_shape=tuple(module.w_linear.shape),
                        kind="functional_down",
                    ),
                )
            )
        elif isinstance(module, nn.Linear) and (
            name.startswith("llm.")
            or name.startswith("img.")
            or name in _ACTION_PROJECTIONS
        ):
            # MultiheadAttention calls its out_proj parameters functionally;
            # replacing that child module alone would not quantize its forward.
            parent_path, _, _ = name.rpartition(".")
            if isinstance(modules.get(parent_path), nn.MultiheadAttention):
                continue
            targets.append(
                TargetManifestEntry(
                    path=name,
                    weight_key=f"{name}.weight",
                    weight_shape=tuple(module.weight.shape),
                    kind="linear",
                )
            )
    required_kinds = {
        "linear",
        "patch_conv",
        "mha_qkv",
        "mha_out_proj",
        "functional_gating",
        "functional_down",
    }
    missing_kinds = sorted(required_kinds - {target.kind for target in targets})
    if missing_kinds:
        raise ValueError(f"π0.5 inventory is missing target kinds: {missing_kinds}")
    return tuple(sorted(targets, key=lambda target: target.path))


def apply_pi05_actor_quantization(
    model: nn.Module,
    targets: tuple[TargetManifestEntry, ...],
    *,
    backend: QuantizationBackend,
) -> tuple[TargetManifestEntry, ...]:
    """Install fake-quantized projections from the resolved target manifest."""
    return _apply_pi05_quantization(
        model,
        targets,
        _ACTOR_PROJECTIONS,
        backend=backend,
    )


def apply_pi05_rollout_quantization(
    model: nn.Module,
    targets: tuple[TargetManifestEntry, ...],
    *,
    backend: QuantizationBackend,
) -> tuple[TargetManifestEntry, ...]:
    """Install true low-bit rollout projections from the resolved target manifest."""
    return _apply_pi05_quantization(
        model,
        targets,
        _ROLLOUT_PROJECTIONS,
        backend=backend,
    )


def load_pi05_rollout_weights(model: nn.Module) -> None:
    """Repack rollout projections; shared runtime owns publication and rollback."""
    for module in model.modules():
        if isinstance(module, _ROLLOUT_PROJECTIONS):
            module.load_weights()


def _build_projection(original, kinds, projection_types, options) -> nn.Module:
    linear, patch, feed, attention = projection_types
    if kinds == {"linear"}:
        return linear.from_linear(original, **options)
    if kinds == {"patch_conv"}:
        return patch.from_conv2d(original, **options)
    if kinds <= {"functional_gating", "functional_down"}:
        return feed.from_feed_forward(
            original,
            quantize_gating="functional_gating" in kinds,
            quantize_down="functional_down" in kinds,
            **options,
        )
    if kinds <= {"mha_qkv", "mha_out_proj"}:
        return attention.from_mha(
            original,
            quantize_qkv="mha_qkv" in kinds,
            quantize_out="mha_out_proj" in kinds,
            **options,
        )
    raise NotImplementedError(
        f"pi0.5 projection kinds are not implemented: {sorted(kinds)}"
    )


def _target_owner_path(target: TargetManifestEntry) -> str:
    """Return the module path replaced for one weight-level target."""
    if target.kind in {"linear", "patch_conv"}:
        return target.path
    return target.path.rpartition(".")[0]


def _apply_pi05_quantization(
    model: nn.Module,
    targets: tuple[TargetManifestEntry, ...],
    projection_types: tuple[type[nn.Module], ...],
    **options,
) -> tuple[TargetManifestEntry, ...]:
    """Consume a fresh manifest and construct replacements before mutating the model."""
    groups: dict[str, set[str]] = {}
    for target in targets:
        owner_path = _target_owner_path(target)
        if owner_path not in groups:
            groups[owner_path] = set()
        groups[owner_path].add(target.kind)
    replacements = []
    for path, kinds in groups.items():
        original = model.get_submodule(path)
        replacement = _build_projection(original, kinds, projection_types, options)
        parent_path, _, child_name = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        replacements.append((parent, child_name, replacement))
    for parent, child_name, replacement in replacements:
        setattr(parent, child_name, replacement)
    if hasattr(model, "_mark_fsdp_wrap_names"):
        model._mark_fsdp_wrap_names()
    return targets


def build_adapter() -> QuantizationAdapter:
    """Build the OpenPI adapter after CPU configuration validation."""
    return QuantizationAdapter(
        discover=discover_pi05_targets,
        apply_actor=apply_pi05_actor_quantization,
        apply_rollout=apply_pi05_rollout_quantization,
        load_rollout_weights=load_pi05_rollout_weights,
    )
