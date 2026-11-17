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

"""Parse quantization recipes without importing models or execution backends."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase

from rlinf.quantization.registry import get_quantization_model


@dataclass(frozen=True)
class TensorQuantizationSpec:
    format: str
    granularity: str
    axis: str
    block_size: int | None = None
    scale: str = "dynamic"


@dataclass(frozen=True)
class QuantizationSpec:
    enabled: bool = False
    weight: TensorQuantizationSpec | None = None
    activation: TensorQuantizationSpec | None = None
    rounding: str = "nearest_even"
    targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendCapability:
    backend: str
    version: str
    supported_profiles: frozenset[str]


@dataclass(frozen=True)
class TargetManifestEntry:
    path: str
    weight_key: str
    weight_shape: tuple[int, ...]
    kind: str


@dataclass(frozen=True)
class ResolvedProfile:
    spec: QuantizationSpec
    targets: tuple[str, ...]
    backend: str
    backend_version: str
    config_hash: str
    target_manifest: tuple[TargetManifestEntry, ...] = ()


_FORMATS = frozenset({"int8", "fp8", "nvfp4"})
_SCALES = frozenset({"dynamic", "static"})
_BLOCK_INT8_PROFILE = "int8_block128"
_NVFP4_PROFILE = "nvfp4"


def _assert_known_keys(value: Mapping, allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")


def _parse_tensor_spec(value: Mapping, name: str) -> TensorQuantizationSpec:
    if not isinstance(value, Mapping):
        raise ValueError(f"quantization.{name} must be a mapping")
    _assert_known_keys(
        value, {"format", "granularity", "axis", "block_size", "scale"}, name
    )
    fmt = value.get("format")
    granularity = value.get("granularity")
    axis = value.get("axis")
    block_size = value.get("block_size")
    scale = value.get("scale", "dynamic")
    if fmt not in _FORMATS:
        raise ValueError(f"unsupported {name} format: {fmt!r}")
    if granularity != "per_block":
        raise ValueError(f"unsupported {name} granularity: {granularity!r}")
    if axis not in {"input", "output", "tensor"}:
        raise ValueError(f"unsupported {name} axis: {axis!r}")
    if scale not in _SCALES:
        raise ValueError(f"unsupported {name} scale mode: {scale!r}")
    if block_size is not None and (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError(f"quantization.{name}.block_size must be a positive integer")
    if block_size is None:
        raise ValueError(f"quantization.{name}.block_size is required for per_block")
    return TensorQuantizationSpec(fmt, granularity, axis, block_size, scale)


def parse_quantization_spec(value: Mapping | None) -> QuantizationSpec:
    if value is None:
        return QuantizationSpec()
    if not isinstance(value, Mapping):
        raise ValueError("quantization must be a mapping")
    _assert_known_keys(
        value,
        {
            "enabled",
            "weight",
            "activation",
            "rounding",
            "targets",
        },
        "quantization",
    )
    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("quantization.enabled must be a boolean")
    if not enabled:
        return QuantizationSpec()
    targets = value.get("targets", ["*"])
    if (
        isinstance(targets, str)
        or not isinstance(targets, Sequence)
        or not targets
        or not all(isinstance(target, str) and target for target in targets)
    ):
        raise ValueError("qat.targets must be a non-empty list of projection selectors")
    rounding = value.get("rounding", "nearest_even")
    if rounding != "nearest_even":
        raise ValueError("unsupported quantization.rounding")
    return QuantizationSpec(
        enabled=True,
        weight=_parse_tensor_spec(value.get("weight"), "weight"),
        activation=_parse_tensor_spec(value.get("activation"), "activation"),
        rounding=rounding,
        targets=tuple(targets),
    )


def _is_block_int8_recipe(spec: QuantizationSpec) -> bool:
    expected_tensor = TensorQuantizationSpec(
        format="int8", granularity="per_block", axis="input", block_size=128
    )
    return (
        spec.weight == expected_tensor
        and spec.activation == expected_tensor
        and spec.rounding == "nearest_even"
    )


def _is_nvfp4_recipe(spec: QuantizationSpec) -> bool:
    expected = TensorQuantizationSpec("nvfp4", "per_block", "input", 16, "dynamic")
    return (
        spec.weight == expected
        and spec.activation == expected
        and spec.rounding == "nearest_even"
    )


def _recipe_id(spec: QuantizationSpec) -> str:
    if _is_block_int8_recipe(spec):
        return _BLOCK_INT8_PROFILE
    if _is_nvfp4_recipe(spec):
        return _NVFP4_PROFILE
    raise ValueError("quantization recipe is expressible but not executable")


def resolve_quantization_profile(
    spec: QuantizationSpec,
    available_targets: list[str] | tuple[str, ...],
    capability: BackendCapability,
    *,
    target_manifest: Sequence[TargetManifestEntry] | None = None,
) -> ResolvedProfile:
    """Bind actual targets to a supported backend and compute their numerical identity."""
    if not spec.enabled:
        raise ValueError("disabled quantization has no executable profile")
    recipe = _recipe_id(spec)
    if recipe not in capability.supported_profiles:
        raise ValueError("quantization recipe is not supported by backend")
    if _is_nvfp4_recipe(spec) and capability.backend != "flashinfer_cutlass":
        raise ValueError("NVFP4 requires the flashinfer_cutlass backend")
    targets = tuple(sorted(set(available_targets)))
    if not targets:
        raise ValueError("quantization selectors matched no layers")
    for selector in spec.targets:
        if not any(fnmatchcase(target, selector) for target in targets):
            raise ValueError(f"quantization selector {selector!r} matched no layers")
    selected = tuple(
        target
        for target in targets
        if any(fnmatchcase(target, selector) for selector in spec.targets)
    )
    manifest = ()
    if target_manifest is not None:
        entries = {entry.path: entry for entry in target_manifest}
        if len(entries) != len(target_manifest):
            raise ValueError("quantization target manifest contains duplicate paths")
        if any(path not in entries for path in selected):
            raise ValueError(
                "quantization target manifest does not cover selected layers"
            )
        manifest = tuple(entries[path] for path in selected)
    identity = {
        "spec": asdict(spec),
        "targets": selected,
        "recipe": recipe,
        "backend": capability.backend,
        "backend_version": capability.version,
        "target_manifest": [asdict(entry) for entry in manifest],
        "quantizer": {"qmin": -128, "qmax": 127, "scale_floor": 1e-12},
    }
    if _is_block_int8_recipe(spec):
        identity["quantizer"]["scale_arithmetic"] = (
            "fp32_amax_times_fp32_reciprocal_127"
        )
    if _is_nvfp4_recipe(spec):
        identity["quantizer"] = {
            "e2m1_values": [0, 0.5, 1, 1.5, 2, 3, 4, 6],
            "rounding": "nearest_even_code_with_independent_sign",
            "canonical_dtype": "bf16",
            "block_scale": "e4m3fn",
            "global_scale": "fp32_amax_whole_logical_projection_tensor/2688",
            "zero_global_scale": 1,
            "scale_mode": "dynamic_activation_and_weight_on_sync",
            "dequantization": "q*(block_scale*global_scale)_then_bf16",
            "gemm": "bf16_output_then_fp32_bias_then_input_dtype",
            "pack_backend": "triton",
            "gemm_backend": "cutlass",
            "packed_schema_version": 1,
            "scale_layout": "128x4",
            "data_padding": "rows_and_k_multiple_of_64",
            "scale_physical_rows": "multiple_of_128",
        }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return ResolvedProfile(
        spec=spec,
        targets=selected,
        backend=capability.backend,
        backend_version=capability.version,
        config_hash=hashlib.sha256(canonical.encode()).hexdigest(),
        target_manifest=manifest,
    )


def validate_qat_config(cfg: Mapping) -> QuantizationSpec:
    """Validate the top-level QAT settings and their execution requirements."""
    qat = cfg.get("qat")
    if qat is None:
        qat = {}
    if not isinstance(qat, Mapping):
        raise ValueError("qat must be a mapping")
    spec = parse_quantization_spec(
        {key: value for key, value in qat.items() if key != "correction"}
    )
    from rlinf.quantization.algorithms.correction import parse_correction_spec

    correction = parse_correction_spec(qat.get("correction"))
    if not spec.enabled:
        return spec
    _recipe_id(spec)

    runner = cfg.get("runner") or {}
    algorithm = cfg.get("algorithm") or {}
    actor = cfg.get("actor") or {}
    rollout = cfg.get("rollout") or {}
    _require_value(runner, "task_type", "embodied", "runner")
    _reject_enabled(runner, "only_eval", "runner")
    _require_value(runner, "weight_sync_interval", 1, "runner", default=1)
    for field in ("use_training_pipeline", "enable_async", "async", "async_rollout"):
        _reject_enabled(runner, field, "runner")
    _require_value(actor, "training_backend", "fsdp", "actor")
    _require_value(rollout, "generation_backend", "huggingface", "rollout")
    _require_value(rollout, "pipeline_stage_num", 1, "rollout", default=1)
    _require_value(algorithm, "loss_type", "decoupled_actor_critic", "algorithm")
    for scope, values in (("actor", actor), ("rollout", rollout)):
        for field in ("enable_offload", "enable_torch_compile"):
            _reject_enabled(values, field, scope)
    _reject_enabled(actor, "enable_cuda_graph", "actor")
    _reject_enabled(rollout, "enable_cuda_graph", "rollout")
    _reject_enabled(actor, "sync_weight_no_wait", "actor")
    _reject_enabled(actor.get("fsdp_config") or {}, "cpu_offload", "actor.fsdp_config")
    fsdp = actor.get("fsdp_config") or {}
    _require_value(fsdp, "strategy", "fsdp", "actor.fsdp_config", default="fsdp")
    _reject_enabled(fsdp, "mixed_precision_rules", "actor.fsdp_config")
    for dtype in ("param_dtype", "reduce_dtype", "buffer_dtype"):
        _require_value(
            fsdp.get("mixed_precision") or {},
            dtype,
            None,
            "actor.fsdp_config.mixed_precision",
        )
    _reject_enabled(
        fsdp.get("amp_autocast") or {}, "enabled", "actor.fsdp_config.amp_autocast"
    )
    for scope, values in (("algorithm", algorithm), ("rollout", rollout)):
        _reject_enabled(values, "recompute_logprobs", scope)
    syncer = cfg.get("weight_syncer") or {}
    if syncer.get("type") == "bucket":
        _require_value(
            syncer.get("bucket") or {}, "bucket_dtype", None, "weight_syncer.bucket"
        )
    elif syncer.get("type") == "patch":
        init_sync = (syncer.get("patch") or {}).get("init_sync") or {}
        _require_value(init_sync, "enabled", True, "weight_syncer.patch.init_sync")
        _require_value(init_sync, "prefixes", None, "weight_syncer.patch.init_sync")
    else:
        raise ValueError("quantization requires weight_syncer.type='bucket' or 'patch'")
    for field in ("expert_model", "rlt_feature_model"):
        if rollout.get(field) is not None:
            raise ValueError(f"quantization does not support rollout.{field}")
    if algorithm.get("adv_type") == "opd":
        raise ValueError("quantization does not support expert algorithm.adv_type=opd")

    actor_model = actor.get("model") or {}
    from omegaconf import OmegaConf

    rollout_model = OmegaConf.merge(actor_model, rollout.get("model") or {})
    model_type = actor_model.get("model_type")
    if model_type != rollout_model.get("model_type"):
        raise ValueError(
            "quantization requires actor and rollout to use the same model_type"
        )
    validator = get_quantization_model(model_type).validate_config
    contracts = []
    for scope, model in (
        ("actor.model", actor_model),
        ("rollout.model", rollout_model),
    ):
        if model.get("precision") not in (None, "null", "bf16", "bf16-mixed"):
            raise ValueError(f"quantization requires BF16 master {scope}.precision")
        _reject_enabled(model, "is_lora", scope)
        _reject_enabled(model, "enable_torch_compile", scope)
        _reject_enabled(model, "torch_compile", scope)
        contract = validator(model, correction, scope)
        if not isinstance(contract, Mapping):
            raise TypeError(
                "quantization model validator must return a contract mapping"
            )
        contracts.append(contract)
    if contracts[0] != contracts[1]:
        different = sorted(
            key
            for key in contracts[0].keys() | contracts[1].keys()
            if contracts[0].get(key) != contracts[1].get(key)
        )
        raise ValueError(
            f"quantization actor/rollout sampling or preprocessing contract mismatch: {different}"
        )
    if algorithm.get("logprob_type") not in {
        "token_level",
        "action_level",
        "chunk_level",
    }:
        raise ValueError(
            "quantization requires token_level/action_level/chunk_level algorithm.logprob_type"
        )
    return spec


def _require_value(
    values: Mapping, key: str, expected: object, scope: str, *, default: object = None
) -> None:
    actual = values.get(key, default)
    if actual != expected or (
        isinstance(expected, (bool, int)) and type(actual) is not type(expected)
    ):
        raise ValueError(
            f"quantization requires {scope}.{key}={expected!r}, found {actual!r}"
        )


def _reject_enabled(values: Mapping, key: str, scope: str) -> None:
    if values.get(key, False):
        raise ValueError(f"quantization does not support {scope}.{key}")
