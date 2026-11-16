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

"""FSDP1 wrapper configuration and precision policy schema."""

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Mapping

from omegaconf import DictConfig, ListConfig
from torch.distributed.fsdp import MixedPrecision

from rlinf.utils.precision import torch_dtype_from_precision

_DTYPE_FIELDS = frozenset({"param_dtype", "reduce_dtype", "buffer_dtype"})
_SELECTORS = frozenset({"module_names", "module_classes", "wrap_names"})


class _WrapMode(str, Enum):
    INDIVIDUAL = "individual"
    SUBTREE = "subtree"
    COMBINED = "combined"


@dataclass(frozen=True)
class _WrapperSpec:
    """A validated selector with its optional precision override."""

    name: str
    selector: str
    targets: tuple[str, ...]
    wrap_mode: _WrapMode
    policy_name: str | None
    mixed_precision: MixedPrecision | None


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, (DictConfig, Mapping)):
        raise ValueError(f"{name} must be a mapping")
    return value


def build_mixed_precision(config: Mapping[str, Any]) -> MixedPrecision:
    """Build precision settings using the installed PyTorch's public fields."""
    config = _require_mapping(config, "mixed_precision")
    supported = {
        field.name for field in fields(MixedPrecision) if not field.name.startswith("_")
    }
    unknown = set(config) - supported
    if unknown:
        raise ValueError(f"Unsupported FSDP1 MixedPrecision options: {sorted(unknown)}")
    kwargs = dict(config)
    kwargs.setdefault("cast_forward_inputs", True)
    for name in _DTYPE_FIELDS & kwargs.keys():
        kwargs[name] = torch_dtype_from_precision(kwargs[name])
    for name in (supported - _DTYPE_FIELDS) & kwargs.keys():
        if not isinstance(kwargs[name], bool):
            raise ValueError(f"MixedPrecision.{name} must be a boolean")
    return MixedPrecision(**kwargs)


def _parse_wrapper_config(config: DictConfig) -> tuple[_WrapperSpec, ...]:
    wrappers = _require_mapping(config.get("wrappers", {}), "wrappers")
    policies = _require_mapping(
        config.get("mixed_precision_policies", {}), "mixed_precision_policies"
    )
    rules = _require_mapping(
        config.get("mixed_precision_rules", {}), "mixed_precision_rules"
    )
    if not wrappers and not policies and not rules:
        return ()
    if config.get("strategy", "fsdp") != "fsdp":
        raise ValueError("Named wrappers require strategy='fsdp' (FSDP1)")
    if config.get("disable", False):
        raise ValueError("Named wrappers conflict with disable=true")
    if rules and config.get("amp_autocast", {}).get("enabled", False):
        raise ValueError("mixed_precision_rules requires amp_autocast.enabled=false")

    precision: dict[str, MixedPrecision] = {}
    for name, policy in policies.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Mixed precision policy names must be non-empty strings")
        policy = _require_mapping(policy, f"Policy {name!r}")
        missing = _DTYPE_FIELDS - policy.keys()
        if missing:
            raise ValueError(
                f"Policy {name!r} is missing dtype fields: {sorted(missing)}"
            )
        precision[name] = build_mixed_precision(policy)
    for name, policy_name in rules.items():
        if name not in wrappers:
            raise ValueError(f"Unknown wrapper {name!r} in mixed_precision_rules")
        if not isinstance(policy_name, str) or policy_name not in precision:
            raise ValueError(f"Wrapper {name!r}: unknown policy {policy_name!r}")

    specs: list[_WrapperSpec] = []
    for name, wrapper in wrappers.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Wrapper names must be non-empty strings")
        wrapper = _require_mapping(wrapper, f"Wrapper {name!r}")
        unknown = set(wrapper) - _SELECTORS - {"wrap_mode"}
        if unknown:
            raise ValueError(f"Wrapper {name!r}: unsupported options {sorted(unknown)}")
        selectors = set(wrapper) & _SELECTORS
        if len(selectors) != 1:
            raise ValueError(f"Wrapper {name!r} requires exactly one module selector")
        selector = next(iter(selectors))
        targets = wrapper[selector]
        if (
            not isinstance(targets, (list, ListConfig))
            or not targets
            or any(not isinstance(target, str) or not target for target in targets)
        ):
            raise ValueError(
                f"Wrapper {name!r}: selector must be a non-empty string list"
            )
        mode = wrapper.get("wrap_mode", "individual")
        try:
            wrap_mode = _WrapMode(mode)
        except ValueError as error:
            raise ValueError(
                f"Wrapper {name!r}: wrap_mode must be individual, subtree, or combined, got {mode!r}"
            ) from error
        policy_name = rules.get(name)
        specs.append(
            _WrapperSpec(
                name=name,
                selector=selector,
                targets=tuple(targets),
                wrap_mode=wrap_mode,
                policy_name=policy_name,
                mixed_precision=precision.get(policy_name),
            )
        )
    return tuple(specs)


def validate_fsdp_wrapper_config(config: DictConfig) -> None:
    """Validate wrapper definitions and policy references before loading a model."""
    _parse_wrapper_config(config)
