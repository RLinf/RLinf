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

"""Named FSDP1 wrapper boundaries and mixed precision policies."""

from collections import defaultdict
from dataclasses import dataclass, fields
from enum import Enum
from fnmatch import fnmatchcase
from typing import Any, Callable, Mapping

import torch.nn as nn
from omegaconf import DictConfig, ListConfig
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import CustomPolicy

from rlinf.config import torch_dtype_from_precision
from rlinf.utils.logging import get_logger

_DTYPE_FIELDS = frozenset({"param_dtype", "reduce_dtype", "buffer_dtype"})
_SELECTORS = frozenset({"module_names", "module_classes", "wrap_names"})
AutoWrapPolicy = Callable[..., bool]


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


class _ModuleIndex:
    """Index original module paths, class names, and existing wrap tags once."""

    def __init__(self, model: nn.Module):
        self.by_path = dict(model.named_modules(remove_duplicate=False))
        self.paths: dict[nn.Module, list[str]] = defaultdict(list)
        self.classes: dict[str, set[str]] = defaultdict(set)
        self.tags: dict[str, set[str]] = defaultdict(set)
        for path, module in self.by_path.items():
            self.paths[module].append(path)
            if not path:
                continue
            cls = type(module)
            self.classes[f"{cls.__module__}.{cls.__qualname__}"].add(path)
            tag = getattr(module, "_fsdp_wrap_name", None)
            if isinstance(tag, str):
                self.tags[tag].add(path)

    def select(self, spec: _WrapperSpec) -> set[str]:
        """Resolve every selector entry, rejecting typos and ambiguous aliases."""
        selected: set[str] = set()
        for target in spec.targets:
            if spec.selector == "module_classes":
                matches = self.classes.get(target, set())
            elif spec.selector == "wrap_names":
                matches = self.tags.get(target, set())
            elif any(char in target for char in "*?["):
                matches = {
                    path for path in self.by_path if path and fnmatchcase(path, target)
                }
            else:
                matches = {target} if target in self.by_path and target else set()
            if not matches:
                raise ValueError(
                    f"Wrapper {spec.name!r}: selector {target!r} matched no non-root modules"
                )
            for path in matches:
                self.require_unique_path(self.by_path[path])
            selected.update(matches)
        return selected

    def require_unique_path(self, module: nn.Module) -> str:
        """Return the unique original path of a wrapper boundary."""
        paths = self.paths[module]
        if len(paths) != 1:
            raise ValueError(f"Shared module has multiple wrapper paths: {paths}")
        return paths[0]


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
        if len(selectors) != 1 or not selectors <= _SELECTORS:
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


def _contains_path(parent: str, path: str) -> bool:
    return path == parent or path.startswith(parent + ".")


def _resolve_combined_scope(spec: _WrapperSpec, selected: set[str]) -> str:
    """Find the non-root common ancestor of all selected modules."""
    ancestor = min(selected).split(".")
    while ancestor and not all(
        _contains_path(".".join(ancestor), path) for path in selected
    ):
        ancestor.pop()
    if not ancestor:
        raise ValueError(
            f"Wrapper {spec.name!r}: group resolves to the model root; select a non-root execution boundary"
        )
    scope_path = ".".join(ancestor)
    return scope_path


def _validate_group_scope(
    index: _ModuleIndex,
    spec: _WrapperSpec,
    scope_path: str,
    selected: set[str],
    use_orig_params: bool,
    ignored_classes: tuple[type[nn.Module], ...],
) -> None:
    """Validate full subtree ownership before suppressing inner boundaries."""
    scope = index.by_path[scope_path]
    covered = {child for path in selected for child in index.by_path[path].modules()}
    for relative_path, module in scope.named_modules():
        path = f"{scope_path}.{relative_path}" if relative_path else scope_path
        has_state = (
            next(module.parameters(recurse=False), None) is not None
            or next(module.buffers(recurse=False), None) is not None
        )
        if has_state and module not in covered:
            raise ValueError(
                f"Wrapper {spec.name!r}: grouping would include unselected states at {path!r}; select the parent explicitly or include this module"
            )
        if isinstance(module, ignored_classes):
            raise ValueError(
                f"Wrapper {spec.name!r}: BatchNorm exclusion would split the requested group"
            )
    params = list(scope.parameters())
    if not params:
        raise ValueError(f"Wrapper {spec.name!r}: group has no parameters")
    if len({param.dtype for param in params}) != 1:
        raise ValueError(
            f"Wrapper {spec.name!r}: FSDP1 groups require uniform original parameter dtype"
        )
    if not use_orig_params and len({param.requires_grad for param in params}) != 1:
        raise ValueError(
            f"Wrapper {spec.name!r}: mixed requires_grad needs use_orig_params=true"
        )


def _resolve_boundaries(
    index: _ModuleIndex,
    specs: tuple[_WrapperSpec, ...],
    use_orig_params: bool,
    ignored_classes: tuple[type[nn.Module], ...],
) -> dict[nn.Module, _WrapperSpec]:
    """Resolve explicit boundaries and reject contradictory definitions."""
    boundaries: dict[nn.Module, _WrapperSpec] = {}
    for spec in specs:
        selected = index.select(spec)
        if spec.wrap_mode is _WrapMode.COMBINED:
            scope = _resolve_combined_scope(spec, selected)
            _validate_group_scope(
                index, spec, scope, selected, use_orig_params, ignored_classes
            )
            selected = {scope}
        for path in sorted(selected):
            if spec.wrap_mode is _WrapMode.SUBTREE:
                _validate_group_scope(
                    index, spec, path, {path}, use_orig_params, ignored_classes
                )
            module = index.by_path[path]
            index.require_unique_path(module)
            if type(module).forward is nn.Module.forward:
                raise ValueError(
                    f"Module {path!r} has no forward execution boundary; select its executable parent or children"
                )
            if module in boundaries:
                raise ValueError(f"Multiple wrapper definitions match module {path!r}")
            if isinstance(module, ignored_classes):
                raise ValueError(
                    f"Module {path!r} is subject to PyTorch's BatchNorm exclusion; explicit wrappers are unsupported"
                )
            boundaries[module] = spec
    for module, spec in boundaries.items():
        if spec.wrap_mode is not _WrapMode.INDIVIDUAL and any(
            child in boundaries for child in module.modules() if child is not module
        ):
            raise ValueError(
                f"Grouped wrapper {spec.name!r} contains explicitly selected inner wrappers"
            )
    return boundaries


def _collect_wrapped_modules(
    model: nn.Module,
    boundaries: Mapping[nn.Module, _WrapperSpec],
    base_policy: AutoWrapPolicy | None,
) -> set[nn.Module]:
    """Apply the base policy bottom-up, stopping traversal at grouped boundaries."""
    wrapped: set[nn.Module] = set()

    def visit(module: nn.Module) -> int:
        numel = sum(param.numel() for param in module.parameters())
        spec = boundaries.get(module)
        if spec is not None and spec.wrap_mode is not _WrapMode.INDIVIDUAL:
            wrapped.add(module)
            return numel
        if base_policy is not None and not base_policy(
            module=module, recurse=True, nonwrapped_numel=numel
        ):
            if spec is not None:
                wrapped.add(module)
                return numel
            return 0
        child_numel = sum(visit(child) for child in module.children())
        should_wrap = spec is not None or (
            base_policy is not None
            and base_policy(
                module=module, recurse=False, nonwrapped_numel=numel - child_numel
            )
        )
        if module is not model and should_wrap:
            wrapped.add(module)
            return numel
        return child_numel

    visit(model)
    unreachable = set(boundaries) - wrapped
    if unreachable:
        names = sorted({boundaries[module].name for module in unreachable})
        raise ValueError(f"Explicit wrappers {names} are below a non-recursing policy")
    return wrapped


def _validate_parameter_owners(
    model: nn.Module,
    wrapped: set[nn.Module],
    boundaries: Mapping[nn.Module, _WrapperSpec],
    ignored_classes: tuple[type[nn.Module], ...],
    use_orig_params: bool,
) -> None:
    """Validate the parameters each final wrapper owns, excluding child wrappers."""
    owners: dict[nn.Parameter, nn.Module] = {}
    params_by_owner: dict[nn.Module, set[nn.Parameter]] = defaultdict(set)

    def visit(module: nn.Module, owner: nn.Module) -> None:
        if module in wrapped or isinstance(module, ignored_classes):
            owner = module
        for param in module.parameters(recurse=False):
            params_by_owner[owner].add(param)
            previous = owners.setdefault(param, owner)
            if previous is not owner and (
                previous in boundaries or owner in boundaries
            ):
                raise ValueError(
                    "Shared parameter crosses an explicit wrapper boundary"
                )
        for child in module.children():
            visit(child, owner)

    visit(model, model)
    for owner, params in params_by_owner.items():
        spec = boundaries.get(owner)
        name = spec.name if spec is not None else type(owner).__name__
        if len({param.dtype for param in params}) != 1:
            raise ValueError(
                f"FSDP wrapper {name!r} requires uniform original parameter dtype"
            )
        if not use_orig_params and len({param.requires_grad for param in params}) != 1:
            raise ValueError(
                f"FSDP wrapper {name!r}: mixed requires_grad needs use_orig_params=true"
            )


def get_mixed_precision_wrap_policy(
    model: nn.Module,
    config: DictConfig,
    base_policy: AutoWrapPolicy | None,
    default_mp: MixedPrecision,
) -> AutoWrapPolicy | CustomPolicy | None:
    """Combine named wrapper overrides with the existing automatic wrap policy.

    Individual matches retain automatic child wrappers. Subtree and combined
    boundaries own their entire subtree. Resolve this plan before FSDP mutates it.
    """
    specs = _parse_wrapper_config(config)
    if not specs:
        return base_policy
    index = _ModuleIndex(model)
    # PyTorch adds these boundaries itself and discards their MP overrides.
    ignored_classes = tuple(default_mp._module_classes_to_ignore)
    boundaries = _resolve_boundaries(
        index, specs, config.get("use_orig_params", False), ignored_classes
    )
    wrapped = _collect_wrapped_modules(model, boundaries, base_policy)
    for module in wrapped:
        index.require_unique_path(module)
    _validate_parameter_owners(
        model,
        wrapped,
        boundaries,
        ignored_classes,
        config.get("use_orig_params", False),
    )
    for module, spec in boundaries.items():
        get_logger().info(
            f"[FSDP] Wrapper {spec.name}: scope={index.require_unique_path(module)}, wrap_mode={spec.wrap_mode.value}, policy={spec.policy_name or 'default'}"
        )

    def policy(module: nn.Module) -> bool | dict[str, MixedPrecision]:
        spec = boundaries.get(module)
        if spec is not None and spec.mixed_precision is not None:
            return {"mixed_precision": spec.mixed_precision}
        return module in wrapped

    return CustomPolicy(policy)
