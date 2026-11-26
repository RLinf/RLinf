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

from collections.abc import Callable
from typing import Any

from rlinf.algorithms.prefix_off_policy.base import PrefixOffPolicyAlgorithm

_ALGORITHM_BUILDERS: dict[
    str,
    Callable[[Any], PrefixOffPolicyAlgorithm],
] = {}


def register_prefix_off_policy_algorithm(
    name: str,
) -> Callable[[type[PrefixOffPolicyAlgorithm]], type[PrefixOffPolicyAlgorithm]]:
    """Register an algorithm component by its config name."""
    normalized_name = str(name).strip().lower()
    if not normalized_name:
        raise ValueError("Prefix off-policy algorithm name must not be empty.")

    def decorator(
        algorithm_cls: type[PrefixOffPolicyAlgorithm],
    ) -> type[PrefixOffPolicyAlgorithm]:
        if normalized_name in _ALGORITHM_BUILDERS:
            raise ValueError(
                f"Prefix off-policy algorithm {normalized_name!r} is registered twice."
            )
        _ALGORITHM_BUILDERS[normalized_name] = algorithm_cls
        return algorithm_cls

    return decorator


def _load_builtin_algorithms() -> None:
    from rlinf.algorithms.prefix_off_policy import ac as _ac  # noqa: F401
    from rlinf.algorithms.prefix_off_policy import td3 as _td3  # noqa: F401


def build_prefix_off_policy_algorithm(cfg: Any) -> PrefixOffPolicyAlgorithm:
    """Build the algorithm selected by ``algorithm.name``."""
    _load_builtin_algorithms()
    name = str(cfg.algorithm.get("name", "")).strip().lower()
    if not name:
        raise ValueError("algorithm.name is required when loss_type=prefix_off_policy.")
    try:
        builder = _ALGORITHM_BUILDERS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_ALGORITHM_BUILDERS))
        raise ValueError(
            f"Unsupported prefix off-policy algorithm {name!r}; "
            f"supported algorithms: {supported}."
        ) from exc
    return builder(cfg)
