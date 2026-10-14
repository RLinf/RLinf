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

"""Registry for embodied sglang adapter classes.

An sglang adapter turns an RLinf env observation batch into model action chunks
over a launched ``sglang serve`` HTTP server. It is registered per
``cfg.model_type`` with a *lazy* builder (mirroring
``rlinf.models._register_builtin_models``) so importing this module never
force-imports a model's heavy deps; the builder runs only on lookup.

An adapter whose server cannot batch every env together also defines
``request_groups(env_obs)``; the worker then sends one request per group.
"""

from typing import Any, Callable

import torch

from rlinf.config import SupportedModel
from rlinf.utils.obs_compression import infer_obs_batch_size

_SGLANG_ADAPTER_REGISTRY: dict[str, Callable[[], type]] = {}


def select_env_rows(env_obs: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    """Select the envs at ``indices`` from an env observation batch.

    Args:
        env_obs: An env observation batch as passed to ``build_request``.
        indices: Env indices to keep.

    Returns:
        The observation with per-env tensors and lists indexed; other values kept.
    """
    batch_size = infer_obs_batch_size(env_obs)
    selected: dict[str, Any] = {}
    for key, value in env_obs.items():
        if (
            isinstance(value, torch.Tensor)
            and value.dim() > 0
            and value.shape[0] == batch_size
        ):
            index = torch.as_tensor(indices, dtype=torch.long, device=value.device)
            selected[key] = value.index_select(0, index)
        elif isinstance(value, list) and len(value) == batch_size:
            selected[key] = [value[i] for i in indices]
        else:
            selected[key] = value
    return selected


def gather_env_rows(parts: list[Any], order: list[int]) -> Any:
    """Concatenate per-group outputs and restore env order.

    Args:
        parts: One tensor, or (nested) dict of tensors, per request group.
        order: The env index of each row of the concatenated parts.

    Returns:
        The concatenated output with row ``i`` belonging to env ``i``.
    """
    first = parts[0]
    if isinstance(first, torch.Tensor):
        inverse = torch.argsort(torch.as_tensor(order, device=first.device))
        return torch.cat(parts, dim=0).index_select(0, inverse)
    if isinstance(first, dict):
        return {key: gather_env_rows([p[key] for p in parts], order) for key in first}
    return first


def register_sglang_adapter(
    model_type: str,
    builder: Callable[[], type],
    force: bool = False,
):
    """Register a lazy sglang adapter builder for ``cfg.model_type``.

    ``builder`` is a zero-arg callable returning the adapter class; keep the
    heavy import inside it so it runs only on lookup. Lookup happens via
    :func:`get_sglang_adapter_cls`.
    """
    if not model_type:
        raise ValueError("model_type must be a non-empty string.")
    key = model_type.lower()
    if not force and key in _SGLANG_ADAPTER_REGISTRY:
        raise ValueError(
            f"SGLang adapter `{key}` is already registered. "
            "Set force=True to override it."
        )
    _SGLANG_ADAPTER_REGISTRY[key] = builder


def _register_builtin_sglang_adapters():
    def _build_dreamzero_sglang_adapter():
        from rlinf.models.embodiment.dreamzero.sglang_adapter import (
            DreamZeroSGLangAdapter,
        )

        return DreamZeroSGLangAdapter

    register_sglang_adapter(
        SupportedModel.DREAMZERO.value,
        _build_dreamzero_sglang_adapter,
        force=True,
    )

    def _build_cosmos3_sglang_adapter():
        from rlinf.models.embodiment.cosmos3.sglang_adapter import (
            Cosmos3SGLangAdapter,
        )

        return Cosmos3SGLangAdapter

    register_sglang_adapter(
        SupportedModel.COSMOS3.value,
        _build_cosmos3_sglang_adapter,
        force=True,
    )


_register_builtin_sglang_adapters()


def get_sglang_adapter_cls(model_type: str):
    """Return the sglang adapter class for ``model_type`` (or ``None``)."""
    builder = _SGLANG_ADAPTER_REGISTRY.get(str(model_type).lower())
    return builder() if builder is not None else None
