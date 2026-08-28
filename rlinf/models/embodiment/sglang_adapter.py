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
"""

from typing import Callable

from rlinf.config import SupportedModel

_SGLANG_ADAPTER_REGISTRY: dict[str, Callable[[], type]] = {}


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
