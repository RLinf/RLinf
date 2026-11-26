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

import asyncio
from typing import Any, Callable

import torch

from rlinf.config import SupportedModel
from rlinf.utils.obs_compression import infer_obs_batch_size

_SGLANG_ADAPTER_REGISTRY: dict[str, Callable[[], type]] = {}


class SGLangEmbodiedAdapter:
    """Model-specific HTTP requests and weight loading for embodied serving."""

    action_path = "/v1/actions/generations"
    rollout_path = "/rollout/generate"
    action_request_msgpack = True
    action_response_msgpack = False
    rollout_request_msgpack = False
    rollout_response_msgpack = True
    supports_training = False
    supports_router = True

    def validate_training(self, cfg) -> None:
        if not self.supports_training:
            raise ValueError(f"{type(self).__name__} supports evaluation only")

    def create_weight_target(self, load_weights, model_path: str):
        """Return a SGLangWeightTarget using the server-local loading callback."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement create_weight_target() "
            "to load or synchronize serving weights"
        )

    @staticmethod
    def map_weight_name(name: str) -> str | None:
        """Map an actor key to serving state, or return None to exclude it."""
        raise NotImplementedError("Training adapters must implement map_weight_name()")

    def build_request(
        self, env_obs: dict[str, Any], mode: str = "eval"
    ) -> tuple[dict[str, Any], Any]:
        """Build an HTTP payload and local state for parsing the server response."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement build_request()"
        )

    def parse_response(
        self, response: Any, state: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Parse an evaluation action response into ``(actions, info)``."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement parse_response()"
        )

    def parse_rollout_response(
        self, response: Any, state: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Parse a training rollout response into ``(actions, info)``."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement parse_rollout_response()"
        )

    def prepare_request_context(
        self, env_obs: dict[str, Any], mode: str = "eval"
    ) -> dict[str, Any]:
        """Extract deterministic per-group state on the event loop before off-thread encoding."""
        del env_obs, mode
        return {}

    def build_request_with_context(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        request_context: dict[str, Any] | None = None,
    ):
        """Build a request using any pre-drawn deterministic context."""
        del request_context
        return self.build_request(env_obs, mode=mode)

    async def async_build_request(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        *,
        request_context: dict[str, Any] | None = None,
    ):
        """Build a request off the event loop so CPU image encoding overlaps HTTP I/O."""
        return await asyncio.to_thread(
            self.build_request_with_context, env_obs, mode, request_context
        )

    async def async_request_actions(self, client, env_obs, mode="eval", **http_options):
        """Build, asynchronously send, and parse one model request."""
        if mode == "train" and not self.supports_training:
            raise ValueError(f"{type(self).__name__} does not support training rollout")
        request_context = self.prepare_request_context(env_obs, mode=mode)
        payload, state = await self.async_build_request(
            env_obs, mode=mode, request_context=request_context
        )
        return await self.async_send_request(
            client, payload, state, mode=mode, **http_options
        )

    async def async_send_request(
        self, client, payload, state, mode="eval", **http_options
    ):
        """Asynchronously send a prepared request and parse its response."""
        if mode == "train" and not self.supports_training:
            raise ValueError(f"{type(self).__name__} does not support training rollout")
        training = mode == "train"
        response = await client.async_post(
            self.rollout_path if training else self.action_path,
            payload,
            msgpack=(
                self.rollout_request_msgpack
                if training
                else self.action_request_msgpack
            ),
            response_msgpack=(
                self.rollout_response_msgpack
                if training
                else self.action_response_msgpack
            ),
            **http_options,
        )
        if training:
            return self.parse_rollout_response(response, state)
        return self.parse_response(response, state)


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
