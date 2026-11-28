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

"""Embodied sglang rollout worker: drive a registered sglang action converter over
channels against a driver-launched ``sglang serve`` (no worker-owned HTTP
server, no in-worker subprocess).

Used by embodied sglang-convert-action models (e.g. DreamZero). The eval driver
launches the ``sglang serve`` server group via
:func:`launch_sglang_router_and_server` and pushes the server URLs to each
rollout worker via :meth:`set_sglang_server_urls`; the worker picks the URL
at its own rank (for N-server parallel throughput), loads the sglang action converter
registered for ``rollout.model.model_type``, and is driven by
``EmbodiedEvalRunner`` over channels (``recv_from``/``send_to``). It does NOT
host its own HTTP server (the agent path uses
:class:`SGLangAgentWorkerWithHTTPServer`).
"""

from typing import Any, Literal, Optional

import torch
from omegaconf import DictConfig

from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.rollout.embodied_http_worker import EmbodiedRolloutRouterWorker


class SGLangEmbodiedWorker(EmbodiedRolloutRouterWorker):
    """Use a driver-launched ``sglang serve`` + sglang action converter + channel eval."""

    http_config_key = "sglang"
    http_service_name = "sglang server"
    missing_urls_error = (
        "sglang server URLs not set; the eval driver must call "
        "rollout_group.set_sglang_server_urls(urls) (after "
        "launch_sglang_router_and_server) before init_workers()."
    )

    def __init__(
        self,
        config: DictConfig,
        placement: HybridComponentPlacement,
        config_rollout: Optional[DictConfig] = None,
    ):
        super().__init__(config, config_rollout=config_rollout)
        self.model_type = str(
            getattr(getattr(self.cfg_rollout, "model", None), "model_type", "")
        ).lower()
        self.sglang_adapter = None

    async def init_worker(self):
        adapter_cls = None
        if self.model_type:
            from rlinf.models.embodiment.sglang_adapter import get_sglang_adapter_cls

            adapter_cls = get_sglang_adapter_cls(self.model_type)
        if adapter_cls is None:
            raise RuntimeError(
                f"no sglang adapter registered for model_type "
                f"'{self.model_type}'; cannot run the embodied sglang path"
            )
        await super().init_worker()
        self.sglang_adapter = adapter_cls(self.cfg, self._rank)

    def set_sglang_server_urls(self, urls) -> None:
        """Receive the sglang server URLs the driver launched."""
        self._set_http_server_urls(urls)

    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """env_obs -> action chunks [N, num_action_chunks, action_dim].

        Owns the sglang HTTP round-trip: the adapter builds the request
        payload and parses the response; this worker performs the msgpack POST.
        Adapters with ``request_groups`` get one request per env group.
        """
        from rlinf.models.embodiment.sglang_adapter import (
            gather_env_rows,
            select_env_rows,
        )

        request_groups = getattr(self.sglang_adapter, "request_groups", None)
        groups = request_groups(env_obs) if request_groups is not None else None
        if groups is None or len(groups) <= 1:
            return self._request_actions(env_obs, mode)
        outputs = [
            self._request_actions(select_env_rows(env_obs, group), mode)
            for group in groups
        ]
        order = [index for group in groups for index in group]
        actions = gather_env_rows([actions for actions, _ in outputs], order)
        info = gather_env_rows([info for _, info in outputs], order)
        return actions, info

    def _request_actions(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        adapter = self.sglang_adapter
        payload, state = adapter.build_request(env_obs, mode=mode)
        resp = self._post_msgpack(adapter.action_path, payload)
        return adapter.parse_response(resp, state)
