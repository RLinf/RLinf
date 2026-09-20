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

"""Remote inference with the standard embodied rollout channel loop."""

from __future__ import annotations

from typing import Any

import torch

from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

from .grpc_policy_adapter import GRPCPolicyAdapter


class GRPCRolloutWorker(MultiStepRolloutWorker):
    """Reuse embodied evaluation routing without loading a local policy."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._server_address = None
        self._client = None

    def set_server_address(self, address: str) -> None:
        """Set the runtime endpoint without mutating the user's configuration."""
        if self._client is not None:
            raise RuntimeError("Cannot change endpoint after initialization")
        self._server_address = address

    def init_worker(self) -> None:
        """Check server readiness and its fixed-checkpoint action contract."""
        if self._client is not None:
            raise RuntimeError("Policy client is already initialized")
        grpc_cfg = self.cfg.rollout.grpc
        address = self._server_address or grpc_cfg.get("server_address")
        self._client = GRPCPolicyAdapter(
            address,
            action_dim=self.model_cfg.action_dim,
            num_action_chunks=self.model_cfg.num_action_chunks,
            policy_id=grpc_cfg.policy_id,
            timeout=grpc_cfg.timeout_s,
        )
        self.log_info(f"Connected to policy service at {address}")

    def predict(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return remote actions through the same interface as local eval."""
        if self._client is None:
            raise RuntimeError("Policy client is not initialized")
        return self._client.predict_action_batch(env_obs, mode=mode)

    def shutdown(self) -> None:
        """Close this worker's client; do not stop an external service."""
        if self._client is not None:
            self._client.close()
            self._client = None
