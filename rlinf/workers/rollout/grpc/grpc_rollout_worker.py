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

"""Remote inference with synchronous and real-time-control rollout loops."""

from __future__ import annotations

from typing import Any

import torch

from rlinf.data.schema.embodied_types import RTCActionResponse, RTCRequest
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
        model_action_dim = int(
            self.model_cfg.get("openpi", {}).get(
                "model_action_dim", self.model_cfg.action_dim
            )
        )
        self._client = GRPCPolicyAdapter(
            address,
            action_dim=self.model_cfg.action_dim,
            num_action_chunks=self.model_cfg.num_action_chunks,
            model_action_dim=model_action_dim,
            policy_id=grpc_cfg.policy_id,
            timeout=grpc_cfg.timeout_s,
        )
        self.log_info(f"Connected to policy service at {address}")

    def predict(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        rtc_context: Any | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return remote actions through the same interface as local eval."""
        if self._client is None:
            raise RuntimeError("Policy client is not initialized")
        return self._client.predict_action_batch(
            env_obs, mode=mode, rtc_context=rtc_context
        )

    def shutdown(self) -> None:
        """Close this worker's client; do not stop an external service."""
        if self._client is not None:
            self._client.close()
            self._client = None


class RTCGRPCRolloutWorker(GRPCRolloutWorker):
    """Serve RTC requests through the fixed remote policy service."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._rtc_model_actions: torch.Tensor | None = None

    async def evaluate(self, input_channel, output_channel):
        """Run remote inference until the environment sends ``stop``."""
        self._rtc_model_actions = None
        try:
            while True:
                request: RTCRequest = await self.recv_from(
                    group_name=self.cfg.env.group_name,
                    channel=input_channel,
                    tag="eval_rtc",
                    route_key=0,
                    async_op=True,
                    batch_size=self.total_num_eval_envs,
                    merge_fn=lambda items: items[0],
                    infer_batch_size_fn=lambda data: 1,
                ).async_wait()
                if request.request_type == "stop":
                    break

                rtc_context = None
                if (
                    request.request_type == "replan"
                    and self._rtc_model_actions is not None
                ):
                    from rlinf.models.embodiment.openpi.rtc_guidance import (
                        RTCGuidanceContext,
                    )

                    rtc_context = RTCGuidanceContext(
                        prev_model_actions=self._rtc_model_actions,
                        executed_horizon=request.executed_horizon,
                        delay_steps=request.predicted_delay_steps,
                    )

                actions, result = self.predict(
                    request.obs, mode="eval", rtc_context=rtc_context
                )
                self._rtc_model_actions = result.get("model_actions")
                if self._rtc_model_actions is None:
                    self._rtc_model_actions = actions
                response = RTCActionResponse(
                    actions=actions,
                    model_actions=self._rtc_model_actions,
                    chunk_id=request.chunk_id,
                    guidance_applied=rtc_context is not None,
                )
                self.send_to(
                    group_name=self.cfg.env.group_name,
                    channel=output_channel,
                    data=response,
                    tag="eval_rtc",
                    route_key=0,
                    batch_size=self.total_num_eval_envs,
                    split_fn=lambda data, sizes: [data],
                )
        finally:
            self._rtc_model_actions = None
