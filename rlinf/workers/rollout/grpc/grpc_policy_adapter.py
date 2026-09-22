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

"""Inference-only client for the RLinf policy service."""

from __future__ import annotations

import math
import uuid
from typing import Any

import grpc
import torch

from rlinf.models.embodiment.base_policy import BasePolicy

from .protocol import (
    OPTIONS,
    SERVICE,
    VERSION,
    pack_message,
    unpack_message,
    validate_actions,
    validate_observations,
)


class GRPCPolicyAdapter(BasePolicy):
    """Connect to one fixed policy; close() releases only the client channel.

    Initialization checks protocol, policy identity and action shape. Inference
    has a deadline and is not retried: a late result must not replace fresh data.
    """

    def __init__(
        self,
        server_address: str,
        *,
        action_dim: int,
        num_action_chunks: int,
        model_action_dim: int,
        policy_id: str,
        timeout: float = 30.0,
    ) -> None:
        if not server_address or not policy_id:
            raise ValueError("server_address and policy_id are required")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.timeout = timeout
        self._closed = False
        self._expected = {
            "version": VERSION,
            "action_dim": action_dim,
            "num_action_chunks": num_action_chunks,
            "model_action_dim": model_action_dim,
            "policy_id": policy_id,
        }
        self._channel = grpc.insecure_channel(server_address, options=OPTIONS)
        self._predict_rpc = self._channel.unary_unary(
            f"/{SERVICE}/Predict",
            request_serializer=pack_message,
            response_deserializer=unpack_message,
        )
        try:
            health = self._channel.unary_unary(
                f"/{SERVICE}/Health",
                request_serializer=pack_message,
                response_deserializer=unpack_message,
            )({"version": VERSION}, timeout=timeout)
            self._check_metadata(health)
        except BaseException:
            self.close()
            raise

    def _check_metadata(self, metadata: dict[str, Any]) -> None:
        for key, expected in self._expected.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"Policy service {key} mismatch: expected {expected!r}, got {metadata.get(key)!r}"
                )

    def default_forward(self, **kwargs: Any) -> Any:
        """Training forwards are deliberately unsupported."""
        raise NotImplementedError(
            "gRPC policy supports fixed-checkpoint evaluation only"
        )

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        rtc_context: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return CPU action chunks with unchanged model output units."""
        if self._closed:
            raise RuntimeError("gRPC policy client is closed")
        if mode != "eval" or kwargs:
            raise ValueError("Only evaluation is supported")
        batch = validate_observations(env_obs)
        request_id = uuid.uuid4().hex
        request = {
            "version": VERSION,
            "policy_id": self._expected["policy_id"],
            "request_id": request_id,
            "observations": env_obs,
        }
        if rtc_context is not None:
            request["rtc_context"] = {
                "prev_model_actions": rtc_context.prev_model_actions,
                "executed_horizon": int(rtc_context.executed_horizon),
                "delay_steps": int(rtc_context.delay_steps),
            }
        response = self._predict_rpc(request, timeout=self.timeout)
        self._check_metadata(response)
        if response.get("request_id") != request_id:
            raise ValueError("Policy response does not match the observation request")
        actions = response["actions"]
        validate_actions(
            actions,
            batch,
            self._expected["num_action_chunks"],
            self._expected["action_dim"],
        )
        result: dict[str, Any] = {}
        model_actions = response.get("model_actions")
        if model_actions is not None:
            if not isinstance(model_actions, type(actions)) or model_actions.shape != (
                batch,
                self._expected["num_action_chunks"],
                self._expected["model_action_dim"],
            ):
                raise ValueError("Policy response model_actions has an invalid shape")
            result["model_actions"] = torch.from_numpy(model_actions)
        return torch.from_numpy(actions), result

    def close(self) -> None:
        """Release the channel without stopping the external policy server."""
        if not self._closed:
            self._closed = True
            self._channel.close()

    def __enter__(self) -> GRPCPolicyAdapter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
