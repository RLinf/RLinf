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

"""HTTP rollout worker for PhyAI Gateway-based embodied evaluation.

Receives batched environment observations through RLinf channels, sends
MessagePack requests to the Gateway's /v1/actions/generations endpoint,
and sends the returned action chunks back to the environment workers.

The evaluation driver supplies Gateway URLs through set_phyai_server_urls
before worker initialization. Each worker selects a URL based on its rank.
This worker supports evaluation only and does not launch the Gateway or
model servers, or load a local policy model.
"""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.utils.obs_compression import infer_obs_batch_size
from rlinf.workers.rollout.embodied_http_worker import EmbodiedRolloutRouterWorker


class PhyAIGatewayRolloutWorker(EmbodiedRolloutRouterWorker):
    """Run channel-based embodied evaluation through a PhyAI Gateway."""

    http_config_key = "phyai"
    http_service_name = "PhyAI Gateway"
    missing_urls_error = (
        "Expected Gateway URLs before init_worker(), got none; "
        "call set_phyai_server_urls(urls) first."
    )

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._model_name = str(
            config.rollout.get("phyai", {}).get("model_name", "pi05")
        )

    def set_phyai_server_urls(self, urls: Sequence[str]) -> None:
        """Set Gateway HTTP addresses before initialization."""
        if (
            isinstance(urls, str)
            or not urls
            or any(not isinstance(url, str) or not url.strip() for url in urls)
        ):
            raise ValueError(f"Expected nonempty Gateway URL strings, got {urls!r}.")
        self._set_http_server_urls([url.strip().rstrip("/") for url in urls])

    def build_phyai_request(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build a batched request without applying model preprocessing.

        Args:
            env_obs: Environment observations and optional routing metadata.
            mode: Evaluation mode; training is not supported.

        Returns:
            The HTTP payload and batch/horizon for response validation.

        Raises:
            NotImplementedError: If mode is not evaluation.
            ValueError: If the action horizon is not positive.
        """
        if mode != "eval":
            raise NotImplementedError(f"Expected mode='eval', got {mode!r}.")
        routing_keys = {"_rlinf_stage_id", "_rlinf_reset"}
        # Model preprocessing belongs to the external server.
        observation = {
            key: value for key, value in env_obs.items() if key not in routing_keys
        }

        states = observation["states"]
        if torch.is_tensor(states):
            observation["states"] = states.to(dtype=torch.float32)
        else:
            observation["states"] = np.asarray(states, dtype=np.float32)

        batch_size = infer_obs_batch_size(observation)
        stage_id = env_obs.get("_rlinf_stage_id", 0)
        if torch.is_tensor(stage_id):
            stage_id = stage_id.item()
        reset = env_obs.get("_rlinf_reset", False)
        if torch.is_tensor(reset):
            reset = bool(reset.item())
        else:
            reset = bool(reset)
        action_horizon = int(self.model_cfg.num_action_chunks)
        if action_horizon <= 0:
            raise ValueError(
                "Expected positive rollout.model.num_action_chunks, "
                f"got {action_horizon}."
            )
        metadata = {
            "mode": mode,
            "batch_size": batch_size,
            "stage_id": int(stage_id),
            "reset": reset,
        }
        payload = {
            "observation": observation,
            "model_name": self._model_name,
            "model": self._model_name,
            "metadata": metadata,
            "requested_action_horizon": action_horizon,
        }
        state = {
            "batch_size": batch_size,
            "requested_action_horizon": action_horizon,
        }
        return payload, state

    def parse_phyai_response(
        self,
        resp: dict[str, Any],
        state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Validate actions returned by the Gateway.

        Args:
            resp: Decoded response with a top-level actions field.
            state: Expected batch size and requested horizon.

        Returns:
            Float32 actions shaped [B, H, D] and empty evaluation metadata.

        Raises:
            RuntimeError: If actions are missing, malformed, or non-finite.
        """
        if "actions" not in resp:
            raise RuntimeError(
                f"Expected an actions field, got response keys {list(resp)}."
            )
        actions = torch.tensor(resp["actions"], dtype=torch.float32)
        expected_shape = (
            state["batch_size"],
            state["requested_action_horizon"],
            int(self.model_cfg.action_dim),
        )
        if tuple(actions.shape) != expected_shape:
            raise RuntimeError(
                f"Expected PhyAI action shape {expected_shape}, "
                f"got {tuple(actions.shape)}."
            )
        if not torch.isfinite(actions).all():
            raise RuntimeError("Expected finite PhyAI actions, got NaN or infinity.")
        return actions, {}

    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        payload, state = self.build_phyai_request(env_obs, mode=mode)
        response = self._post_msgpack("/v1/actions/generations", payload)
        return self.parse_phyai_response(response, state)
