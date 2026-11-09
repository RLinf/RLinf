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
from tqdm import tqdm

from rlinf.scheduler import Channel, Worker
from rlinf.utils.obs_compression import decompress_obs, infer_obs_batch_size


class PhyaiEmbodiedWorker(Worker):
    """Run channel-based embodied evaluation through a PhyAI Gateway."""

    def __init__(self, config: DictConfig):
        Worker.__init__(self)
        self.cfg = config
        self.model_cfg = config.rollout.model
        self.http_client = None
        self.phyai_server_url = None
        self._phyai_server_urls: list[str] = []
        only_eval = config.runner.get("only_eval", True)
        if not only_eval:
            raise ValueError(
                "Expected runner.only_eval=True for HTTP evaluation, "
                f"got {only_eval!r}."
            )
        decoupled = config.runner.get("enable_decoupled_mode", False)
        if decoupled:
            raise ValueError(
                "Expected runner.enable_decoupled_mode=False for HTTP evaluation, "
                f"got {decoupled!r}."
            )
        eval_cfg = config.env.eval
        self.num_pipeline_stages = int(config.rollout.pipeline_stage_num)
        self.eval_batch_size = int(eval_cfg.total_num_envs) // self.num_pipeline_stages
        self.eval_rollout_epoch = int(eval_cfg.get("rollout_epoch", 1))
        self.n_eval_chunk_steps = int(eval_cfg.max_steps_per_rollout_epoch) // int(
            self.model_cfg.num_action_chunks
        )

    async def init_worker(self) -> None:
        """Initialize the HTTP client after the driver supplies Gateway URLs.

        Raises:
            RuntimeError: If Gateway URLs have not been supplied.
        """
        from rlinf.utils.http_client import InferenceHTTPClient

        if not self._phyai_server_urls:
            raise RuntimeError(
                "Expected Gateway URLs before init_worker(), got none; "
                "call set_phyai_server_urls(urls) first."
            )
        self.phyai_server_url = self._phyai_server_urls[
            int(self._rank) % len(self._phyai_server_urls)
        ]
        self.http_client = InferenceHTTPClient(self.phyai_server_url)
        phyai_cfg = self.cfg.rollout.get("phyai", {})
        self._http_timeout_s = float(
            phyai_cfg.get("http_timeout_s", phyai_cfg.get("timeout_s", 120.0))
        )
        self._http_max_retries = int(phyai_cfg.get("http_max_retries", 5))
        self._http_retry_backoff_s = float(phyai_cfg.get("http_retry_backoff_s", 1.0))
        self.log_info(f"PhyAI Gateway assigned: rank={self._rank} -> {self.phyai_server_url}")

    def set_phyai_server_urls(self, urls: Sequence[str]) -> None:
        """Set Gateway HTTP addresses before initialization.

        Args:
            urls: Nonempty sequence of Gateway base URLs.

        Raises:
            ValueError: If a URL is missing or empty.
        """
        if isinstance(urls, str) or not urls or any(
            not isinstance(url, str) or not url.strip() for url in urls
        ):
            raise ValueError(f"Expected nonempty Gateway URL strings, got {urls!r}.")
        self._phyai_server_urls = [url.strip().rstrip("/") for url in urls]

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_batches = [decompress_obs(batch) for batch in obs_batches]
        obs_dicts = [batch.get("obs", batch) for batch in obs_batches]
        merged: dict[str, Any] = {}
        for key in obs_dicts[0].keys():
            values = [d[key] for d in obs_dicts]
            first = next((v for v in values if v is not None), None)
            if first is None:
                merged[key] = None
            elif isinstance(first, torch.Tensor):
                merged[key] = torch.cat(values, dim=0)
            elif isinstance(first, list):
                merged[key] = [item for sub in values for item in sub]
            else:
                merged[key] = values
        reset = any(b.get("final_obs") is not None for b in obs_batches)
        return {
            "obs": merged,
            "reset": reset,
            "eval_epoch_done": any(
                bool(batch.get("eval_epoch_done", False)) for batch in obs_batches
            ),
        }

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
        action_horizon = self.model_cfg.get("action_horizon", None)
        if action_horizon is None:
            action_horizon = self.model_cfg.get("num_action_chunks", 1)
        action_horizon = int(action_horizon)
        if action_horizon <= 0:
            raise ValueError(
                f"requested action horizon must be positive, got {action_horizon}"
            )
        metadata = {
            "mode": mode,
            "batch_size": batch_size,
            "stage_id": int(stage_id),
            "reset": reset,
        }
        payload = {
            "observation": observation,
            "model_name": "pi05",
            "model": "pi05",
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
        """Request action chunks from the PhyAI Gateway.

        Args:
            env_obs: Batched raw environment observations.
            mode: Evaluation mode; training is not supported.

        Returns:
            Actions shaped [B, H, D] and empty evaluation metadata.
        """
        payload, state = self.build_phyai_request(env_obs, mode=mode)
        resp = self.http_client.post(
            "/v1/actions/generations",
            payload,
            msgpack=True,
            timeout_s=self._http_timeout_s,
            max_retries=self._http_max_retries,
            retry_backoff_s=self._http_retry_backoff_s,
        )
        return self.parse_phyai_response(resp, state)

    async def evaluate(self, input_channel: Channel, output_channel: Channel) -> None:
        """Receive observations and return action chunks for each evaluation epoch.

        Args:
            input_channel: Channel receiving observations from environment workers.
            output_channel: Channel delivering actions to environment workers.
        """
        for _ in tqdm(
            range(self.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            eval_epoch_done = False
            for _ in range(self.n_eval_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_from(
                        group_name=self.cfg.env.group_name,
                        channel=input_channel,
                        tag="eval_rollout_results",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.eval_batch_size,
                        merge_fn=self._merge_obs_batches,
                        infer_batch_size_fn=infer_obs_batch_size,
                    ).async_wait()
                    if env_output.get("eval_epoch_done"):
                        eval_epoch_done = True
                        break
                    obs = {
                        **env_output["obs"],
                        "_rlinf_stage_id": stage_id,
                        "_rlinf_reset": env_output.get("reset", False),
                    }
                    actions, _ = self.predict(obs, mode="eval")
                    actions = actions.detach().cpu().contiguous()
                    self.send_to(
                        group_name=self.cfg.env.group_name,
                        channel=output_channel,
                        data=actions,
                        tag="eval_rollout_results",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.eval_batch_size,
                    )
                if eval_epoch_done:
                    break
