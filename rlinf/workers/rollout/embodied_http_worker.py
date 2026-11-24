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
"""Shared HTTP lifecycle for embodied evaluation rollout workers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Optional

import torch
from omegaconf import DictConfig

from rlinf.scheduler import Worker
from rlinf.utils.obs_compression import decompress_obs, infer_obs_batch_size


class EmbodiedRolloutRouterWorker(Worker):
    """Shared channel evaluation and HTTP transport for embodied workers."""

    http_config_key = ""
    http_service_name = "HTTP server"
    missing_urls_error = (
        "HTTP server URLs must be set before init_worker(); "
        "call the corresponding set_*_server_urls() method first."
    )

    def __init__(
        self,
        config: DictConfig,
        config_rollout: Optional[DictConfig] = None,
    ) -> None:
        Worker.__init__(self)
        self.cfg = config
        self.cfg_rollout = (
            config_rollout if config_rollout is not None else config.rollout
        )
        self.model_cfg = self.cfg_rollout.model
        self.http_client = None
        self.http_server_url: Optional[str] = None
        self._http_server_urls: list[str] = []
        self._http_timeout_s = 120.0
        self._http_max_retries = 5
        self._http_retry_backoff_s = 1.0

        self._validate_eval_config(config)
        eval_env_cfg = config.env.get("eval", None)
        self.num_pipeline_stages = int(config.rollout.pipeline_stage_num)
        total_eval = int(eval_env_cfg.total_num_envs) if eval_env_cfg else 0
        self.eval_batch_size = (
            total_eval // self.num_pipeline_stages
            if self.num_pipeline_stages
            else total_eval
        )
        self.eval_rollout_epoch = (
            int(eval_env_cfg.get("rollout_epoch", 1)) if eval_env_cfg else 1
        )
        self.n_eval_chunk_steps = (
            int(eval_env_cfg.max_steps_per_rollout_epoch)
            // int(self.model_cfg.num_action_chunks)
            if eval_env_cfg is not None
            else 0
        )

    def _validate_eval_config(self, config: DictConfig) -> None:
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

    def _set_http_server_urls(self, urls: Sequence[str]) -> None:
        normalized = list(urls)
        self._http_server_urls = normalized

    def _init_http_server(self) -> None:
        if not self._http_server_urls:
            raise RuntimeError(self.missing_urls_error)
        self.http_server_url = self._http_server_urls[
            int(self._rank) % len(self._http_server_urls)
        ]

    async def init_worker(self) -> None:
        """Initialize the HTTP client after the driver supplies server URLs."""
        from rlinf.utils.http_client import InferenceHTTPClient

        self._init_http_server()
        self.http_client = InferenceHTTPClient(self.http_server_url)
        http_cfg = self.cfg.rollout.get(self.http_config_key, {})
        self._http_timeout_s = float(
            http_cfg.get("http_timeout_s", http_cfg.get("timeout_s", 120.0))
        )
        self._http_max_retries = int(http_cfg.get("http_max_retries", 5))
        self._http_retry_backoff_s = float(http_cfg.get("http_retry_backoff_s", 1.0))
        self.log_info(
            f"{self.http_service_name} assigned: rank={self._rank} -> "
            f"{self.http_server_url}"
        )

    def _post_msgpack(self, path: str, payload: dict[str, Any]) -> Any:
        if self.http_client is None:
            raise RuntimeError(
                "HTTP client is not initialized; call init_worker() first."
            )
        return self.http_client.post(
            path,
            payload,
            msgpack=True,
            timeout_s=self._http_timeout_s,
            max_retries=self._http_max_retries,
            retry_backoff_s=self._http_retry_backoff_s,
        )

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        return infer_obs_batch_size(obs_batch)

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_batches = [decompress_obs(batch) for batch in obs_batches]
        obs_dicts = [batch.get("obs", batch) for batch in obs_batches]
        merged: dict[str, Any] = {}
        for key in obs_dicts[0].keys():
            values = [obs[key] for obs in obs_dicts]
            first = next((value for value in values if value is not None), None)
            if first is None:
                merged[key] = None
            elif isinstance(first, torch.Tensor):
                merged[key] = torch.cat(values, dim=0)
            elif isinstance(first, list):
                merged[key] = [item for value in values for item in value]
            else:
                merged[key] = values
        reset = any(batch.get("final_obs") is not None for batch in obs_batches)
        return {
            "obs": merged,
            "reset": reset,
            "eval_epoch_done": any(
                bool(batch.get("eval_epoch_done", False)) for batch in obs_batches
            ),
        }

    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError(
            f"{type(self).__name__}.predict() must implement its backend protocol."
        )

    async def evaluate(self, input_channel, output_channel) -> None:
        """Receive observations and return action chunks through channels."""
        from tqdm import tqdm

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
                        infer_batch_size_fn=self._infer_env_batch_size,
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
                    if isinstance(actions, torch.Tensor):
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
