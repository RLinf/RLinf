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

"""Eval-only ApxInf rollout worker for embodied environments."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.scheduler import Worker
from rlinf.utils.obs_compression import decompress_obs, infer_obs_batch_size


class ApxInfRolloutWorker(Worker):
    """Drive ApxInf PI0.5 inference over RLinf eval channels."""

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        self.cfg = cfg
        self.model_cfg = cfg.rollout.model
        assert cfg.runner.get("only_eval", True), (
            "ApxInfRolloutWorker is eval-only; set runner.only_eval: true"
        )
        assert not cfg.runner.get("enable_decoupled_mode", False), (
            "ApxInfRolloutWorker does not support runner.enable_decoupled_mode"
        )

        self.num_pipeline_stages = int(cfg.rollout.pipeline_stage_num)
        eval_env_cfg = cfg.env.get("eval", None)
        total_eval = int(eval_env_cfg.total_num_envs) if eval_env_cfg else 0
        self.eval_batch_size = total_eval // self.num_pipeline_stages
        self.eval_rollout_epoch = int(eval_env_cfg.rollout_epoch) if eval_env_cfg else 1
        self.n_eval_chunk_steps = (
            int(eval_env_cfg.max_steps_per_rollout_epoch)
            // int(self.model_cfg.num_action_chunks)
            if eval_env_cfg is not None
            else 0
        )
        self.apxinf_adapter = None

    def init_worker(self) -> None:
        from rlinf.models.embodiment.openpi.apxinf_adapter import (
            OpenPIApxInfAdapter,
        )

        current_device = self.torch_platform.current_device()
        device = (
            f"cuda:{current_device}"
            if isinstance(current_device, int)
            else str(current_device)
        )
        self.apxinf_adapter = OpenPIApxInfAdapter(
            self.model_cfg,
            device,
        )
        self.log_info(
            "ApxInf policy loaded: "
            f"device={device}, metadata={dict(self.apxinf_adapter.metadata)}"
        )

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        return infer_obs_batch_size(obs_batch)

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_batches = [decompress_obs(batch) for batch in obs_batches]
        obs_dicts = [batch["obs"] if "obs" in batch else batch for batch in obs_batches]
        merged: dict[str, Any] = {}
        for key in obs_dicts[0]:
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
        return {"obs": merged, "reset": reset}

    def predict(self, env_obs: dict[str, Any]) -> torch.Tensor:
        actions, _ = self.apxinf_adapter.predict_action_batch(env_obs, mode="eval")
        return actions.detach().cpu().contiguous()

    async def evaluate(self, input_channel, output_channel):
        for _ in tqdm(
            range(self.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
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
                    actions = self.predict(env_output["obs"])
                    self.send_to(
                        group_name=self.cfg.env.group_name,
                        channel=output_channel,
                        data=actions,
                        tag="eval_rollout_results",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.eval_batch_size,
                    )

    def shutdown_engine(self) -> None:
        adapter = getattr(self, "apxinf_adapter", None)
        if adapter is not None:
            adapter.close()
            self.apxinf_adapter = None

    def __del__(self):
        self.shutdown_engine()
