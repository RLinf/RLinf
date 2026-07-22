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

"""Embodied sglang rollout worker: drive a registered action policy over
channels against a driver-launched ``sglang serve`` (no worker-owned HTTP
server, no in-worker subprocess).

Used by embodied action-policy models (e.g. DreamZero). The eval driver
launches the ``sglang serve`` server group via
:func:`launch_sglang_router_and_server` and pushes the server URLs to each
rollout worker via :meth:`set_sglang_server_urls`; the worker picks the URL
at its own rank (for N-server parallel throughput), loads the action policy
registered for ``rollout.model.model_type``, and is driven by
``EmbodiedEvalRunner`` over channels (``recv_from``/``send_to``). It does NOT
host its own HTTP server (the agent path uses
:class:`SGLangAgentWorkerWithHTTPServer`).
"""

from typing import Any, Literal, Optional

import torch
from omegaconf import DictConfig
from omegaconf import open_dict as _open_dict

from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker


class SGLangEmbodiedWorker(SGLangWorker):
    """Use a driver-launched ``sglang serve`` + action policy + channel eval."""

    def __init__(
        self,
        config: DictConfig,
        placement: ModelParallelComponentPlacement,
        weight_reload="sync",
        config_rollout: Optional[DictConfig] = None,
    ):
        rollout_cfg = config_rollout if config_rollout is not None else config.rollout
        model_cfg = rollout_cfg.model
        original_model_path = model_cfg.model_path
        tokenizer_path = model_cfg.get("tokenizer_path", None)
        if tokenizer_path:
            with _open_dict(model_cfg):
                model_cfg.model_path = tokenizer_path
        try:
            super().__init__(config, placement, weight_reload, config_rollout)
        finally:
            if tokenizer_path:
                with _open_dict(model_cfg):
                    model_cfg.model_path = original_model_path
        self.cfg_rollout = self._cfg_rollout
        # Embodied action-policy path. model_type selects the action policy (see
        # rlinf/workers/rollout/sglang/action_policies); the policy turns env obs
        # into action chunks by calling the launched sglang serve.
        self.model_type = str(
            getattr(getattr(self._cfg_rollout, "model", None), "model_type", "")
        ).lower()
        self.action_policy = None
        self.sglang_server_url = None
        # sglang server URLs the driver launched (via launch_sglang_router_and_server)
        # and pushed to us via set_sglang_server_urls(). Set at runtime, not in cfg,
        # because the cfg is serialized into each Ray actor at .launch() time —
        # before the driver knows the URLs (the server group hasn't started yet).
        self._sglang_server_urls = None
        # NOTE: no _setup_http_routes() — this worker does not host HTTP; the
        # embodied eval is driven over channels, not via the /evaluate route.

        # Channel-eval attrs (mirror MultiStepRolloutWorker.__init__ so
        # EmbodiedEvalRunner can drive this worker). Set at construction, not in
        # init_worker, so they're available before the serve is spawned.
        cfg = self._cfg
        self.cfg = cfg
        self.model_cfg = self._cfg_rollout.model
        # This worker now is eval-only (spawn a serve + channel eval; no training).
        assert cfg.runner.get("only_eval", True), (
            "SGLangEmbodiedWorker is eval-only; set runner.only_eval: true"
        )
        self.only_eval = True
        eval_env_cfg = cfg.env.get("eval", None)
        self.num_pipeline_stages = int(cfg.rollout.pipeline_stage_num)
        total_eval = int(eval_env_cfg.total_num_envs) if eval_env_cfg else 0
        self.total_num_eval_envs = total_eval
        self.eval_batch_size = (
            total_eval // self.num_pipeline_stages
            if self.num_pipeline_stages
            else total_eval
        )
        self.eval_rollout_epoch = int(eval_env_cfg.rollout_epoch) if eval_env_cfg else 1
        if eval_env_cfg is not None:
            self.n_eval_chunk_steps = int(
                eval_env_cfg.max_steps_per_rollout_epoch
            ) // int(self.model_cfg.num_action_chunks)
        else:
            self.n_eval_chunk_steps = 0
        self.env_decoupled_mode = cfg.runner.get("enable_decoupled_mode", False)
        self.collect_prev_infos = cfg.rollout.get("collect_prev_infos", True)

    async def init_worker(self):
        # Pick the driver-launched sglang server URL assigned to this rank
        # (the driver ran launch_sglang_router_and_server + pushed URLs via
        # set_sglang_server_urls), then load the model's registered action
        # policy. No worker-owned HTTP server (the eval loop is channel-based);
        # eval attrs are already set in __init__.
        policy_cls = None
        if self.model_type:
            from rlinf.workers.rollout.sglang.action_policies import (
                get_action_policy_cls,
            )

            try:
                policy_cls = get_action_policy_cls(self.model_type)
            except ValueError:
                policy_cls = None
        if policy_cls is None:
            raise RuntimeError(
                f"no action policy registered for model_type "
                f"'{self.model_type}'; cannot run the embodied sglang path"
            )
        self._init_sglang_server()
        self.action_policy = policy_cls(self._cfg, self.sglang_server_url, self._rank)

    def set_sglang_server_urls(self, urls) -> None:
        """Receive the sglang server URLs the driver launched."""
        self._sglang_server_urls = list(urls)

    def _init_sglang_server(self) -> None:
        """Pick the pre-launched sglang server URL assigned to this rank."""
        urls = self._sglang_server_urls
        if not urls:
            raise RuntimeError(
                "sglang server URLs not set; the eval driver must call "
                "rollout_group.set_sglang_server_urls(urls) (after "
                "launch_sglang_router_and_server) before init_workers()."
            )
        self.sglang_server_url = urls[int(self._rank) % len(urls)]
        self.log_info(
            f"sglang server assigned: rank={self._rank} -> "
            f"{self.sglang_server_url} ({len(urls)} server(s))"
        )

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [b["obs"] if "obs" in b else b for b in obs_batches]
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
        return {"obs": merged}

    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """env_obs -> action chunks [N, num_action_chunks, action_dim].

        Delegates to the registered action policy (selected by model_type);
        the policy builds the model-specific request, calls the launched
        sglang serve, and parses the returned action.
        """
        if self.action_policy is None:
            raise RuntimeError(
                "no action policy loaded (init_worker not called, or model_type "
                f"'{self.model_type}' has no registered policy)"
            )
        return self.action_policy.infer(env_obs, mode=mode)

    async def evaluate(self, input_channel, output_channel):
        """Channel-based embodied eval loop, driven by EmbodiedEvalRunner.

        Mirrors MultiStepRolloutWorker.evaluate's non-decoupled path: recv an
        obs batch from env, predict actions, send them back. Inherited from the
        SGLangWorker base's channel machinery (recv_from/send_to).
        """
        from tqdm import tqdm

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
                    obs = {
                        **env_output["obs"],
                        "_rlinf_stage_id": stage_id,
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
