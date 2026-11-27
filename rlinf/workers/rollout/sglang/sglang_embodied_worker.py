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

"""Drive embodied training and evaluation through SGLang Diffusion servers.

Training and evaluation own the same server backend and direct control client.
"""

import asyncio
from typing import Any, Literal, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.schema.embodied_types import (
    PolicyInput,
    PolicyOutput,
    PolicyPart,
    split_batch_value,
)
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.obs_compression import decompress_obs, infer_obs_batch_size
from rlinf.utils.placement import EmbodiedComponentPlacement


class SGLangEmbodiedWorker(Worker):
    """Drive SGLang Diffusion over embodied training and evaluation channels."""

    def __init__(
        self,
        config: DictConfig,
        config_rollout: Optional[DictConfig] = None,
    ):
        Worker.__init__(self)
        self.cfg = config
        self.cfg_rollout = (
            config_rollout if config_rollout is not None else config.rollout
        )
        self.model_type = str(
            getattr(getattr(self.cfg_rollout, "model", None), "model_type", "")
        ).lower()
        self.model_cfg = self.cfg_rollout.model
        self.sglang_adapter = None
        self.http_client = None
        self._router_client = None
        self.sglang_server_url = None
        self._backend = None
        self._placement = EmbodiedComponentPlacement(config, Cluster())
        self.only_eval = bool(
            config.runner.get("only_eval", False)
            or config.runner.get("task_type") == "embodied_eval"
        )
        self.enable_offload = bool(config.rollout.get("enable_offload", False))
        self._is_offloaded = False
        self._weights_ready = False
        self.version = 0
        self.global_step = 0
        # Decoupled env/rollout is not implemented on the sglang embodied path.
        assert not config.runner.get("enable_decoupled_mode", False), (
            "SGLangEmbodiedWorker does not support runner.enable_decoupled_mode"
        )
        eval_env_cfg = config.env.get("eval", None)
        self.num_pipeline_stages = int(config.rollout.pipeline_stage_num)
        total_eval = int(eval_env_cfg.total_num_envs) if eval_env_cfg else 0
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

        if not self.only_eval:
            if config.rollout.get("enable_cuda_graph", False):
                raise ValueError(
                    "SGLang embodied training requires CUDA graphs disabled"
                )
            train_env_cfg = config.env.train
            self.rollout_epoch = int(train_env_cfg.rollout_epoch)
            self.train_batch_size = (
                int(train_env_cfg.total_num_envs) // self.num_pipeline_stages
            )
            self.n_train_chunk_steps = int(
                train_env_cfg.max_steps_per_rollout_epoch
            ) // int(self.model_cfg.num_action_chunks)
            self.collect_prev_infos = config.rollout.get("collect_prev_infos", True)
            self.collect_transitions = config.rollout.get("collect_transitions", False)

    async def init_worker(self):
        adapter_cls = None
        if self.model_type:
            from rlinf.models.embodiment.sglang_adapter import (
                get_sglang_adapter_cls,
            )

            adapter_cls = get_sglang_adapter_cls(self.model_type)
        if adapter_cls is None:
            raise RuntimeError(
                f"no sglang adapter registered for model_type "
                f"'{self.model_type}'; cannot run the embodied sglang path"
            )
        self.sglang_adapter = adapter_cls(self.cfg, self._rank)
        if not self.only_eval:
            self.sglang_adapter.validate_training(self.cfg)
        sglang_cfg = self.cfg_rollout.sglang
        self._http_timeout_s = float(
            sglang_cfg.get("http_timeout_s", sglang_cfg.get("timeout_s", 120.0))
        )
        self._http_max_retries = int(
            sglang_cfg.get("http_max_retries", 5 if self.only_eval else 0)
        )
        self._http_retry_backoff_s = float(sglang_cfg.get("http_retry_backoff_s", 1.0))
        configured_concurrency = sglang_cfg.get("max_concurrent_requests")
        self._max_concurrent_requests = (
            max(2, self._placement.rollout_dp_size * 2)
            if configured_concurrency is None
            else int(configured_concurrency)
        )
        if self._max_concurrent_requests < 1:
            raise ValueError("SGLang max_concurrent_requests must be positive")
        if not self.only_eval and self._http_max_retries != 0:
            raise ValueError("SGLang training requires http_max_retries=0")
        from rlinf.workers.rollout.backend import RlinfContext
        from rlinf.workers.rollout.sglang.backends import make_backend

        if not sglang_cfg.get("launch_server", True):
            raise ValueError("Worker-owned SGLang requires launch_server=true")
        launch_router = bool(sglang_cfg.get("launch_router", False))
        if launch_router and not getattr(self.sglang_adapter, "supports_router", True):
            raise ValueError(
                f"{type(self.sglang_adapter).__name__} maintains per-server session "
                "state and does not support rollout.sglang.launch_router=true"
            )
        from rlinf.scheduler.hardware import AcceleratorUtil

        visible_devices = AcceleratorUtil.get_visible_devices(self._accelerator_type)
        num_gpus = self._placement.rollout_gpus_per_process
        if len(visible_devices) != num_gpus:
            raise ValueError(
                f"SGLang server requires {num_gpus} assigned GPUs, "
                f"got {visible_devices}; group rollout placement per instance"
            )
        server_args = OmegaConf.to_container(sglang_cfg.server, resolve=True)
        server_args["num_gpus"] = num_gpus
        needs_rlinf_patch = bool(
            not self.only_eval
            or self.cfg.runner.get("ckpt_path")
            or self.enable_offload
        )
        self._backend = make_backend(
            sglang_cfg.get("backend_type", "server"),
            server_args,
            RlinfContext(
                parent_address=self.worker_address,
                weight_reload=None if self.only_eval else "sync",
                placement=self._placement,
                cfg=self.cfg,
                model_instance_id=self._rank,
            ),
            acquire_free_port=self.acquire_free_port,
            log_info=self.log_info,
            log_error=self.log_error,
            server_type="embodied",
            expose_to_router=launch_router,
            apply_rlinf_patch=needs_rlinf_patch,
        )
        await self._backend.initialize()
        self.http_client = self._backend.http_client
        self.sglang_server_url = self._backend.server_url
        if not self.only_eval:
            torch.manual_seed(int(self.model_cfg.get("seed", 1140)) + self._rank)
        self._weights_ready = self.only_eval
        if self.enable_offload and not self.only_eval:
            await self.offload_model()

    def get_server_url(self) -> str:
        """Return this instance's direct URL for router registration."""
        if self.sglang_server_url is None:
            raise RuntimeError("Initialize the SGLang server before requesting its URL")
        return self.sglang_server_url

    def get_required_router_paths(self) -> list[str]:
        """Return the router HTTP paths required by the active adapter and mode."""
        paths = [self.sglang_adapter.action_path]
        if not self.only_eval:
            rollout_path = getattr(self.sglang_adapter, "rollout_path", None)
            if rollout_path and rollout_path not in paths:
                paths.append(rollout_path)
        return paths

    async def set_sglang_router_url(self, url: str) -> None:
        """Route generation through the router; retain direct control requests."""
        from rlinf.utils.http_client import InferenceHTTPClient

        if self._router_client is not None:
            await self._router_client.aclose()
        self._router_client = InferenceHTTPClient(url)

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        # Delegates to the shared helper, which also understands compressed
        # image markers (inference runs before decompression on the recv path).
        return infer_obs_batch_size(obs_batch)

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        # Reconstruct any image tensors compressed by the env workers. This is a
        # no-op when `env.obs_compression` is disabled (no compression markers).
        obs_batches = [decompress_obs(b) for b in obs_batches]
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
        reset = any(b.get("final_obs") is not None for b in obs_batches)
        return {
            "obs": merged,
            "reset": reset,
            "eval_epoch_done": any(
                bool(batch.get("eval_epoch_done")) for batch in obs_batches
            ),
        }

    async def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """env_obs -> action chunks [N, num_action_chunks, action_dim].

        The model adapter builds, sends, and parses the HTTP request.
        Adapters with ``request_groups`` get one request per env group.
        """
        if not self._weights_ready:
            raise RuntimeError("SGLang serving weights are not ready for rollout")
        from rlinf.models.embodiment.sglang_adapter import (
            gather_env_rows,
            select_env_rows,
        )

        request_groups = getattr(self.sglang_adapter, "request_groups", None)
        groups = request_groups(env_obs) if request_groups is not None else None
        if groups is None or len(groups) <= 1:
            return await self._request_actions(env_obs, mode)

        # Draw per-group seeds/contexts in deterministic env order before
        # running off-thread encoding and concurrent HTTP requests.
        prepared = []
        for group in groups:
            group_obs = select_env_rows(env_obs, group)
            request_context = self.sglang_adapter.prepare_request_context(
                group_obs, mode=mode
            )
            prepared.append((group_obs, request_context))
        client = self._router_client or self.http_client
        if self._max_concurrent_requests == 1:
            outputs = []
            for group_obs, request_context in prepared:
                payload, state = await self.sglang_adapter.async_build_request(
                    group_obs, mode=mode, request_context=request_context
                )
                outputs.append(
                    await self.sglang_adapter.async_send_request(
                        client,
                        payload,
                        state,
                        mode=mode,
                        timeout_s=self._http_timeout_s,
                        max_retries=self._http_max_retries,
                        retry_backoff_s=self._http_retry_backoff_s,
                    )
                )
        else:
            concurrency = min(self._max_concurrent_requests, len(prepared))
            encode_semaphore = asyncio.Semaphore(concurrency)
            http_semaphore = asyncio.Semaphore(concurrency)

            async def _run_request(group_obs, request_context):
                async with encode_semaphore:
                    payload, state = await self.sglang_adapter.async_build_request(
                        group_obs, mode=mode, request_context=request_context
                    )
                async with http_semaphore:
                    return await self.sglang_adapter.async_send_request(
                        client,
                        payload,
                        state,
                        mode=mode,
                        timeout_s=self._http_timeout_s,
                        max_retries=self._http_max_retries,
                        retry_backoff_s=self._http_retry_backoff_s,
                    )

            outputs = await asyncio.gather(
                *(
                    _run_request(group_obs, request_context)
                    for group_obs, request_context in prepared
                )
            )
        order = [index for group in groups for index in group]
        actions = gather_env_rows([actions for actions, _ in outputs], order)
        info = gather_env_rows([info for _, info in outputs], order)
        return actions, info

    async def _request_actions(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return await self.sglang_adapter.async_request_actions(
            self._router_client or self.http_client,
            env_obs,
            mode=mode,
            timeout_s=self._http_timeout_s,
            max_retries=self._http_max_retries,
            retry_backoff_s=self._http_retry_backoff_s,
        )

    async def offload_model(self, tags: Optional[list[str]] = None) -> None:
        """Offload the embodied SGLang server weights to CPU."""
        if self._backend is None or self._is_offloaded:
            return
        await self._backend.offload(tags=tags)
        self._is_offloaded = True

    async def reload_model(self, tags: Optional[list[str]] = None) -> None:
        """Restore offloaded embodied SGLang server weights back to GPU."""
        if self._backend is None or not self._is_offloaded:
            return
        await self._backend.onload(tags=tags)
        self._is_offloaded = False

    async def evaluate(self, input_channel, output_channel):
        """Channel-based embodied eval loop, driven by EmbodiedEvalRunner."""
        from tqdm import tqdm

        if self.enable_offload:
            await self.reload_model()
        try:
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
                        actions, _ = await self.predict(obs, mode="eval")
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
        finally:
            if self.enable_offload and not self.only_eval:
                await self.offload_model()

    @Worker.timer("sync_model_from_actor")
    async def sync_model_from_actor(self) -> None:
        """Trigger direct actor-to-server transfer through the shared control API."""
        if self.only_eval:
            raise RuntimeError("Standalone evaluation has no actor to synchronize")
        if self.enable_offload:
            await self.reload_model()
        self._weights_ready = False
        result = await self._backend.sync_weights()
        self.version = int(result["version"])
        self._weights_ready = True

    async def stop(self) -> None:
        """Stop the server owned by this worker and release its HTTP client."""
        self._weights_ready = False
        self.sglang_server_url = None
        if self._backend is not None:
            await self._backend.async_shutdown()
            self._backend = None
        self.http_client = None
        if self._router_client is not None:
            await self._router_client.aclose()
            self._router_client = None

    def set_global_step(self, global_step: int) -> None:
        """Record the runner step independently of the serving weight version."""
        self.global_step = global_step

    @classmethod
    def _merge_policy_inputs(cls, inputs: list[PolicyInput]) -> PolicyInput:
        for item in inputs:
            item.obs = decompress_obs(item.obs)
            for part in item.env_parts:
                if part is not None and part.next_obs is not None:
                    part.next_obs = decompress_obs(part.next_obs)
        return PolicyInput.merge(inputs)

    async def _receive_policy_input(
        self, channel: Channel, stage_id: int
    ) -> PolicyInput:
        return await self.recv_from(
            group_name=self.cfg.env.group_name,
            channel=channel,
            tag="train_policy",
            route_key=stage_id,
            async_op=True,
            batch_size=self.train_batch_size,
            merge_fn=self._merge_policy_inputs,
            infer_batch_size_fn=lambda item: self._infer_env_batch_size(item.obs),
        ).async_wait()

    async def _publish_env_parts(
        self, policy_input: PolicyInput, actor_channel: Channel
    ) -> None:
        """Complete the environment halves of previously executed chunks."""
        if not policy_input.env_parts:
            return
        if len(policy_input.env_parts) != len(policy_input.request_sizes):
            raise ValueError("Environment parts and request sizes must align")
        observations = split_batch_value(policy_input.obs, policy_input.request_sizes)
        for part, obs in zip(policy_input.env_parts, observations):
            if part is None:
                continue
            next_obs = part.next_obs if part.next_obs is not None else obs
            final_values = None
            if part.requires_inference:
                if next_obs is None:
                    raise ValueError(
                        "Terminal inference requires a post-action observation"
                    )
                _, result = await self.predict(next_obs, mode="train")
                final_values = result.get("prev_values")
            actor_channel.put(
                part.complete(
                    next_obs=next_obs if self.collect_transitions else None,
                    next_rlt_obs=None,
                    final_prev_values=final_values,
                ),
                async_op=True,
            )

    @staticmethod
    def _split_actions(actions: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
        return list(torch.split(actions, sizes, dim=0))

    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel,
    ) -> None:
        """Publish action chunks, diffusion replay inputs and environment outcomes."""
        for _ in range(self.n_train_chunk_steps):
            for stage_id in range(self.num_pipeline_stages):
                policy_input = await self._receive_policy_input(input_channel, stage_id)
                if policy_input.is_last:
                    raise ValueError("Received a final input in the policy stream")
                output = None
                external_actions = None
                if policy_input.requires_inference:
                    actions, result = await self.predict(policy_input.obs, mode="train")
                    logprobs = (
                        result["prev_logprobs"] if self.collect_prev_infos else None
                    )
                    output = PolicyOutput(
                        forward_inputs=result["forward_inputs"],
                        prev_logprobs=logprobs,
                        prev_values=result.get("prev_values")
                        if self.collect_prev_infos
                        else None,
                        versions=torch.full_like(
                            logprobs, float(self.version), dtype=torch.float32
                        )
                        if logprobs is not None
                        else None,
                    )
                    # CPU storage prevents a same-device env from aliasing CUDA buffers.
                    actions = actions.detach().cpu().contiguous()
                    self.send_to(
                        group_name=self.cfg.env.group_name,
                        channel=output_channel,
                        data=actions,
                        tag="policy",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.train_batch_size,
                        split_fn=self._split_actions,
                    )
                else:
                    external_actions = (
                        policy_input.external_actions.detach().cpu().contiguous()
                    )
                actor_channel.put(
                    PolicyPart(
                        sources=policy_input.sources,
                        obs=policy_input.obs if self.collect_transitions else {},
                        output=output,
                        external_actions=external_actions,
                    ),
                    async_op=True,
                )
                await self._publish_env_parts(policy_input, actor_channel)

        for stage_id in range(self.num_pipeline_stages):
            policy_input = await self._receive_policy_input(input_channel, stage_id)
            if not policy_input.is_last:
                raise ValueError("Expected a final policy input at epoch end")
            await self._publish_env_parts(policy_input, actor_channel)

    @Worker.timer("rollout/generate")
    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel,
    ) -> None:
        """Run the synchronous training rollout requested by EmbodiedRunner."""
        if self.only_eval:
            raise RuntimeError("Training rollout requires runner.only_eval=false")
        if self.enable_offload:
            await self.reload_model()
        try:
            for _ in range(self.rollout_epoch):
                await self.generate_one_epoch(
                    input_channel, output_channel, actor_channel
                )
        finally:
            if self.enable_offload:
                await self.offload_model()
