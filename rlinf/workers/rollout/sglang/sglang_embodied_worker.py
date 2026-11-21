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

DreamZero eval uses the action endpoint. Qwen VLM training uses the same worker
against a patched language-model server: each LIBERO turn is a text
``/generate``, and CUDA-IPC weight sync plus memory offload stay on that
server. The driver launches the server via
:func:`launch_sglang_router_and_server` and pushes the URLs through
:meth:`set_sglang_server_urls`.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Optional

import torch
from omegaconf import DictConfig

from rlinf.data.schema.embodied_types import PolicyInput, PolicyOutput, PolicyPart
from rlinf.scheduler import Channel, Worker
from rlinf.utils.obs_compression import decompress_obs, infer_obs_batch_size
from rlinf.utils.placement import HybridComponentPlacement


def next_sglang_resume(state: str | None) -> tuple[list[str] | None, str | None]:
    """Two-step resume so FSDP and the SGLang KV cache are not both resident.

    ``None`` means the server is fully on GPU. ``"all"`` follows a full
    release: the next resume brings weights back for the CUDA-IPC copy, and
    the one after that brings the KV cache and CUDA graph back for generation.
    """
    if state == "all":
        return ["weights"], "kv"
    if state == "kv":
        return ["kv_cache", "cuda_graph"], None
    return None, state


class SGLangEmbodiedWorker(Worker):
    """Use a driver-launched ``sglang serve`` + sglang action converter + channel eval."""

    def __init__(
        self,
        config: DictConfig,
        placement: HybridComponentPlacement,
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
        self.sglang_server_url = None
        self._sglang_server_urls = None
        self._sglang_memory: str | None = None
        self._is_server_leader = True
        self.version = 0
        self.global_step = 0
        self.env_decoupled_mode = bool(
            config.runner.get("enable_decoupled_mode", False)
        )
        if self.env_decoupled_mode:
            self.batch_router: dict[str, list] = {
                "policy": [],
                "rollout_results": [],
            }
        self._max_inflight_sessions = config.rollout.get("sglang", {}).get(
            "max_inflight_sessions", None
        )
        self.only_eval = bool(config.runner.get("only_eval", False))
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
        train_env_cfg = config.env.get("train", None) if not self.only_eval else None
        self.rollout_epoch = int(train_env_cfg.rollout_epoch) if train_env_cfg else 1
        total_train = int(train_env_cfg.total_num_envs) if train_env_cfg else 0
        self.train_batch_size = (
            total_train // self.num_pipeline_stages
            if self.num_pipeline_stages
            else total_train
        )
        if train_env_cfg is not None:
            self.n_train_chunk_steps = int(
                train_env_cfg.max_steps_per_rollout_epoch
            ) // int(self.model_cfg.num_action_chunks)
        else:
            self.n_train_chunk_steps = 0

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
        self._init_sglang_server()
        from rlinf.utils.http_client import InferenceHTTPClient

        # uvicorn closes idle connections after 5s. A longer client
        # keepalive reuses that socket on the next control call
        # (release, then a long actor init, then sync) and the write
        # fails with "Cannot write to closing transport".
        self.http_client = InferenceHTTPClient(
            self.sglang_server_url, keepalive_timeout=4.0
        )
        sglang_cfg = self.cfg.rollout.get("sglang", {})
        self._http_timeout_s = float(
            sglang_cfg.get("http_timeout_s", sglang_cfg.get("timeout_s", 120.0))
        )
        self._http_max_retries = int(sglang_cfg.get("http_max_retries", 5))
        self._http_retry_backoff_s = float(sglang_cfg.get("http_retry_backoff_s", 1.0))
        self.sglang_adapter = adapter_cls(self.cfg, self._rank)
        bind = getattr(self.sglang_adapter, "bind_http_client", None)
        if bind is not None:
            bind(self.http_client)

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
        # Exactly one rollout rank per launched SGLang server sends server-level
        # control RPCs (sync weights, release/resume memory).
        self._is_server_leader = int(self._rank) < len(urls)
        self.log_info(
            f"sglang server assigned: rank={self._rank} -> "
            f"{self.sglang_server_url} ({len(urls)} server(s), "
            f"leader={self._is_server_leader})"
        )

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
        if getattr(self.sglang_adapter, "per_env", False):
            outputs = self._request_per_env(env_obs, mode)
            merge = getattr(self.sglang_adapter, "merge_env_results", None)
            if merge is not None:
                return merge(outputs)
            order = list(range(len(outputs)))
            actions = gather_env_rows([item[0] for item in outputs], order)
            info = gather_env_rows([item[1] for item in outputs], order)
            return actions, info
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

    def _request_per_env(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"]
    ) -> list[tuple[torch.Tensor, dict[str, Any]]]:
        """Post one ``/generate`` per environment without waiting between them.

        Session state is read and updated on this thread. Only the HTTP posts
        run together, and responses are parsed in environment order.
        """
        from rlinf.models.embodiment.sglang_adapter import select_env_rows

        batch = self._infer_env_batch_size(env_obs)
        outputs: list[tuple[torch.Tensor, dict[str, Any]] | None] = [None] * batch
        pending: list[tuple[int, tuple[dict[str, Any], dict[str, Any]]]] = []
        for index in range(batch):
            single = select_env_rows(env_obs, [index])
            single["_env_index"] = index
            if self.sglang_adapter.env_is_finished(single):
                outputs[index] = self.sglang_adapter.finished_result(single)
                continue
            payload, state = self.sglang_adapter.build_request(single, mode=mode)
            if isinstance(state, dict) and state.get("budget_exhausted", False):
                outputs[index] = self.sglang_adapter.parse_response({}, state)
                continue
            pending.append((index, (payload, state)))
        if not pending:
            return [item for item in outputs if item is not None]

        def post(
            item: tuple[int, tuple[dict[str, Any], dict[str, Any]]],
        ) -> tuple[int, dict[str, Any], dict[str, Any]]:
            index, (payload, state) = item
            response = self.http_client.post(
                self.sglang_adapter.action_path,
                payload,
                msgpack=bool(getattr(self.sglang_adapter, "use_msgpack", True)),
                timeout_s=self._http_timeout_s,
                max_retries=self._http_max_retries,
                retry_backoff_s=self._http_retry_backoff_s,
            )
            return index, response, state

        cap = getattr(self, "_max_inflight_sessions", None)
        max_workers = len(pending) if not cap else max(1, min(len(pending), int(cap)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            posted = list(pool.map(post, pending))
        for index, response, state in posted:
            outputs[index] = self.sglang_adapter.parse_response(response, state)
        return [item for item in outputs if item is not None]

    def _request_actions(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        adapter = self.sglang_adapter
        payload, state = adapter.build_request(env_obs, mode=mode)
        resp = self.http_client.post(
            adapter.action_path,
            payload,
            msgpack=bool(getattr(adapter, "use_msgpack", True)),
            timeout_s=self._http_timeout_s,
            max_retries=self._http_max_retries,
            retry_backoff_s=self._http_retry_backoff_s,
        )
        return adapter.parse_response(resp, state)

    def set_global_step(self, global_step: int) -> None:
        self.global_step = int(global_step)

    async def sync_model_from_actor(self) -> None:
        """Ask the patched server to receive the actor's CUDA-IPC buckets."""
        close = getattr(self.sglang_adapter, "on_weights_synced", None)
        if close is not None:
            close()
        self.version += 1
        if not getattr(self, "_is_server_leader", self._rank == 0):
            return
        from rlinf.hybrid_engines.sglang.common.io_struct import SyncHFWeightInput

        await self.http_client.async_sync_hf_weight(SyncHFWeightInput())

    async def resume_sglang_memory(self) -> None:
        if not getattr(self, "_is_server_leader", self._rank == 0):
            return
        tags, new_state = next_sglang_resume(self._sglang_memory)
        if tags is None:
            return
        await self.http_client.async_resume_memory_occupation(tags=tags)
        self._sglang_memory = new_state

    async def release_sglang_memory(self) -> None:
        close = getattr(self.sglang_adapter, "close_sessions", None)
        if close is not None:
            close()
        if not getattr(self, "_is_server_leader", self._rank == 0):
            return
        if self._sglang_memory is not None:
            return
        await self.http_client.async_release_memory_occupation()
        self._sglang_memory = "all"

    @staticmethod
    def _decompress_policy_input(policy_input: PolicyInput) -> PolicyInput:
        policy_input.obs = decompress_obs(policy_input.obs)
        for env_part in policy_input.env_parts:
            if env_part is not None and env_part.next_obs is not None:
                env_part.next_obs = decompress_obs(env_part.next_obs)
        return policy_input

    @classmethod
    def _merge_policy_inputs(cls, policy_inputs: list[PolicyInput]) -> PolicyInput:
        return PolicyInput.merge(
            [cls._decompress_policy_input(item) for item in policy_inputs]
        )

    @classmethod
    def _infer_policy_input_batch_size(cls, policy_input: PolicyInput) -> int:
        return cls._infer_env_batch_size(policy_input.obs)

    @staticmethod
    def _split_actions(
        actions: torch.Tensor, split_sizes: list[int]
    ) -> list[torch.Tensor]:
        return list(torch.split(actions, split_sizes, dim=0))

    async def _receive_policy_input(
        self, channel: Channel, tag: str, stage_id: int
    ) -> tuple[PolicyInput, list[int] | None]:
        if getattr(self, "env_decoupled_mode", False):
            return await self.recv_from_and_record_batch_routes_with_timeout(
                group_name=self.cfg.env.group_name,
                channel=channel,
                tag=f"train_{tag}",
                batch_size=self.train_batch_size,
                merge_fn=self._merge_policy_inputs,
                infer_batch_size_fn=self._infer_policy_input_batch_size,
            )
        policy_input = await self.recv_from(
            group_name=self.cfg.env.group_name,
            channel=channel,
            tag=f"train_{tag}",
            route_key=stage_id,
            async_op=True,
            batch_size=self.train_batch_size,
            merge_fn=self._merge_policy_inputs,
            infer_batch_size_fn=self._infer_policy_input_batch_size,
        ).async_wait()
        return policy_input, None

    def _publish_env_parts(
        self, policy_input: PolicyInput, actor_channel: Channel
    ) -> None:
        if not policy_input.env_parts:
            return
        for env_part in policy_input.env_parts:
            if env_part is None:
                continue
            actor_channel.put(
                env_part.complete(
                    next_obs=None, next_rlt_obs=None, final_prev_values=None
                ),
                async_op=True,
            )

    def _send_actions(
        self,
        channel: Channel,
        actions: torch.Tensor,
        stage_id: int,
        split_sizes: list[int] | None = None,
    ) -> None:
        actions = actions.detach().cpu().contiguous()
        if getattr(self, "env_decoupled_mode", False):
            if split_sizes is None:
                raise ValueError("Decoupled policy output requires recorded routes.")
            self.send_to_recorded_batch_routes(
                group_name=self.cfg.env.group_name,
                channel=channel,
                data=actions,
                tag="policy",
                split_fn=self._split_actions,
                split_sizes=split_sizes,
            )
            return
        self.send_to(
            group_name=self.cfg.env.group_name,
            channel=channel,
            data=actions,
            tag="policy",
            route_key=stage_id,
            async_op=True,
            batch_size=self.train_batch_size,
        )

    @Worker.timer("rollout/generate")
    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel,
    ) -> None:
        """LIBERO turns for one rollout epoch, then the closing env message."""
        from tqdm import tqdm

        for _ in tqdm(
            range(self.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _chunk in range(self.n_train_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    policy_input, split_sizes = await self._receive_policy_input(
                        input_channel, "policy", stage_id
                    )
                    if policy_input.is_last:
                        if getattr(self, "env_decoupled_mode", False):
                            self._publish_env_parts(policy_input, actor_channel)
                            continue
                        raise ValueError("Received a final input in the policy stream.")
                    if not policy_input.requires_inference:
                        actor_channel.put(
                            PolicyPart(
                                sources=policy_input.sources,
                                obs={},
                                external_actions=policy_input.external_actions,
                            ),
                            async_op=True,
                        )
                        self._publish_env_parts(policy_input, actor_channel)
                        continue
                    actions, result = self.predict(policy_input.obs, mode="train")
                    prev_logprobs = result["prev_logprobs"]
                    actor_channel.put(
                        PolicyPart(
                            sources=policy_input.sources,
                            obs={},
                            output=PolicyOutput(
                                prev_logprobs=prev_logprobs,
                                prev_values=None,
                                forward_inputs=result["forward_inputs"],
                                versions=torch.full_like(
                                    prev_logprobs,
                                    float(self.version),
                                    dtype=torch.float32,
                                ),
                            ),
                        ),
                        async_op=True,
                    )
                    self._send_actions(output_channel, actions, stage_id, split_sizes)
                    self._publish_env_parts(policy_input, actor_channel)
            if getattr(self, "env_decoupled_mode", False):
                continue
            for stage_id in range(self.num_pipeline_stages):
                policy_input, _ = await self._receive_policy_input(
                    input_channel, "policy_final", stage_id
                )
                if not policy_input.is_last:
                    raise ValueError("Expected a final policy input at epoch end.")
                self._publish_env_parts(policy_input, actor_channel)

    async def evaluate(self, input_channel, output_channel):
        """Channel-based embodied eval loop, driven by EmbodiedEvalRunner."""
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
