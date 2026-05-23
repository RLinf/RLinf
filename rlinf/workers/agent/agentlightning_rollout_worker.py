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

from __future__ import annotations

import logging
import os
import typing
import uuid
from typing import Any, Optional, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

if typing.TYPE_CHECKING:
    from agentlightning import NamedResources, RolloutLegacy
    from agentlightning.adapter.triplet import TraceToTripletBase
    from agentlightning.llm_proxy import LLMProxy
    from agentlightning.store.base import LightningStore
    from agentlightning.types.core import (
        EnqueueRolloutRequest,
        Rollout,
        Triplet,
    )

from rlinf.data.io_struct import DynamicRolloutResult
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement


class AgentLightningRolloutWorker(Worker):
    """Worker for agentlightning task rollout."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__()
        self.cfg = cfg
        self.store: Optional[LightningStore] = None
        self.llm_proxy: Optional[LLMProxy] = None
        self.adapter: Optional[TraceToTripletBase] = None
        self.server_addresses: list[str] = []
        self.llm_timeout_seconds: float = 1200.0
        self.model: str = "default-model"
        self.group_size: int = 1
        self.reward_fillna_value: float = 0.0
        self._resources_id: Optional[str] = None
        self._rollout_ids: set[str] = set()
        self._total_tasks_queued = 0
        self._completed_rollout_ids: dict[str, RolloutLegacy] = {}
        self._data_id_to_rollout_ids: dict[str, list[str]] = {}
        self.is_eval_mode: bool = False

    def init_worker(
        self,
        store: "LightningStore",
        adapter: "TraceToTripletBase",
        server_addresses: Optional[list[str]] = None,
        group_size: int = 1,
        model: str = "default-model",
        reward_fillna_value: float = 0.0,
        is_eval_mode: bool = False,
    ):
        from agentlightning.llm_proxy import LLMProxy

        self.store = store
        self.llm_proxy = LLMProxy(
            port=self.acquire_free_port(),
            model_list=[],
            store=store,
        )
        self.llm_proxy.start()
        self.adapter = adapter
        self.server_addresses = server_addresses or []
        self.group_size = 1 if is_eval_mode else group_size
        self.model = model
        self.reward_fillna_value = reward_fillna_value
        self.is_eval_mode = is_eval_mode

    async def _async_setup_data(
        self,
        data: dict[str, Any],
    ):
        from agentlightning.types.core import (
            EnqueueRolloutRequest,
            RolloutConfig,
        )

        if (
            self._resources_id is None
            and self.server_addresses
            and len(self.server_addresses) > 0
        ):
            await self._update_proxy_server()
            sampling_params = self.cfg.algorithm.get("sampling_params", {})
            if isinstance(sampling_params, DictConfig):
                sampling_params = OmegaConf.to_container(sampling_params, resolve=True)
            if self.is_eval_mode:
                sampling_params["temperature"] = 0.0

            llm_resource = self.llm_proxy.as_resource(
                sampling_parameters=sampling_params
            )

            resources: NamedResources = {"main_llm": llm_resource}
            resources_update = await self.store.add_resources(resources)
            self._resources_id = resources_update.resources_id

        resources_id = self._resources_id

        keys = list(data.keys())
        num_samples = len(data[keys[0]])
        group_size = self.group_size

        enqueue_rollout_requests: list[EnqueueRolloutRequest] = []
        data_id_to_original_sample: dict[str, dict[str, Any]] = {}

        for i in range(num_samples):
            data_id = str(uuid.uuid4())
            original_sample = {key: data[key][i] for key in keys}
            original_sample["data_id"] = data_id
            data_id_to_original_sample[data_id] = original_sample
            self._data_id_to_rollout_ids[data_id] = []

            for rollout_idx in range(group_size):
                task_metadata = {"data_id": data_id}
                rollout_mode = "val" if self.is_eval_mode else "train"
                enqueue_rollout_requests.append(
                    EnqueueRolloutRequest(
                        input=original_sample,
                        mode=rollout_mode,
                        resources_id=resources_id,
                        config=RolloutConfig(
                            unresponsive_seconds=self.llm_timeout_seconds,
                            timeout_seconds=self.llm_timeout_seconds,
                        ),
                        metadata=task_metadata,
                    )
                )

        rollouts = await self.store.enqueue_many_rollouts(enqueue_rollout_requests)
        for rollout in rollouts:
            data_id = cast(dict[str, Any], rollout.metadata)["data_id"]
            self._rollout_ids.add(rollout.rollout_id)
            self._data_id_to_rollout_ids[data_id].append(rollout.rollout_id)
        self._total_tasks_queued += len(rollouts)

    async def _update_proxy_server(self):
        from agentlightning.llm_proxy import ModelConfig

        model_name = (
            os.path.basename(str(self.model))
            if os.path.sep in str(self.model)
            else str(self.model)
        )

        self.llm_proxy.update_model_list(
            [
                ModelConfig(
                    {
                        "model_name": model_name,
                        "litellm_params": {
                            "model": "openai/" + model_name,
                            "api_base": f"http://{address}/v1/",
                            "api_key": "sk-placeholder",
                        },
                    }
                )
                for address in self.server_addresses
            ],
        )
        await self.llm_proxy.restart()

    async def _change_to_triplets(self, rollout: "Rollout") -> "RolloutLegacy":
        from agentlightning import RolloutLegacy
        from agentlightning.types.core import Task

        spans = list(
            await self.store.query_spans(rollout.rollout_id, attempt_id="latest")
        )
        triplets = self.adapter.adapt(spans)
        final_reward: Optional[float] = None
        for triplet in reversed(triplets):
            if triplet.reward is not None:
                final_reward = triplet.reward
                break
        if final_reward is None:
            final_reward = self.reward_fillna_value
        task = Task(
            rollout_id=rollout.rollout_id,
            input=rollout.input,
            mode=rollout.mode,
            resources_id=rollout.resources_id,
            metadata=rollout.metadata or {},
        )

        result_rollout = RolloutLegacy(
            rollout_id=rollout.rollout_id,
            task=task,
            final_reward=final_reward,
            triplets=triplets,
            metadata=rollout.metadata or {},
        )
        return result_rollout

    def _count_tool_calls_in_triplet(self, triplet: "Triplet") -> int:
        if not isinstance(triplet.response, dict):
            return 0
        response_raw_content = triplet.response.get("raw_content")
        if isinstance(response_raw_content, list):
            for msg in response_raw_content:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "assistant":
                    continue
                if msg.get("finish_reason") == "tool_calls":
                    return 1
        return 0

    def _compute_rollout_metrics(
        self,
        rollout_results: list[DynamicRolloutResult],
        rollouts: list["RolloutLegacy"],
    ) -> dict[str, float]:
        if not rollout_results:
            return {
                "agent/mean/reward": 0.0,
                "agent/count/n_rollouts": 0,
                "agent/count/n_rollouts_w_trace": 0,
                "agent/count/n_rollouts_w_reward": 0,
            }

        all_rewards: list[float] = []
        total_prompt_lengths: list[int] = []
        total_response_lengths: list[int] = []
        n_rollouts = 0
        n_rollouts_w_trace = 0

        total_tool_calls = 0
        total_turns = 0
        n_rollouts_w_reward = 0

        for rollout_result in rollout_results:
            batch_size = rollout_result.group_size
            n_rollouts += batch_size
            if rollout_result.rewards is not None:
                if isinstance(rollout_result.rewards, torch.Tensor):
                    rewards_list = rollout_result.rewards.tolist()
                else:
                    rewards_list = rollout_result.rewards
                all_rewards.extend(rewards_list)
            else:
                all_rewards.extend([self.reward_fillna_value] * batch_size)

            if rollout_result.response_lengths:
                n_rollouts_w_trace += batch_size
                total_prompt_lengths.extend(rollout_result.prompt_lengths)
                total_response_lengths.extend(rollout_result.response_lengths)

        for rollout_legacy in rollouts:
            if rollout_legacy.final_reward is not None:
                n_rollouts_w_reward += 1

            if rollout_legacy.triplets:
                num_turns = len(rollout_legacy.triplets)
                total_turns += num_turns
                for triplet in rollout_legacy.triplets:
                    total_tool_calls += self._count_tool_calls_in_triplet(triplet)

        training_reward = np.mean(all_rewards) if all_rewards else 0.0

        def _p90_and_mean_top10p(lengths: list[int]) -> tuple[float, float]:
            if not lengths:
                return 0.0, 0.0
            sorted_l = sorted(lengths)
            n_l = len(sorted_l)
            p90_idx_l = min(n_l - 1, int(np.ceil(0.9 * n_l) - 1))
            top_k_l = max(1, int(np.ceil(0.1 * n_l)))
            return float(sorted_l[p90_idx_l]), float(np.mean(sorted_l[-top_k_l:]))

        p90_prompt_length, mean_top10p_prompt_length = _p90_and_mean_top10p(
            total_prompt_lengths
        )
        p90_response_length, mean_top10p_response_length = _p90_and_mean_top10p(
            total_response_lengths
        )

        metrics = {
            "agent/mean/reward": float(training_reward),
            "agent/count/n_rollouts": n_rollouts,
            "agent/count/n_rollouts_w_trace": n_rollouts_w_trace,
            "agent/count/n_rollouts_w_reward": n_rollouts_w_reward,
            "agent/sum/turn_count": total_turns,
            "agent/mean/turn_count_per_rollout": float(total_turns / n_rollouts)
            if n_rollouts > 0
            else 0.0,
            "agent/p90/prompt_length": p90_prompt_length,
            "agent/mean_top10p/prompt_length": mean_top10p_prompt_length,
            "agent/p90/response_length": p90_response_length,
            "agent/mean_top10p/response_length": mean_top10p_response_length,
            "agent/sum/total_tool_calls": total_tool_calls,
            "agent/mean/tool_calls_per_rollout": float(total_tool_calls / n_rollouts)
            if n_rollouts > 0
            else 0.0,
            "agent/mean/tool_calls_per_turn": float(total_tool_calls / total_turns)
            if total_turns > 0
            else 0.0,
        }

        return metrics

    def _clear_data(self):
        self._completed_rollout_ids.clear()
        self._rollout_ids.clear()
        self._data_id_to_rollout_ids.clear()
        self._total_tasks_queued = 0

    async def _async_get_completed_data_ids(self) -> list[str]:
        completed_data_ids = []
        for data_id, rollout_ids in self._data_id_to_rollout_ids.items():
            if all(
                rollout_id in self._completed_rollout_ids for rollout_id in rollout_ids
            ):
                if data_id not in completed_data_ids:
                    completed_data_ids.append(data_id)
        return completed_data_ids

    async def _async_get_rollout_result_for_data_id(
        self, data_id: str
    ) -> DynamicRolloutResult:
        rollout_ids = self._data_id_to_rollout_ids[data_id]
        rollouts = [
            self._completed_rollout_ids[rollout_id] for rollout_id in rollout_ids
        ]

        max_prompt_len = int(self.cfg.data.get("max_prompt_length", 4096))
        max_response_length = int(self.cfg.data.get("max_response_length", 2048))

        idx_to_traj: list[int] = []
        input_ids_list: list[list[int]] = []
        prompt_lengths_list: list[int] = []
        response_lengths_list: list[int] = []
        is_end_list: list[bool] = []
        rewards_list: list[float] = []
        rollout_logprobs_list: list[list[float]] = []

        for traj_idx, rollout_legacy in enumerate(rollouts):
            for triplet in rollout_legacy.triplets:
                prompt_ids = triplet.prompt.get("token_ids", [])
                response_ids = triplet.response.get("token_ids", [])

                if len(prompt_ids) > max_prompt_len:
                    prompt_ids = prompt_ids[:max_prompt_len]
                if len(response_ids) > max_response_length:
                    response_ids = response_ids[:max_response_length]

                # Actor training requires at least one response token.
                # If token_ids are missing (adapter didn't tokenize) or response is empty,
                # skip this triplet to avoid producing invalid training samples.
                if not response_ids:
                    continue

                input_ids = prompt_ids + response_ids

                turn_logprobs: list[float] = []
                if self.cfg.rollout.return_logprobs:
                    logprobs = triplet.response.get("logprobs", [])
                    if logprobs:
                        turn_logprobs = [lp.get("logprob", 0.0) for lp in logprobs]
                        if len(turn_logprobs) > max_response_length:
                            turn_logprobs = turn_logprobs[:max_response_length]

                idx_to_traj.append(traj_idx)
                input_ids_list.append(input_ids)
                prompt_lengths_list.append(len(prompt_ids))
                response_lengths_list.append(len(response_ids))
                is_end_list.append(True)
                rewards_list.append(rollout_legacy.final_reward)
                rollout_logprobs_list.append(turn_logprobs)

        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)

        dynamic_rollout_result = DynamicRolloutResult(
            num_sequence=len(input_ids_list),
            group_size=len(rollouts),
            idx_to_traj=idx_to_traj,
            input_ids=input_ids_list,
            prompt_lengths=prompt_lengths_list,
            response_lengths=response_lengths_list,
            is_end=is_end_list,
            rewards=rewards_tensor,
            rollout_logprobs=rollout_logprobs_list
            if self.cfg.rollout.return_logprobs
            else None,
        )
        return dynamic_rollout_result

    async def process_rollout_batch(
        self, input_channel: Channel, output_channel: Channel
    ):
        from agentlightning.types.core import Rollout

        with self.worker_timer():
            batch_data = await input_channel.get(async_op=True).async_wait()

            await self._async_setup_data(
                data=batch_data,
            )

            initial_data_ids_count = len(self._data_id_to_rollout_ids)
            processed_data_ids = set()
            rollout_results: list[DynamicRolloutResult] = []

            while len(processed_data_ids) < initial_data_ids_count:
                rollout_ids_to_query = [
                    rid
                    for rid in self._rollout_ids
                    if rid not in self._completed_rollout_ids
                ]

                completed_batch = await self.store.wait_for_rollouts(
                    rollout_ids=rollout_ids_to_query, timeout=0.1
                )

                for rollout in completed_batch:
                    rollout = (
                        await self._change_to_triplets(rollout)
                        if isinstance(rollout, Rollout)
                        else rollout
                    )
                    self._completed_rollout_ids[rollout.rollout_id] = rollout

                completed_data_ids = await self._async_get_completed_data_ids()
                for data_id in completed_data_ids:
                    if data_id in processed_data_ids:
                        continue

                    rollout_result = await self._async_get_rollout_result_for_data_id(
                        data_id
                    )
                    rollout_results.append(rollout_result)
                    output_channel.put(rollout_result, async_op=True)

                    processed_data_ids.add(data_id)

            rollouts_list = list(self._completed_rollout_ids.values())
            metrics = self._compute_rollout_metrics(rollout_results, rollouts_list)
            self._clear_data()
            return metrics

    async def process_eval_batch(self, input_channel: Channel):
        from agentlightning.types.core import Rollout

        with self.worker_timer():
            batch_data = await input_channel.get(async_op=True).async_wait()

            await self._async_setup_data(
                data=batch_data,
            )

            initial_data_ids_count = len(self._data_id_to_rollout_ids)
            processed_data_ids = set()

            while len(processed_data_ids) < initial_data_ids_count:
                rollout_ids_to_query = [
                    rid
                    for rid in self._rollout_ids
                    if rid not in self._completed_rollout_ids
                ]

                completed_batch = await self.store.wait_for_rollouts(
                    rollout_ids=rollout_ids_to_query, timeout=0.1
                )

                for rollout in completed_batch:
                    rollout = (
                        await self._change_to_triplets(rollout)
                        if isinstance(rollout, Rollout)
                        else rollout
                    )
                    self._completed_rollout_ids[rollout.rollout_id] = rollout

                completed_data_ids = await self._async_get_completed_data_ids()
                for data_id in completed_data_ids:
                    if data_id in processed_data_ids:
                        continue
                    processed_data_ids.add(data_id)

            all_rewards: list[float] = []
            for rollout_legacy in self._completed_rollout_ids.values():
                all_rewards.append(rollout_legacy.final_reward)

            avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
            logging.info(
                f"Eval rewards: {all_rewards}, count: {len(all_rewards)}, avg: {avg_reward}"
            )
            self._clear_data()
            return avg_reward

    def update_server_addresses(self, server_addresses: list[str]):
        self.server_addresses = server_addresses
