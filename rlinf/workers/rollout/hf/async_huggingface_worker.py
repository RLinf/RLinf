# Copyright 2025 The RLinf Authors.
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

import asyncio
import gc
from typing import Any, Literal

import numpy as np
import torch
from omegaconf.omegaconf import DictConfig

from rlinf.data.embodied_io_struct import (
    RolloutResult,
)
from rlinf.scheduler import Channel
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.utils import _build_channel_message, _split_channel_message
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class AsyncMultiStepRolloutWorker(MultiStepRolloutWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._generate_task: asyncio.Task = None
        self.staleness_threshold = cfg.algorithm.get("staleness_threshold", None)
        self.num_envs_per_stage = (
            self.cfg.env.train.total_num_envs
            // self._world_size
            // self.num_pipeline_stages
        )
        assert not self.enable_offload, (
            "Offload not supported in AsyncMultiStepRolloutWorker"
        )

        self._background_weight_sync_active = self.cfg.actor.get(
            "sync_weight_no_wait", False
        )
        self._weight_sync_requested = False
        self._weight_sync_work = None
        self._weight_sync_apply_total = 0
        self._weight_sync_coalesced_total = 0
        self._weight_sync_request_total = 0

    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
    ):
        assert self._generate_task is None, (
            "generate task is not None but generate function is called."
        )
        self._generate_task = asyncio.create_task(
            self._generate(input_channel, output_channel, metric_channel)
        )
        try:
            await self._generate_task
        except asyncio.CancelledError:
            pass

    async def _generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
    ):
        if self.env_decoupled_mode:
            if self._background_weight_sync_active:
                await self._poll_background_weight_sync()
            await self.wait_if_stale()
            await self.decoupled_generate_one_epoch(input_channel, output_channel)
        else:
            while True:
                if self._background_weight_sync_active:
                    await self._poll_background_weight_sync()
                await self.wait_if_stale()
                for _ in range(self.rollout_epoch):
                    await self.generate_one_epoch(input_channel, output_channel)
                if self.finished_episodes is not None:
                    self.finished_episodes += (
                        self.total_num_train_envs * self.rollout_epoch
                    )
                rollout_metrics = self.pop_execution_times()
                rollout_metrics = {
                    f"time/rollout/{k}": v for k, v in rollout_metrics.items()
                }
                metric_channel.put(
                    {"rank": self._rank, "time": rollout_metrics},
                    async_op=True,
                )

    async def wait_if_stale(self) -> None:
        if self.staleness_threshold is None:
            return
        assert self.finished_episodes is not None, (
            "finished_episodes should be initialized."
        )
        while True:
            capacity = (
                (self.staleness_threshold + self.version + 1)
                * self.total_num_train_envs
                * self.rollout_epoch
            )
            if (
                self.finished_episodes + self.total_num_train_envs * self.rollout_epoch
                <= capacity
            ):
                break
            await asyncio.sleep(0.01)

    def stop(self):
        if self._generate_task is not None and not self._generate_task.done():
            self._generate_task.cancel()

    async def recv_actor_buckets(self) -> dict[str, torch.Tensor]:
        """Receive actor weights in buckets and merge into one state dict.

        Same wire protocol as ``sync_model_from_actor`` in the parent class: the first
        recv is ``bucket_length``, then ``bucket_length`` shard dicts. The merged dict
        is returned so the caller can ``load_state_dict`` once (e.g. background sync).

        If ``sync_weight_load_instant`` is False, tensors are moved to CPU while
        merging to cap GPU memory; if True, tensors stay on their current device.
        """

        # Receive first bucket to get bucket_length
        bucket_length = await self.recv(
            self.actor_group_name,
            src_rank=self.actor_weight_src_rank,
            async_op=True,
            options=self._sync_weight_comm_options,
        ).async_wait()

        merged: dict[str, torch.Tensor] = {}
        for _ in range(bucket_length):
            bucket: dict[str, torch.Tensor] = await self.recv(
                self.actor_group_name,
                src_rank=self.actor_weight_src_rank,
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()
            for k, v in bucket.items():
                if not self.sync_weight_load_instant:
                    v = v.to("cpu")
                merged[k] = v
            del bucket

        return merged

    def _start_background_weight_sync_if_needed(self):
        if (
            not self._background_weight_sync_active
            or not self._weight_sync_requested
            or self._weight_sync_work is not None
        ):
            return

        self._weight_sync_requested = False
        self._weight_sync_work = asyncio.create_task(self.recv_actor_buckets())

    def _apply_synced_model_weights(self, param_state_dict):
        self.hf_model.load_state_dict(param_state_dict)

        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    async def _poll_background_weight_sync(self):
        self._start_background_weight_sync_if_needed()
        if self._weight_sync_work is None:
            return

        if not self._weight_sync_work.done():
            return

        param_state_dict = await self._weight_sync_work
        self._weight_sync_work = None
        self._apply_synced_model_weights(param_state_dict)
        self._weight_sync_apply_total += 1

        self._start_background_weight_sync_if_needed()

    async def request_actor_sync_model(self):
        self._weight_sync_request_total += 1
        if self._weight_sync_requested or self._weight_sync_work is not None:
            self._weight_sync_coalesced_total += 1
        self._weight_sync_requested = True
        self._start_background_weight_sync_if_needed()

    def _split_rollout_result_by_last_run(
        self,
        rollout_result: RolloutResult,
        sizes: list[int],
        is_last_run: list[bool],
    ) -> list[RolloutResult]:
        """This func according the is_last_run to get the return_result
        if the is_last_run is True:
            the return result:
            RolloutResult(
                actions,
                prev_values,
                bootstrap_values,
            )
        else the is_last_run is False:
            the return result:
            RolloutResult(
                actions,
                prev_logprobs,
                prev_values,
                bootstrap_values,
                save_flags,
                forward_inputs,
                versions,
            )
        the return results is a list of RolloutResult
        """
        assert len(is_last_run) == len(sizes), (
            f"is_last_run and sizes must have the same length, but got {len(is_last_run)} and {len(sizes)}."
        )

        def _split_optional_tensor(
            tensor: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            if tensor is None:
                return tuple(None for _ in sizes)
            return tuple(torch.split(tensor, sizes, dim=0))

        split_actions = _split_optional_tensor(rollout_result.actions)
        split_prev_logprobs = _split_optional_tensor(rollout_result.prev_logprobs)
        split_prev_values = _split_optional_tensor(rollout_result.prev_values)
        split_bootstrap_values = _split_optional_tensor(rollout_result.bootstrap_values)
        split_save_flags = _split_optional_tensor(rollout_result.save_flags)
        split_versions = _split_optional_tensor(rollout_result.versions)
        split_forward_inputs = (
            [{} for _ in sizes]
            if not rollout_result.forward_inputs
            else [
                {
                    key: torch.split(value, sizes, dim=0)[idx]
                    for key, value in rollout_result.forward_inputs.items()
                }
                for idx in range(len(sizes))
            ]
        )

        return_results = []
        for idx in range(len(sizes)):
            if is_last_run[idx]:
                return_results.append(
                    RolloutResult(
                        actions=split_actions[idx],
                        prev_logprobs=None,
                        prev_values=split_prev_values[idx],
                        bootstrap_values=split_bootstrap_values[idx],
                        save_flags=None,
                        forward_inputs=None,
                        versions=None,
                    )
                )
            else:
                return_results.append(
                    RolloutResult(
                        actions=split_actions[idx],
                        prev_logprobs=split_prev_logprobs[idx],
                        prev_values=split_prev_values[idx],
                        bootstrap_values=split_bootstrap_values[idx],
                        save_flags=split_save_flags[idx],
                        forward_inputs=split_forward_inputs[idx],
                        versions=split_versions[idx],
                    )
                )
        return return_results

    def send_rollout_result_to_channel(
        self,
        output_channel: Channel,
        rollout_result: RolloutResult,
        mode: Literal["train", "eval"] = "train",
    ):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        assert mode == "train", "Now eval mode is not supported in env decoupled mode"
        batch_size_map = self.batch_size_map[mode]
        batch_index_map = self.batch_index_map[mode]
        assert len(batch_index_map) == len(batch_size_map), (
            f"batch_index_map and batch_size_map must have the same length, but got {len(batch_index_map)} and {len(batch_size_map)}."
        )

        is_last_run = []
        for i in range(len(batch_size_map)):
            batch_index = batch_index_map[i]
            _, _, _, last_run = _split_channel_message(batch_index)
            is_last_run.append(last_run)

        split_rollout_results = self._split_rollout_result_by_last_run(
            rollout_result, batch_size_map, is_last_run
        )
        for i, shard_result in enumerate(split_rollout_results):
            batch_index = batch_index_map[i]
            env_rank, batch_idx, _, last_run = _split_channel_message(batch_index)

            item = {
                "batch_index": _build_channel_message(
                    env_rank, batch_idx, mode, last_run, "rollout_results"
                ),
                "batch": shard_result,
            }

            output_channel.put(
                item=item,
                key=CommMapper.build_channel_key(
                    None, env_rank, extra=f"{mode}_rollout_results"
                ),
                async_op=True,
            )
        # delete the batch index map
        self.batch_index_map[mode] = []
        return

    async def recv_env_output_from_channel(
        self, input_channel: Channel, mode: Literal["train", "eval"] = "train"
    ) -> dict[str, Any]:
        """Receive env outputs from mapped env ranks and merge if needed.

        Args:
            input_channel: Channel carrying env->rollout outputs.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.

        Returns:
            A single env output dict. When multiple env ranks are mapped to this
            rollout worker, outputs are merged on batch dimension.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        assert mode == "train", "Now eval mode is not supported in env decoupled mode"

        batch_size_map = self.batch_size_map[mode]
        batch_index_map = self.batch_index_map[mode]
        assert len(batch_index_map) == 0, (
            f"batch_index_map must be empty, but got batch_index_map {batch_index_map}."
        )

        obs_batches = []
        for expected_size in batch_size_map:
            obs_batch = await input_channel.get(
                key=CommMapper.build_channel_key(None, None, extra=f"{mode}_obs"),
                async_op=True,
            ).async_wait()
            batch_index = obs_batch["batch_index"]
            batch_index_map.append(batch_index)
            actual_size = self._infer_env_batch_size(obs_batch["batch"])

            assert actual_size == expected_size, (
                f"Expected env output batch size {expected_size} get the batch_index {batch_index}, "
                f"got {actual_size}."
            )
            obs_batches.append(obs_batch["batch"])
        return self._merge_obs_batches(obs_batches)

    async def decoupled_generate_one_epoch(
        self, input_channel: Channel, output_channel: Channel
    ):
        self.update_dagger_beta()
        while True:
            env_output = await self.recv_env_output_from_channel(input_channel)
            actions, result = self.predict(env_output["obs"])
            save_flags = None
            if result.get("expert_label_flag", False):
                save_flags = torch.full(
                    (actions.shape[0], self.cfg.actor.model.num_action_chunks),
                    True,
                    dtype=torch.bool,
                    device=actions.device,
                )
            rollout_result = RolloutResult(
                actions=actions,
                prev_logprobs=result["prev_logprobs"]
                if self.collect_prev_infos
                else None,
                prev_values=result["prev_values"] if self.collect_prev_infos else None,
                bootstrap_values=self.get_bootstrap_values(
                    env_output.get("final_obs", None)
                ),
                save_flags=save_flags,
                forward_inputs=result["forward_inputs"],
                versions=torch.full_like(
                    result["prev_logprobs"],
                    float(self.version),
                    dtype=torch.float32,
                ),
            )
            self.send_rollout_result_to_channel(
                output_channel, rollout_result, mode="train"
            )

    def send_chunk_actions_to_channel(
        self,
        output_channel: Channel,
        chunk_actions: torch.Tensor | np.ndarray,
        mode: Literal["train", "eval"] = "train",
    ):
        """Send action shards to one of the env ranks.

        Args:
            output_channel: Channel carrying rollout->env action chunks.
            chunk_actions: Predicted action chunk batch (tensor or ndarray).
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        batch_size_map = self.batch_size_map[mode]
        batch_index_map = self.batch_index_map[mode]
        assert len(batch_index_map) == len(batch_size_map), (
            f"batch_index_map and batch_size_map must have the same length, but got {len(batch_index_map)} and {len(batch_size_map)}."
        )
        chunk_actions_split = self._split_actions(chunk_actions, batch_size_map)
        for i, chunk_action_i in enumerate(chunk_actions_split):
            if isinstance(chunk_action_i, torch.Tensor):
                chunk_action_i = (
                    chunk_action_i.detach().cpu().contiguous()
                )  # for evaluation

            batch_index = batch_index_map[i]
            env_rank, batch_idx, _, _ = _split_channel_message(batch_index)

            item = {
                "batch_index": _build_channel_message(
                    env_rank, batch_idx, mode, False, "actions"
                ),
                "batch": chunk_action_i,
            }

            output_channel.put(
                item,
                key=CommMapper.build_channel_key(
                    None, env_rank, extra=f"{mode}_actions"
                ),
                async_op=True,
            )

        # delete the batch index map
        self.batch_index_map[mode] = []
        return
