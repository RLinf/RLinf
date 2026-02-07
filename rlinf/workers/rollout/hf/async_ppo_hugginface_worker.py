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

import asyncio

import torch
from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.scheduler import Channel
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class AsyncPPOMultiStepRolloutWorker(MultiStepRolloutWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._pause_lock = asyncio.Lock()
        self._pause = asyncio.Event()

        self.version = 0
        self.staleness_threshold = cfg.algorithm.get("staleness_threshold", 0)
        self.finished_episodes = 0
        self.num_envs_per_stage = (
            self.cfg.env.train.total_num_envs
            // self._world_size
            // self.num_pipeline_stages
        )

    async def stage_step(
        self, stage_id: int, env_output: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        self.rollout_results: list[EmbodiedRolloutResult]
        if env_output["intervene_actions"] is not None:
            self.rollout_results[stage_id].update_last_actions(
                env_output["intervene_actions"], env_output["intervene_flags"]
            )

        dones, rewards = self.get_dones_and_rewards(env_output)
        await self.wait_if_paused()
        async with self._pause_lock:
            actions, result = self.predict(env_output["obs"])
        chunk_step_result = ChunkStepResult(
            actions=result.get("action", None),
            dones=dones,
            rewards=rewards,
            truncations=env_output["truncations"],
            terminations=env_output["terminations"],
            prev_logprobs=result["prev_logprobs"],
            prev_values=result["prev_values"],
            forward_inputs=result["forward_inputs"],
            versions=torch.full(
                result["prev_logprobs"].shape, self.version, dtype=torch.float32
            ),
        )
        self.rollout_results[stage_id].append_step_result(result=chunk_step_result)

        return actions

    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel,
    ) -> None:
        assert not self.enable_offload, (
            "Offload not supported in AsyncPPOMultiStepRolloutWorker"
        )
        assert not self.collect_transitions, (
            "collect_transitions not supported in AsyncPPOMultiStepRolloutWorker"
        )
        while not self.should_stop:
            self.rollout_results: list[EmbodiedRolloutResult] = [
                EmbodiedRolloutResult(
                    max_episode_length=self.cfg.env.train.max_episode_steps,
                    model_weights_id=self.model_weights_id,
                )
                for _ in range(self.num_pipeline_stages)
            ]

            await self.wait_if_stale()
            for _ in range(self.n_train_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)
                    actions = await self.stage_step(
                        stage_id,
                        env_output,
                    )
                    self.send_chunk_actions(output_channel, actions)

            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                if env_output["intervene_actions"] is not None:
                    self.rollout_results[stage_id].update_last_actions(
                        env_output["intervene_actions"], env_output["intervene_flags"]
                    )

                # Get dones and rewards from environment batch (final step of epoch)
                dones, rewards = self.get_dones_and_rewards(env_output)

                await self.wait_if_paused()
                async with self._pause_lock:
                    _, result = self.predict(env_output["obs"])

                chunk_step_result = ChunkStepResult(
                    dones=dones,
                    rewards=rewards,
                    truncations=env_output["truncations"],
                    terminations=env_output["terminations"],
                    prev_logprobs=None,
                    prev_values=result["prev_values"],
                    forward_inputs=None,
                )
                self.rollout_results[stage_id].append_step_result(chunk_step_result)

            for stage_id in range(self.num_pipeline_stages):
                await self.send_rollout_trajectories(
                    self.rollout_results[stage_id], actor_channel
                )
                self.finished_episodes += self.num_envs_per_stage

    async def pause_generation(self):
        async with self._pause_lock:
            self._pause.set()

    async def resume_generation(self):
        async with self._pause_lock:
            self._pause.clear()

    async def wait_if_paused(self):
        while self._pause.is_set():
            await asyncio.sleep(0.1)

    async def wait_if_stale(self):
        while True:
            capacity = (
                (self.staleness_threshold + self.version + 1)
                * self.num_envs_per_stage
                * self.num_pipeline_stages
            )
            if self.finished_episodes + self.num_envs_per_stage <= capacity:
                break
            await asyncio.sleep(0.1)

    async def set_version(self, version: int):
        self.version = version
