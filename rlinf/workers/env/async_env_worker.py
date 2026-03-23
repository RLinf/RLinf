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
from collections import defaultdict
from typing import Any

import numpy as np
import ray
import torch
from omegaconf.omegaconf import DictConfig

from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult, EnvOutput
from rlinf.envs.action_utils import prepare_actions
from rlinf.scheduler import Channel, Worker
from rlinf.workers.env.env_worker import EnvWorker


class AsyncEnvWorker(EnvWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._interact_task: asyncio.Task = None
        assert not self.enable_offload, "Offload not supported in AsyncEnvWorker"

        self._pending_top_reward_refs: list = [None for _ in range(self.stage_num)]
        self._pending_top_reward_outputs: list[EnvOutput | None] = [
            None for _ in range(self.stage_num)
        ]
        self._pending_top_reward_done_flags: list[bool] = [
            False for _ in range(self.stage_num)
        ]
        self._async_episode_frames: list[list[np.ndarray]] = [
            [] for _ in range(self.stage_num)
        ]
        self._async_prev_top_scores: list[float] = [
            0.0 for _ in range(self.stage_num)
        ]
        self._subtask_steps_since_update: list[int] = [
            0 for _ in range(self.stage_num)
        ]
        self._pending_subtask_refs: list = [None for _ in range(self.stage_num)]

    def bootstrap_step(self) -> list[EnvOutput]:
        self._subtask_steps_since_update = [0 for _ in range(self.stage_num)]
        self._pending_subtask_refs = [None for _ in range(self.stage_num)]
        self._reset_top_reward_state()
        return super().bootstrap_step()

    def _reset_top_reward_state(self, stage_id: int | None = None) -> None:
        stage_ids = range(self.stage_num) if stage_id is None else [stage_id]
        for idx in stage_ids:
            self._async_episode_frames[idx] = []
            self._async_prev_top_scores[idx] = 0.0
            self._pending_top_reward_refs[idx] = None
            self._pending_top_reward_outputs[idx] = None
            self._pending_top_reward_done_flags[idx] = False

    def _queue_top_reward(self, env_output: EnvOutput, stage_id: int) -> EnvOutput:
        if not self._top_reward_enabled or self._vlm_planner is None:
            return env_output

        main_images = env_output.obs.get("main_images", None)
        if main_images is not None:
            if isinstance(main_images, torch.Tensor):
                frame = main_images[0].cpu().numpy()
            else:
                frame = np.asarray(main_images[0])
            self._async_episode_frames[stage_id].append(frame)

        if len(self._async_episode_frames[stage_id]) > self._top_reward_max_frames:
            self._async_episode_frames[stage_id] = self._async_episode_frames[stage_id][
                -self._top_reward_max_frames :
            ]

        instruction = self._get_top_reward_instruction(stage_id)
        frames = list(self._async_episode_frames[stage_id])
        self._pending_top_reward_refs[stage_id] = (
            self._vlm_planner.compute_top_reward.remote(frames, instruction)
        )
        self._pending_top_reward_outputs[stage_id] = env_output
        self._pending_top_reward_done_flags[stage_id] = bool(
            env_output.dones is not None and env_output.dones[:, -1].any()
        )
        return env_output

    def _resolve_pending_top_reward(
        self,
        stage_id: int,
        env_output: EnvOutput,
        *,
        wait: bool,
    ) -> EnvOutput:
        if not self._top_reward_enabled or self._vlm_planner is None:
            return env_output

        score_ref = self._pending_top_reward_refs[stage_id]
        if score_ref is None:
            return env_output

        if not wait:
            ready, _ = ray.wait([score_ref], num_returns=1, timeout=0)
            if not ready:
                return env_output

        score_t = float(ray.get(score_ref))
        pending_output = self._pending_top_reward_outputs[stage_id]
        reward = score_t - self._async_prev_top_scores[stage_id]
        self._async_prev_top_scores[stage_id] = score_t
        if pending_output is not None and pending_output.rewards is not None:
            pending_output.rewards[:, -1] = reward

        should_reset = self._pending_top_reward_done_flags[stage_id]
        self._pending_top_reward_refs[stage_id] = None
        self._pending_top_reward_outputs[stage_id] = None
        self._pending_top_reward_done_flags[stage_id] = False

        self.log_info(
            f"[AsyncEnvWorker] TOPReward stage={stage_id}: "
            f"score={score_t:.4f}, delta={reward:.4f}"
        )

        if should_reset:
            self._reset_top_reward_state(stage_id)

        return pending_output if pending_output is not None else env_output

    def _poll_pending_subtask_result(self, stage_id: int) -> None:
        subtask_ref = self._pending_subtask_refs[stage_id]
        if subtask_ref is None:
            return

        ready, _ = ray.wait([subtask_ref], num_returns=1, timeout=0)
        if not ready:
            return

        new_subtask = str(ray.get(subtask_ref))
        self._pending_subtask_refs[stage_id] = None
        self._apply_subtask_update(stage_id, new_subtask)

    def _maybe_update_subtask(self, stage_id: int) -> None:
        self._poll_pending_subtask_result(stage_id)

        if self._subtask_interval <= 0 or self._vlm_planner is None:
            return

        self._subtask_steps_since_update[stage_id] += 1
        if self._subtask_steps_since_update[stage_id] < self._subtask_interval:
            return
        if self._pending_subtask_refs[stage_id] is not None:
            return

        self._subtask_steps_since_update[stage_id] = 0
        env = self.env_list[stage_id]
        obs = getattr(env, "last_obs", None) or {}
        images = []
        main_images = obs.get("main_images", None)
        if main_images is not None:
            if isinstance(main_images, torch.Tensor):
                main_images = main_images.cpu().numpy()
            if main_images.ndim == 4:
                images.append(main_images[0])
            else:
                images.append(main_images)

        self._pending_subtask_refs[stage_id] = self._vlm_planner.get_next_subtask.remote(
            images
        )

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self,
        chunk_actions: torch.Tensor,
        stage_id: int,
    ) -> tuple[EnvOutput, dict[str, Any]]:
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(chunk_actions)
        )
        extracted_obs = obs_list[-1] if isinstance(obs_list, (list, tuple)) else None
        infos = infos_list[-1] if isinstance(infos_list, (list, tuple)) else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any() and "final_info" in infos:
            final_info = infos["final_info"]
            for key in final_info["episode"]:
                env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            if "intervene_action" in infos["final_info"]:
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
            rewards=chunk_rewards,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
        )

        env_output = self._queue_top_reward(env_output, stage_id)
        return env_output, env_info

    async def _run_interact_once(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        self.rollout_results = [
            EmbodiedRolloutResult(
                max_episode_length=self.cfg.env.train.max_episode_steps,
            )
            for _ in range(self.stage_num)
        ]
        env_metrics = defaultdict(list)

        for epoch in range(self.rollout_epoch):
            env_outputs = self.bootstrap_step()
            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                env_batch = env_output.to_dict()
                self.send_env_batch(
                    output_channel,
                    {
                        "obs": env_batch["obs"],
                        "final_obs": env_batch["final_obs"],
                    },
                )

            for _ in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    if cooperative_yield:
                        await asyncio.sleep(0)

                    env_output = env_outputs[stage_id]
                    env_output = self._resolve_pending_top_reward(
                        stage_id,
                        env_output,
                        wait=True,
                    )
                    env_outputs[stage_id] = env_output
                    curr_obs = env_output.obs
                    if env_output.intervene_actions is not None:
                        self.rollout_results[stage_id].update_last_actions(
                            env_output.intervene_actions,
                            env_output.intervene_flags,
                        )

                    rollout_result = self.recv_rollout_results(
                        input_channel,
                        mode="train",
                    )
                    rewards = self.compute_bootstrap_rewards(
                        env_output,
                        rollout_result.bootstrap_values,
                    )
                    chunk_step_result = ChunkStepResult(
                        actions=rollout_result.forward_inputs.get("action", None),
                        prev_logprobs=rollout_result.prev_logprobs
                        if self.collect_prev_infos
                        else None,
                        prev_values=rollout_result.prev_values
                        if self.collect_prev_infos
                        else None,
                        forward_inputs=rollout_result.forward_inputs,
                        versions=rollout_result.versions,
                        dones=env_output.dones,
                        truncations=env_output.truncations,
                        terminations=env_output.terminations,
                        rewards=rewards,
                    )
                    self.rollout_results[stage_id].append_step_result(chunk_step_result)

                    env_output, env_info = self.env_interact_step(
                        rollout_result.actions,
                        stage_id,
                    )
                    self._maybe_update_subtask(stage_id)
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        output_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                    )
                    if self.collect_transitions:
                        next_obs = (
                            env_output.final_obs
                            if env_output.dones.any() and self.cfg.env.train.auto_reset
                            else env_output.obs
                        )
                        self.rollout_results[stage_id].append_transitions(
                            curr_obs,
                            next_obs,
                        )

                    env_outputs[stage_id] = env_output
                    self.record_env_metrics(env_metrics, env_info, epoch)

            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                env_output = self._resolve_pending_top_reward(
                    stage_id,
                    env_output,
                    wait=True,
                )
                env_outputs[stage_id] = env_output
                if env_output.intervene_actions is not None:
                    self.rollout_results[stage_id].update_last_actions(
                        env_output.intervene_actions,
                        env_output.intervene_flags,
                    )

                rollout_result = self.recv_rollout_results(input_channel, mode="train")
                rewards = self.compute_bootstrap_rewards(
                    env_output,
                    rollout_result.bootstrap_values,
                )
                chunk_step_result = ChunkStepResult(
                    prev_values=rollout_result.prev_values
                    if self.collect_prev_infos
                    else None,
                    dones=env_output.dones,
                    truncations=env_output.truncations,
                    terminations=env_output.terminations,
                    rewards=rewards,
                )
                self.rollout_results[stage_id].append_step_result(chunk_step_result)

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        if actor_channel is not None:
            for stage_id in range(self.stage_num):
                await self.send_rollout_trajectories(
                    self.rollout_results[stage_id],
                    actor_channel,
                )

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    async def interact(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
        replay_channel: Channel | None = None,
    ):
        assert self._interact_task is None or self._interact_task.done(), (
            "Previous interact task is still running while a new interact call is made."
        )
        self._interact_task = asyncio.create_task(
            self._interact(
                input_channel,
                output_channel,
                metric_channel,
                replay_channel,
            )
        )
        try:
            await self._interact_task
        except asyncio.CancelledError:
            pass

    async def _interact(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
        replay_channel: Channel | None,
    ):
        while True:
            env_metrics = await self._run_interact_once(
                input_channel,
                output_channel,
                replay_channel,
                cooperative_yield=True,
            )

            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            env_interact_time_metrics = self.pop_execution_times()
            env_interact_time_metrics = {
                f"time/env/{k}": v for k, v in env_interact_time_metrics.items()
            }
            metrics = {
                "rank": self._rank,
                "env": env_metrics,
                "time": env_interact_time_metrics,
            }
            metric_channel.put(metrics, async_op=True)

    async def stop(self):
        if self._interact_task is not None and not self._interact_task.done():
            self._interact_task.cancel()
