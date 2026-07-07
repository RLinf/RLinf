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

"""Real-Time Control (RTC) env worker for real-world OpenPI evaluation.

This subclass overrides :meth:`evaluate` so that actions are executed
step-by-step while the next action chunk is requested asynchronously from the
rollout worker, hiding inference latency behind control execution.
"""

import time
from collections import defaultdict, deque
from typing import Any

import numpy as np
import torch
from omegaconf.omegaconf import DictConfig

from rlinf.data.embodied_io_struct import (
    EnvOutput,
    RTCActionResponse,
    RTCRequest,
)
from rlinf.envs.action_utils import prepare_actions
from rlinf.scheduler import Channel
from rlinf.utils.comm_mapping import CommMapper
from rlinf.workers.env.env_worker import EnvWorker


class RTCEnvWorker(EnvWorker):
    """Env worker that drives real-time-control evaluation.

    Channel convention (same as the base class): ``input_channel`` carries
    data received from the rollout worker (RTC action responses) and
    ``rollout_channel`` carries data sent to the rollout worker (RTC
    requests).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._assert_rtc_eval_supported()

    def _assert_rtc_eval_supported(self):
        rtc_cfg = self.cfg.runner.get("rtc", {})
        if not rtc_cfg.get("enabled", False):
            return
        assert self.cfg.env.eval.env_type == "realworld", (
            "RTC evaluation is currently only integrated for real-world envs."
        )
        assert str(self.cfg.actor.model.model_type) == "openpi", (
            "RTC real-world evaluation is currently integrated for the OpenPI policy path."
        )
        assert self.stage_num == 1, (
            "RTC real-world evaluation currently supports a single pipeline stage."
        )
        assert self.eval_num_envs_per_stage == 1, (
            "RTC real-world evaluation currently supports a single env per worker."
        )

    def send_rtc_request(
        self, rollout_channel: Channel, rtc_request: RTCRequest, mode: str = "eval"
    ) -> None:
        """Send an RTC bootstrap/replan request to the mapped rollout worker."""
        assert mode == "eval", "RTC requests are only supported in eval mode."
        dst_ranks_and_sizes = self.dst_rank_map[f"rollout_{mode}"]
        assert len(dst_ranks_and_sizes) == 1, (
            "RTC real-world evaluation currently supports a single env->rollout route."
        )
        dst_rank, _ = dst_ranks_and_sizes[0]
        rollout_channel.put(
            item=rtc_request,
            key=CommMapper.build_channel_key(self._rank, dst_rank, extra=f"{mode}_rtc"),
        )

    def recv_rtc_response(
        self, input_channel: Channel, mode: str = "eval", async_op: bool = False
    ):
        """Receive an RTC action response from the mapped rollout worker."""
        assert mode == "eval", "RTC responses are only supported in eval mode."
        src_ranks_and_sizes = self.src_rank_map[f"rollout_{mode}"]
        assert len(src_ranks_and_sizes) == 1, (
            "RTC real-world evaluation currently supports a single rollout->env route."
        )
        src_rank, _ = src_ranks_and_sizes[0]
        return input_channel.get(
            key=CommMapper.build_channel_key(src_rank, self._rank, extra=f"{mode}_rtc"),
            async_op=async_op,
        )

    def _copy_rtc_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            return obs.clone()
        if isinstance(obs, dict):
            return {key: self._copy_rtc_obs(value) for key, value in obs.items()}
        if isinstance(obs, list):
            return [self._copy_rtc_obs(value) for value in obs]
        if isinstance(obs, tuple):
            return tuple(self._copy_rtc_obs(value) for value in obs)
        return obs

    def _maybe_rewrite_eval_chunk_gripper(self, chunk_actions):
        rewrite_chunk_gripper = bool(
            self.cfg.actor.model.get("rewrite_chunk_gripper", False)
        )
        if not rewrite_chunk_gripper:
            return chunk_actions

        # Keep the gripper command stable within a chunk. If the policy predicts
        # at least two close commands, close for the whole chunk; otherwise open.
        gripper = chunk_actions[..., -1]
        if isinstance(gripper, torch.Tensor):
            gripper_binary = (gripper > 0.5).to(dtype=gripper.dtype)
            zeros_count = (1.0 - gripper_binary).sum(dim=1, keepdim=True)
            final_gripper = (zeros_count < 2).to(dtype=gripper.dtype)
            chunk_actions[..., -1] = final_gripper
        else:
            gripper_np = np.asarray(gripper)
            gripper_binary = (gripper_np > 0.5).astype(gripper_np.dtype)
            zeros_count = (1.0 - gripper_binary).sum(axis=1, keepdims=True)
            final_gripper = (zeros_count < 2).astype(gripper_np.dtype)
            chunk_actions[..., -1] = final_gripper
        return chunk_actions

    def _evaluate_rtc_action(
        self, env_action: torch.Tensor | np.ndarray, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """Execute exactly one real-world action during RTC evaluation."""
        extracted_obs, step_reward, terminations, truncations, infos = (
            self.eval_env_list[stage_id].step(
                env_action, auto_reset=self.cfg.env.eval.auto_reset
            )
        )

        env_info = {}
        dones = torch.logical_or(terminations, truncations)
        final_obs = (
            infos["final_observation"]
            if isinstance(infos, dict) and "final_observation" in infos
            else None
        )

        if isinstance(infos, dict):
            if dones.any():
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
                if "final_info" in infos:
                    final_info = infos["final_info"]
                    if "episode" in final_info:
                        for key in final_info["episode"]:
                            env_info[key] = final_info["episode"][key][dones].cpu()
        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            rewards=step_reward,
            dones=dones,
            terminations=terminations,
            truncations=truncations,
        )
        return env_output, env_info

    @staticmethod
    def _success_from_info(env_info: dict[str, Any]) -> bool:
        """Check whether an episode succeeded based on env_info keys."""
        for key in ("success_once", "success_at_end", "success"):
            value = env_info.get(key)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                return bool(value.any().item())
            return bool(np.asarray(value).any())
        return False

    def _prepare_rtc_actions(self, rtc_response: RTCActionResponse):
        """Prepare and optionally rewrite gripper actions from an RTC response."""
        chunk_actions = prepare_actions(
            raw_chunk_actions=rtc_response.actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
        )
        return self._maybe_rewrite_eval_chunk_gripper(chunk_actions)

    def evaluate(self, input_channel: Channel, rollout_channel: Channel):
        """Run real-time-control evaluation.

        Args:
            input_channel: Channel carrying RTC action responses from the
                rollout worker.
            rollout_channel: Channel used to send RTC requests to the rollout
                worker.
        """
        rtc_cfg = self.cfg.runner.get("rtc", {})
        min_exec_horizon = int(rtc_cfg.get("min_exec_horizon", 2))
        initial_delay_steps = int(rtc_cfg.get("initial_delay_steps", 1))
        delay_buffer_size = int(rtc_cfg.get("delay_buffer_size", 8))

        eval_metrics = defaultdict(list)
        stage_id = 0
        stop_eval_on_success = bool(
            self.cfg.runner.get("stop_eval_on_keyboard_success", False)
        )

        for eval_rollout_epoch in range(self.cfg.algorithm.eval_rollout_epoch):
            self.eval_env_list[stage_id].is_start = True

            extracted_obs, infos = self.eval_env_list[stage_id].reset()
            env_output = EnvOutput(
                obs=extracted_obs,
                final_obs=(
                    infos["final_observation"] if "final_observation" in infos else None
                ),
            )
            env_batch = env_output.to_dict()

            chunk_id = 0
            episode_step = 0
            episode_success = False
            episode_done = False
            delay_buffer = deque([initial_delay_steps], maxlen=delay_buffer_size)

            self.send_rtc_request(
                rollout_channel,
                RTCRequest(
                    obs=env_batch["obs"],
                    request_type="bootstrap",
                    executed_horizon=0,
                    predicted_delay_steps=initial_delay_steps,
                    chunk_id=chunk_id,
                ),
                mode="eval",
            )
            rtc_response: RTCActionResponse = self.recv_rtc_response(
                input_channel, mode="eval", async_op=False
            )
            current_chunk_actions = self._prepare_rtc_actions(rtc_response)
            current_chunk_index = 0
            current_chunk_len = current_chunk_actions.shape[1]
            pending_rtc_response = None
            request_start_step = 0

            max_eval_steps = self.cfg.env.eval.max_steps_per_rollout_epoch
            while episode_step < max_eval_steps:
                if pending_rtc_response is not None and pending_rtc_response.done():
                    rtc_response = pending_rtc_response.wait()
                    observed_delay_steps = max(episode_step - request_start_step, 0)
                    delay_buffer.append(observed_delay_steps)
                    current_chunk_actions = self._prepare_rtc_actions(rtc_response)
                    current_chunk_len = current_chunk_actions.shape[1]
                    current_chunk_index = observed_delay_steps
                    pending_rtc_response = None
                    chunk_id = rtc_response.chunk_id

                if (
                    pending_rtc_response is None
                    and current_chunk_index >= min_exec_horizon
                ):
                    predicted_delay_steps = int(max(delay_buffer))
                    # Request the next chunk without blocking control execution.
                    self.send_rtc_request(
                        rollout_channel,
                        RTCRequest(
                            obs=self._copy_rtc_obs(env_output.to_dict()["obs"]),
                            request_type="replan",
                            executed_horizon=current_chunk_index,
                            predicted_delay_steps=predicted_delay_steps,
                            chunk_id=chunk_id + 1,
                        ),
                        mode="eval",
                    )
                    pending_rtc_response = self.recv_rtc_response(
                        input_channel, mode="eval", async_op=True
                    )
                    request_start_step = episode_step

                action_index = current_chunk_index
                if action_index >= current_chunk_len:
                    action_index = current_chunk_len - 1

                env_action = current_chunk_actions[:, action_index]
                env_output, env_info = self._evaluate_rtc_action(env_action, stage_id)
                if self.eval_chunk_pause_seconds > 0:
                    time.sleep(self.eval_chunk_pause_seconds)

                for key, value in env_info.items():
                    eval_metrics[key].append(value)
                episode_success = episode_success or self._success_from_info(env_info)

                episode_step += 1
                current_chunk_index += 1

                episode_done = env_output.dones is not None and bool(
                    env_output.dones.any()
                )
                if episode_done:
                    break

            if pending_rtc_response is not None:
                pending_rtc_response.wait()
                pending_rtc_response = None

            if not episode_done:
                eval_metrics["success_once"].append(
                    torch.tensor([0.0], dtype=torch.float32)
                )
                eval_metrics["return"].append(torch.tensor([0.0], dtype=torch.float32))
                eval_metrics["episode_len"].append(
                    torch.tensor([episode_step], dtype=torch.float32)
                )
                eval_metrics["reward"].append(torch.tensor([0.0], dtype=torch.float32))
                eval_metrics["intervened_once"].append(
                    torch.tensor([0.0], dtype=torch.float32)
                )
                eval_metrics["intervened_steps"].append(
                    torch.tensor([0.0], dtype=torch.float32)
                )
                eval_metrics["success_no_intervened"].append(
                    torch.tensor([0.0], dtype=torch.float32)
                )
            self.finish_rollout(mode="eval")
            if stop_eval_on_success and episode_success:
                break

        self.send_rtc_request(
            rollout_channel,
            RTCRequest(
                obs={},
                request_type="stop",
                executed_horizon=0,
                predicted_delay_steps=0,
                chunk_id=0,
            ),
            mode="eval",
        )

        for stage_id in range(self.stage_num):
            if self.cfg.env.eval.get("enable_offload", False) and hasattr(
                self.eval_env_list[stage_id], "offload"
            ):
                self.eval_env_list[stage_id].offload()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics
