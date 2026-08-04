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
        rtc_cfg = self.cfg.runner.get("rtc", {})
        self.eval_chunk_pause_seconds = float(rtc_cfg.get("chunk_pause_seconds", 0.0))
        self._assert_rtc_eval_supported()

    def _assert_rtc_eval_supported(self):
        rtc_cfg = self.cfg.runner.get("rtc", {})
        if not rtc_cfg.get("enabled", False):
            return
        assert str(self.cfg.actor.model.model_type) == "openpi", (
            "RTC real-world evaluation is currently integrated for the OpenPI policy path."
        )
        assert self.stage_num == 1, (
            "RTC real-world evaluation currently supports a single pipeline stage."
        )
        assert self.eval_num_envs_per_stage == 1, (
            "RTC real-world evaluation currently supports a single env per worker."
        )

        env_type = self.cfg.env.eval.env_type
        chunk_pause_seconds = float(rtc_cfg.get("chunk_pause_seconds", 0.0))
        inject_delay_ms = float(rtc_cfg.get("inject_delay_ms", 0.0))
        fixed_delay_steps = int(rtc_cfg.get("fixed_delay_steps", 0))

        if env_type == "realworld":
            assert chunk_pause_seconds == 0.0, (
                f"RTC real-world evaluation: chunk_pause_seconds must be 0.0 "
                f"(real robot has real execution time), got {chunk_pause_seconds}."
            )
            assert fixed_delay_steps == 0, (
                f"RTC real-world evaluation: fixed_delay_steps must be 0 "
                f"(only available in simulation with chunk_pause_seconds), got {fixed_delay_steps}."
            )
        else:
            assert inject_delay_ms == 0.0, (
                f"RTC simulation evaluation: inject_delay_ms must be 0.0 "
                f"(use chunk_pause_seconds to simulate delay instead), got {inject_delay_ms}."
            )

    def send_rtc_request(
        self, rollout_channel: Channel, rtc_request: RTCRequest, mode: str = "eval"
    ) -> None:
        """Send an RTC bootstrap/replan request to the mapped rollout worker."""
        assert mode == "eval", "RTC requests are only supported in eval mode."
        self.send_to(
            group_name=self.cfg.rollout.group_name,
            channel=rollout_channel,
            data=rtc_request,
            mode=mode,
            tag="rtc",
            route_key=0,
            batch_size=self.cfg.env.eval.total_num_envs,
            split_fn=lambda data, sizes: [data],
        )

    def recv_rtc_response(
        self, input_channel: Channel, mode: str = "eval", async_op: bool = False
    ):
        """Receive an RTC action response from the mapped rollout worker."""
        assert mode == "eval", "RTC responses are only supported in eval mode."
        work = self.recv_from(
            group_name=self.cfg.rollout.group_name,
            channel=input_channel,
            tag=f"{mode}_rtc",
            route_key=0,
            async_op=True,
            batch_size=self.cfg.env.eval.total_num_envs,
            merge_fn=lambda items: items[0],
            infer_batch_size_fn=lambda data: 1,
        )
        if async_op:
            return work
        return work.wait()

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
        rewrite_chunk_gripper = bool(self.model_cfg.get("rewrite_chunk_gripper", False))
        if not rewrite_chunk_gripper:
            return chunk_actions

        # Keep the gripper command stable within a chunk. If the policy predicts
        # fewer than two open commands, close for the whole chunk; otherwiseopen.
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
            self.eval_env_list[stage_id].step(env_action)
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
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
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

        for eval_rollout_epoch in range(self.eval_rollout_epoch):
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
                if self.eval_chunk_pause_seconds > 0:
                    step_start = time.time()

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

                for key, value in env_info.items():
                    eval_metrics[key].append(value)
                episode_success = episode_success or self._success_from_info(env_info)

                episode_step += 1
                current_chunk_index += 1

                episode_done = env_output.dones is not None and bool(
                    env_output.dones.any()
                )

                if self.eval_chunk_pause_seconds > 0:
                    step_elapsed = time.time() - step_start
                    remaining = self.eval_chunk_pause_seconds - step_elapsed
                    if remaining > 0:
                        time.sleep(remaining)

                if episode_done:
                    break

            if pending_rtc_response is not None:
                pending_rtc_response.wait()
                pending_rtc_response = None

            if not episode_done:
                existing_keys = set(eval_metrics.keys())
                default_values = {
                    "success_once": 0.0,
                    "return": 0.0,
                    "episode_len": float(episode_step),
                    "reward": 0.0,
                    "intervened_once": 0.0,
                    "intervened_steps": 0.0,
                    "success_no_intervened": 0.0,
                }
                for key in existing_keys:
                    value = default_values.get(key, 0.0)
                    eval_metrics[key].append(torch.tensor([value], dtype=torch.float32))
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
