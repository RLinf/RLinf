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

import io

import torch

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv as BaseManiskillEnv
from rlinf.envs.offload_wrapper.base import (
    EnvOffloadMixin,
    get_batch_rng_state,
    recursive_to_device,
    set_batch_rng_state,
)


class ManiskillEnv(BaseManiskillEnv, EnvOffloadMixin):
    def get_state(self) -> bytes:
        env_state = self.env.get_state()
        rng_state = {
            "_main_rng": self.env.unwrapped._main_rng,
            "_batched_main_rng": get_batch_rng_state(
                self.env.unwrapped._batched_main_rng
            ),
            "_main_seed": self.env.unwrapped._main_seed,
            "_episode_rng": self.env.unwrapped._episode_rng,
            "_batched_episode_rng": get_batch_rng_state(
                self.env.unwrapped._batched_episode_rng
            ),
            "_episode_seed": self.env.unwrapped._episode_seed,
        }
        action_space_state = {
            "action_space": self.env.unwrapped.action_space,
            "single_action_space": self.env.unwrapped.single_action_space,
            "_orig_single_action_space": self.env.unwrapped._orig_single_action_space,
        }
        physx_state = {
            "cuda_articulation_link_data": self.env.unwrapped.scene.px.cuda_articulation_link_data.torch().cpu(),
            "cuda_articulation_qacc": self.env.unwrapped.scene.px.cuda_articulation_qacc.torch().cpu(),
            "cuda_articulation_qf": self.env.unwrapped.scene.px.cuda_articulation_qf.torch().cpu(),
            "cuda_articulation_qpos": self.env.unwrapped.scene.px.cuda_articulation_qpos.torch().cpu(),
            "cuda_articulation_qvel": self.env.unwrapped.scene.px.cuda_articulation_qvel.torch().cpu(),
            "cuda_articulation_target_qpos": self.env.unwrapped.scene.px.cuda_articulation_target_qpos.torch().cpu(),
            "cuda_articulation_target_qvel": self.env.unwrapped.scene.px.cuda_articulation_target_qvel.torch().cpu(),
            "cuda_rigid_body_data": self.env.unwrapped.scene.px.cuda_rigid_body_data.torch().cpu(),
            "cuda_rigid_dynamic_data": self.env.unwrapped.scene.px.cuda_rigid_dynamic_data.torch().cpu(),
        }

        user_defined_task_reset_states = getattr(self.env, "task_reset_states", {})
        for key, value in user_defined_task_reset_states.items():
            if torch.is_tensor(value):
                user_defined_task_reset_states[key] = value.cpu()

        user_defined_task_metric_states = getattr(self.env, "task_metric_states", {})

        simulator_state = {
            "sim_state": env_state,
            "sim_timestep": self.env.unwrapped.scene.get_timestep(),
            "elapsed_steps": self.env.unwrapped._elapsed_steps.cpu(),
            "rng_state": rng_state,
            "action_space_state": action_space_state,
            "prev_step_reward": self.prev_step_reward.cpu(),
            "reset_state_ids": self.reset_state_ids.cpu(),
            "generator_state": self._generator.get_state(),
            "is_start": self.is_start,
            "video_cnt": self.video_cnt,
            "_init_raw_obs": self.env.unwrapped._init_raw_obs,
            "agent_controller_state": recursive_to_device(
                self.env.agent.controller.get_state(), "cpu"
            ),
            "physx_state": physx_state,
            "user_defined_task_reset_states": user_defined_task_reset_states,
            "user_defined_task_metric_states": recursive_to_device(
                user_defined_task_metric_states, "cpu"
            ),
        }

        if self.record_metrics:
            simulator_state.update(
                {
                    "success_once": self.success_once.cpu(),
                    "fail_once": self.fail_once.cpu(),
                    "returns": self.returns.cpu(),
                }
            )
        buffer = io.BytesIO()
        torch.save(simulator_state, buffer)

        # force refresh GPU state
        self.env.unwrapped.scene._gpu_apply_all()
        self.env.unwrapped.scene.px.gpu_update_articulation_kinematics()
        self.env.unwrapped.scene._gpu_fetch_all()

        return buffer.getvalue()

    def load_state(self, state_buffer: bytes):
        """Load simulator state from bytes buffer"""

        buffer = io.BytesIO(state_buffer)
        state = torch.load(buffer, map_location="cpu", weights_only=False)

        options = {
            "reconfigure": False,
        }
        for key, value in state["user_defined_task_reset_states"].items():
            options[key] = value.to(self.env.device)

        self.env.reset(seed=self.seed, options=options)

        self.env.set_state(state["sim_state"])

        self.env.unwrapped.scene.set_timestep(state["sim_timestep"])
        self.env.unwrapped._elapsed_steps = state["elapsed_steps"].to(
            self.env.unwrapped.device
        )
        controller_state = recursive_to_device(
            state["agent_controller_state"], self.env.device
        )
        self.env.agent.controller.set_state(controller_state)

        # Restore RNG state
        rng_state = state["rng_state"]
        self.env.unwrapped._main_rng = rng_state["_main_rng"]
        self.env.unwrapped._batched_main_rng = set_batch_rng_state(
            rng_state["_batched_main_rng"]
        )
        self.env.unwrapped._main_seed = rng_state["_main_seed"]
        self.env.unwrapped._episode_rng = rng_state["_episode_rng"]
        self.env.unwrapped._batched_episode_rng = set_batch_rng_state(
            rng_state["_batched_episode_rng"]
        )
        self.env.unwrapped._episode_seed = rng_state["_episode_seed"]

        # Restore simulator task state
        self.prev_step_reward = state["prev_step_reward"].to(self.device)
        self.reset_state_ids = state["reset_state_ids"].to(self.device)
        self._generator.set_state(state["generator_state"])
        self.is_start = state["is_start"]

        task_metric_states = recursive_to_device(
            state["user_defined_task_metric_states"], self.device
        )
        for key, value in task_metric_states.items():
            setattr(self.env, key, value)

        # Restore utils state
        self.video_cnt = state["video_cnt"]

        if self.record_metrics and "success_once" in state:
            self.success_once = state["success_once"].to(self.device)
            self.fail_once = state["fail_once"].to(self.device)
            self.returns = state["returns"].to(self.device)


__all__ = ["ManiskillEnv"]
