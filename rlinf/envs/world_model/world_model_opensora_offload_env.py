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

from rlinf.envs.env_manager import (
    EnvOffloadMixin,
    recursive_to_device,
)
from rlinf.envs.world_model.world_model_opensora_env import OpenSoraEnv

__all__ = ["OpenSoraOffloadEnv"]


class OpenSoraOffloadEnv(OpenSoraEnv, EnvOffloadMixin):
    def get_state(self) -> bytes:
        """Serialize environment state to bytes buffer"""
        # Collect all state that needs to be saved
        env_state = {
            "current_obs": recursive_to_device(self.current_obs, "cpu")
            if self.current_obs is not None
            else None,
            "task_descriptions": self.task_descriptions,
            "init_ee_poses": self.init_ee_poses,
            "elapsed_steps": self.elapsed_steps,
            "prev_step_reward": self.prev_step_reward.cpu(),
            "_is_start": self._is_start,
            "video_cnt": self.video_cnt,
            "render_images": self.render_images,
            "render_rgb": getattr(self, "render_rgb", None),
            "render_actions": getattr(self, "render_actions", None),
            "render_rewards": getattr(self, "render_rewards", None),
            "reset_state_ids": self.reset_state_ids.cpu(),
            "generator_state": self._generator.get_state(),
        }

        # Save image_queue (list of deques containing latent frames)
        image_queue_state = []
        for env_idx in range(self.num_envs):
            queue_frames = []
            for frame in self.image_queue[env_idx]:
                # frame is a tensor [1, C, 1, H', W']
                queue_frames.append(recursive_to_device(frame, "cpu"))
            image_queue_state.append(queue_frames)
        env_state["image_queue"] = image_queue_state

        # Save metrics if recording
        if self.record_metrics:
            env_state.update(
                {
                    "success_once": self.success_once.cpu(),
                    "returns": self.returns.cpu(),
                }
            )

        # Serialize to bytes
        buffer = io.BytesIO()
        torch.save(env_state, buffer)
        return buffer.getvalue()

    def load_state(self, state_buffer: bytes):
        """Load environment state from bytes buffer"""
        buffer = io.BytesIO(state_buffer)
        state = torch.load(buffer, map_location="cpu", weights_only=False)

        # Restore basic state
        self.current_obs = (
            recursive_to_device(state["current_obs"], self.device)
            if state["current_obs"] is not None
            else None
        )
        self.task_descriptions = state["task_descriptions"]
        self.init_ee_poses = state["init_ee_poses"]
        self.elapsed_steps = state["elapsed_steps"]
        self.prev_step_reward = state["prev_step_reward"].to(self.device)
        self._is_start = state["_is_start"]
        self.video_cnt = state["video_cnt"]
        self.render_images = state["render_images"]
        self.render_rgb = state.get("render_rgb", None)
        self.render_actions = state.get("render_actions", None)
        self.render_rewards = state.get("render_rewards", None)

        # Restore reset state management
        self.reset_state_ids = state["reset_state_ids"].to(self.device)
        self._generator.set_state(state["generator_state"])

        # Restore image_queue
        image_queue_state = state["image_queue"]
        for env_idx in range(self.num_envs):
            self.image_queue[env_idx].clear()
            for frame in image_queue_state[env_idx]:
                # frame is a tensor [1, C, 1, H', W']
                frame_device = recursive_to_device(frame, self.device)
                self.image_queue[env_idx].append(frame_device)

        # Restore metrics if recording
        if self.record_metrics and "success_once" in state:
            self.success_once = state["success_once"].to(self.device)
            self.returns = state["returns"].to(self.device)
