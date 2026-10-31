# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Episode histories and GRPO groups over the VLN-CE HTTP server."""

from __future__ import annotations

import copy
from dataclasses import asdict
from uuid import uuid4

import numpy as np
import torch


class VLNCEServerEnv:
    """One RLinf chunk is one server turn, containing up to three actions."""

    def __init__(
        self, cfg, num_envs, seed_offset, total_num_processes, worker_info=None
    ):
        from transformers import AutoTokenizer
        from vlnce_server.client import BatchEnvClient
        from vlnce_server.env_config import VLNCEEnvConfig

        self.worker_info = worker_info
        self.cfg = cfg
        self.num_envs = num_envs
        self.group_size = cfg.group_size
        if self.group_size < 1 or num_envs % self.group_size or not cfg.episode_ids:
            raise ValueError(
                "VLN-CE requires complete GRPO groups and nonempty episode_ids"
            )
        if cfg.auto_reset:
            raise ValueError("VLN-CE GRPO requires auto_reset=False")
        self.client = BatchEnvClient(cfg.server_url, timeout=cfg.get("timeout", 600))
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_path, trust_remote_code=False
        )
        self._config_type = VLNCEEnvConfig
        self._ids = [str(uuid4()) for _ in range(num_envs)]
        self._created = False
        self._has_reset = False
        self._cursor = seed_offset * (num_envs // self.group_size)
        self._reset_stride = total_num_processes * (num_envs // self.group_size)
        self._elapsed_steps = np.zeros(num_envs, dtype=np.int64)
        self._done = np.zeros(num_envs, dtype=bool)
        self._success = torch.zeros(num_envs, dtype=torch.bool)
        self._returns = torch.zeros(num_envs)
        self._is_start = True
        self._messages = [[] for _ in range(num_envs)]
        self._images = [[] for _ in range(num_envs)]
        self._task_descriptions = [""] * num_envs

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def info_logging_keys(self):
        return []

    def update_reset_state_ids(self):
        if self._has_reset:
            self._cursor += self._reset_stride

    def _append_observation(self, row, observation):
        text = observation["obs_str"]
        images = observation["multi_modal_data"]["<image>"]
        parts = text.split("<image>")
        if len(parts) - 1 != len(images):
            raise ValueError("VLN-CE server image placeholders and images do not match")
        content = []
        for index, part in enumerate(parts):
            if part:
                content.append({"type": "text", "text": part})
            if index < len(images):
                content.append({"type": "image"})
        self._messages[row].append({"role": "user", "content": content})
        self._images[row].extend(images)

    def _obs(self):
        return {
            "messages": copy.deepcopy(self._messages),
            "images": [list(x) for x in self._images],
            "task_descriptions": list(self._task_descriptions),
        }

    def reset(self, env_idx=None):
        if env_idx is not None and list(env_idx) != list(range(self.num_envs)):
            raise ValueError("VLN-CE resets complete GRPO groups together")
        if self._created:
            self.client.close_batch(self._ids)
            self._created = False
        configs = {}
        for index, env_id in enumerate(self._ids):
            episode = self.cfg.episode_ids[
                (self._cursor + index // self.group_size) % len(self.cfg.episode_ids)
            ]
            config = self._config_type(
                episode_id=int(episode),
                data_source=self.cfg.data_source,
                action_space=self.cfg.action_space,
                prompt_format="no_think_no_tag",
                max_actions_per_step=3,
                step_budget=self.cfg.step_budget,
                turn_budget=self.cfg.max_episode_steps,
                reward_type=self.cfg.get("server_reward_type", "success"),
                success_reward_base=self.cfg.reward_coef,
                save_as_video=False,
            )
            configs[env_id] = {"env_name": "vlnce", "env_config": asdict(config)}
        try:
            self.client.create_environments_batch(configs)
            self._created = True
            observations = self.client.reset_batch(
                dict.fromkeys(self._ids, self.cfg.seed)
            )
            prompts = self.client.get_system_prompts_batch(self._ids)
            for index, env_id in enumerate(self._ids):
                self._messages[index] = [{"role": "system", "content": prompts[env_id]}]
                self._images[index] = []
                history, _ = observations[env_id]
                self._task_descriptions[index] = history[0]["obs_str"]
                for turn, observation in enumerate(history):
                    if turn:
                        self._messages[index].append(
                            {"role": "assistant", "content": observation["response"]}
                        )
                    self._append_observation(index, observation)
        except Exception:
            self.client.close_batch(self._ids)
            self._created = False
            raise
        self._elapsed_steps.fill(0)
        self._done.fill(False)
        self._success.zero_()
        self._returns.zero_()
        self._has_reset = True
        return self._obs(), {}

    def chunk_step(self, chunk_actions):
        if tuple(chunk_actions.shape[:2]) != (self.num_envs, 1):
            raise ValueError(
                "VLN-CE expects [num_envs, 1, max_new_tokens] token actions"
            )
        responses = self.tokenizer.batch_decode(
            chunk_actions[:, 0], skip_special_tokens=True
        )
        active = [index for index in range(self.num_envs) if not self._done[index]]
        results = (
            self.client.step_batch({self._ids[i]: responses[i] for i in active})
            if active
            else {}
        )
        rewards = torch.zeros(self.num_envs, 1)
        terminated = torch.from_numpy(self._done.copy())[:, None]
        truncated = torch.zeros(self.num_envs, 1, dtype=torch.bool)
        for index in active:
            obs, reward, done, info = results[self._ids[index]]
            self._messages[index].append(
                {"role": "assistant", "content": responses[index]}
            )
            self._append_observation(index, obs)
            self._elapsed_steps[index] += 1
            timeout = (
                self._elapsed_steps[index] >= self.cfg.max_episode_steps and not done
            )
            rewards[index, 0] = float(reward)
            self._returns[index] += float(reward)
            self._success[index] |= bool(info.get("task_success", False))
            terminated[index, 0] = bool(done)
            truncated[index, 0] = bool(timeout)
            self._done[index] = done or timeout
        infos = {
            "episode": {
                "success_once": self._success.clone(),
                "return": self._returns.clone(),
                "episode_length": torch.from_numpy(self._elapsed_steps.copy()),
            }
        }
        if self._done.all():
            self.close()
        return [self._obs()], rewards, terminated, truncated, [infos]

    def close(self):
        if self._created:
            self.client.close_batch(self._ids)
            self._created = False
