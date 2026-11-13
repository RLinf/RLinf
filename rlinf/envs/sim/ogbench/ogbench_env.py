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


from contextlib import contextmanager
from typing import Any

import gymnasium as gym
import numpy as np
import torch


class OGBenchEnv(gym.Env):
    """Evaluate independent OGBench environments through the EnvWorker chunk API."""

    def __init__(
        self,
        cfg: Any,
        num_envs: int = 1,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        worker_info: Any = None,
        record_metrics: bool = True,
    ):
        if num_envs < 1:
            raise ValueError(
                f"OGBench evaluation requires num_envs >= 1, got {num_envs}"
            )
        if not cfg.is_eval or cfg.auto_reset:
            raise ValueError(
                "OGBench supports offline evaluation with is_eval=true and auto_reset=false, "
                f"got is_eval={cfg.is_eval}, auto_reset={cfg.auto_reset}"
            )
        if num_envs > 1 and cfg.get("early_stop_on_done", False):
            raise ValueError(
                "OGBench evaluation with num_envs > 1 requires early_stop_on_done=false"
            )
        import ogbench

        kwargs = {"render_mode": "rgb_array"} if cfg.video_cfg.save_video else {}
        self.envs = [
            ogbench.make_env_and_datasets(cfg.task_name, env_only=True, **kwargs)
            for _ in range(num_envs)
        ]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.num_envs = num_envs
        self.seed = int(cfg.seed) + int(cfg.get("evaluation_seed_offset", 0))
        self.seed_offset = seed_offset
        self.seed_stride = total_num_processes * num_envs
        self.episode_index = 0
        self.max_episode_steps = int(cfg.max_steps_per_rollout_epoch)
        self.record_metrics = record_metrics
        self.is_start = True
        self.elapsed_steps = np.zeros(num_envs, dtype=np.int32)
        self.info_logging_keys = []
        self._closed = False

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        """Start one independent episode per environment."""
        observations = []
        self._rng_states = []
        for index, env in enumerate(self.envs):
            episode_seed = (
                self.seed
                + self.seed_offset * self.num_envs
                + index
                + self.episode_index * self.seed_stride
                if seed is None
                else seed + index
            )
            self._rng_states.append(np.random.RandomState(episode_seed).get_state())
            with self._numpy_rng(index):
                obs, _ = env.reset(seed=episode_seed, options=options)
            observations.append(obs)
        self.episode_index += 1
        self.elapsed_steps[:] = 0
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self.success = np.zeros(self.num_envs, dtype=np.float32)
        self.terminated = np.zeros(self.num_envs, dtype=bool)
        self.truncated = np.zeros(self.num_envs, dtype=bool)
        self.last_obs = np.stack(observations)
        self.is_start = False
        return self._observation(), {}

    def _observation(self) -> dict[str, torch.Tensor]:
        return {"states": torch.as_tensor(self.last_obs.copy(), dtype=torch.float32)}

    @contextmanager
    def _numpy_rng(self, index: int):
        # OGBench 1.2.1 uses NumPy's global RNG for maze resets and teleports.
        previous = np.random.get_state()
        np.random.set_state(self._rng_states[index])
        try:
            yield
        finally:
            self._rng_states[index] = np.random.get_state()
            np.random.set_state(previous)

    def step(
        self, actions: torch.Tensor | np.ndarray
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step active environments and hold each completed episode's final state."""
        actions = (
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions)
        )
        expected = (self.num_envs, self.action_space.shape[0])
        if actions.shape != expected:
            raise ValueError(
                f"Expected OGBench action shape {expected}, got {actions.shape}"
            )
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        for index, env in enumerate(self.envs):
            if self.terminated[index] or self.truncated[index]:
                continue
            with self._numpy_rng(index):
                obs, reward, terminated, truncated, info = env.step(
                    np.clip(
                        actions[index], self.action_space.low, self.action_space.high
                    )
                )
            self.last_obs[index] = obs
            rewards[index] = reward
            self.elapsed_steps[index] += 1
            self.terminated[index] = terminated
            self.truncated[index] = (
                truncated or self.elapsed_steps[index] >= self.max_episode_steps
            )
            self.episode_return[index] += reward
            self.success[index] = max(
                self.success[index], float(info.get("success", 0))
            )
        infos = {}
        if self.record_metrics:
            infos["episode"] = {
                "success": torch.as_tensor(self.success.copy()),
                "return": torch.as_tensor(self.episode_return.copy()),
                "episode_len": torch.as_tensor(
                    self.elapsed_steps.copy(), dtype=torch.float32
                ),
            }
        return (
            self._observation(),
            torch.as_tensor(rewards),
            torch.as_tensor(self.terminated.copy()),
            torch.as_tensor(self.truncated.copy()),
            infos,
        )

    def chunk_step(
        self, chunk_actions: torch.Tensor | np.ndarray
    ) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        """Execute [num_envs, chunk_length, action_dim] actions."""
        actions = (
            chunk_actions.detach().cpu().numpy()
            if isinstance(chunk_actions, torch.Tensor)
            else np.asarray(chunk_actions)
        )
        if (
            actions.ndim != 3
            or actions.shape[0] != self.num_envs
            or actions.shape[1] < 1
            or actions.shape[2] != self.action_space.shape[0]
        ):
            raise ValueError(
                f"Expected OGBench actions [{self.num_envs}, chunk_length >= 1, "
                f"{self.action_space.shape[0]}], got {actions.shape}"
            )
        observations, rewards, terminated, truncated, infos = zip(
            *(self.step(actions[:, index]) for index in range(actions.shape[1]))
        )
        return (
            list(observations),
            torch.stack(rewards, dim=1),
            torch.stack(terminated, dim=1),
            torch.stack(truncated, dim=1),
            list(infos),
        )

    def update_reset_state_ids(self) -> None:
        """Episode seeds advance on reset; no separate task sampler is needed."""

    def render(self) -> Any:
        """Return frames for the standard video wrapper."""
        frames = [env.render() for env in self.envs]
        return frames[0] if self.num_envs == 1 else frames

    def close(self) -> None:
        """Release every owned simulator once."""
        if not self._closed:
            for env in self.envs:
                env.close()
            self._closed = True
