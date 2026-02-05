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

"""Environment wrappers for data collection and trajectory recording."""

import os
import pickle
import random
from typing import Any, Optional

import gymnasium as gym

from rlinf.utils.logging import get_logger

logger = get_logger()


class DataCollectorWrapper(gym.Wrapper):
    """Wrapper for collecting environment interaction data during rollouts.

    This wrapper transparently records observations, actions, rewards, termination
    flags, and info dicts at each step. Episode data is saved to separate pickle
    files when episodes terminate, supporting both single and vectorized environments.

    Attributes:
        save_dir: Directory path where episode data will be saved.
        enabled: Whether data collection is currently active.
        rank: Worker rank identifier for distributed training.
        collect_mode: Collection mode - "train", "eval", or "all".
        num_envs: Number of parallel environments being wrapped.

    Args:
        env: The gymnasium environment to wrap.
        save_dir: Directory path for saving collected episode data.
        enabled: Whether to enable data collection. Defaults to True.
        rank: Worker rank for file naming in distributed settings. Defaults to 0.
        mode: Collection mode - "train", "eval", or "all". Defaults to "all".
        num_envs: Number of parallel environments. Defaults to 1.
        show_goal_site: Whether to show goal visualization in renders. Defaults to True.
        sample_rate_success: Probability of saving successful episodes. Defaults to 1.0.
        sample_rate_fail: Probability of saving failed episodes. Defaults to 0.1.
    """

    def __init__(
        self,
        env: gym.Env,
        save_dir: str,
        enabled: bool = True,
        rank: int = 0,
        mode: str = "all",
        num_envs: int = 1,
        show_goal_site: bool = True,
        sample_rate_success: float = 1.0,
        sample_rate_fail: float = 0.1,
    ):
        super().__init__(env)
        self.save_dir = save_dir
        self.rank = rank
        self.collect_mode = mode
        self.num_envs = num_envs
        self.show_goal_site = show_goal_site
        self.sample_rate_success = sample_rate_success
        self.sample_rate_fail = sample_rate_fail

        self.env_mode = self._detect_env_mode()

        should_collect = (self.collect_mode == "all") or (
            self.collect_mode == self.env_mode
        )
        self.enabled = enabled and should_collect

        logger.info(
            f"DataCollectorWrapper initialized: env_mode={self.env_mode}, "
            f"collect_mode={self.collect_mode}, enabled={self.enabled}"
        )

        self._episode_ids = [0] * num_envs
        self._episode_success = [False] * num_envs
        self._next_episode_obs = [None] * num_envs
        self._buffers = [self._create_empty_buffer() for _ in range(num_envs)]

        if self.enabled:
            self._setup_save_dir()

    def _detect_env_mode(self) -> str:
        """Detect whether this environment is configured for training or evaluation.

        Traverses the wrapper chain to find configuration attributes that indicate
        the environment mode.

        Returns:
            "train" or "eval" based on detected configuration, defaults to "train".
        """
        current = self.env
        visited = set()

        while current is not None and id(current) not in visited:
            visited.add(id(current))

            if hasattr(current, "cfg"):
                cfg = current.cfg
                is_eval = getattr(cfg, "is_eval", None)
                if is_eval is not None:
                    return "eval" if is_eval else "train"

            if hasattr(current, "is_eval") and not callable(
                getattr(current, "is_eval")
            ):
                return "eval" if current.is_eval else "train"

            if hasattr(current, "env"):
                current = current.env
            elif hasattr(current, "unwrapped") and current.unwrapped is not current:
                current = current.unwrapped
            else:
                break

        return "train"

    def _create_empty_buffer(self) -> dict[str, list]:
        """Create an empty buffer for storing episode data.

        Returns:
            Dictionary with empty lists for observations, actions, rewards,
            terminated flags, truncated flags, and info dicts.
        """
        return {
            "observations": [],
            "actions": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
            "infos": [],
        }

    def _setup_save_dir(self):
        """Create the save directory if it does not exist."""
        os.makedirs(self.save_dir, exist_ok=True)

    def _show_goal_site_visual(self):
        """Make goal site visualization visible in rendered observations.

        Some environments hide goal indicators by default. This method attempts
        to unhide them for data collection with visual observations.
        """
        if not self.show_goal_site:
            return

        unwrapped = self.env
        while hasattr(unwrapped, "env"):
            unwrapped = unwrapped.env
        if hasattr(unwrapped, "unwrapped"):
            unwrapped = unwrapped.unwrapped

        if not hasattr(unwrapped, "goal_site"):
            return

        goal_site = unwrapped.goal_site

        if hasattr(unwrapped, "_hidden_objects"):
            while goal_site in unwrapped._hidden_objects:
                unwrapped._hidden_objects.remove(goal_site)

        if hasattr(goal_site, "show_visual"):
            goal_site.show_visual()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        """Reset the environment and initialize episode buffers.

        Args:
            seed: Optional random seed for environment reset.
            options: Optional dictionary of reset options.

        Returns:
            Tuple of (observation, info) from the underlying environment.
        """
        self._buffers = [self._create_empty_buffer() for _ in range(self.num_envs)]
        self._episode_success = [False] * self.num_envs

        try:
            obs, info = self.env.reset(seed=seed, options=options)
        except TypeError:
            obs, info = self.env.reset()

        self._show_goal_site_visual()

        if self.enabled:
            self._record_initial_obs(obs)

        return obs, info

    def _record_initial_obs(self, obs):
        """Record initial observation for each environment after reset.

        Args:
            obs: Observation returned from environment reset.
        """
        import torch

        if isinstance(obs, dict):
            for env_idx in range(self.num_envs):
                env_obs = {}
                for key, value in obs.items():
                    if (
                        isinstance(value, (torch.Tensor,))
                        and value.shape[0] == self.num_envs
                    ):
                        env_obs[key] = self._copy_data(value[env_idx])
                    elif isinstance(value, list) and len(value) == self.num_envs:
                        env_obs[key] = self._copy_data(value[env_idx])
                    else:
                        env_obs[key] = self._copy_data(value)
                self._buffers[env_idx]["observations"].append(env_obs)
        else:
            for env_idx in range(self.num_envs):
                self._buffers[env_idx]["observations"].append(
                    self._copy_data(
                        obs[env_idx] if obs.shape[0] == self.num_envs else obs
                    )
                )

    def step(self, action, **kwargs):
        """Execute a step and record the transition data.

        Args:
            action: Action to execute in the environment.
            **kwargs: Additional arguments passed to the underlying step.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)

        if self.enabled:
            self._record_step_data(action, obs, reward, terminated, truncated, info)
            self._check_and_flush_episodes(terminated, truncated)

        return obs, reward, terminated, truncated, info

    def chunk_step(self, chunk_actions):
        """Execute a chunk of actions for embodied control environments.

        Args:
            chunk_actions: Sequence of actions to execute as a chunk.

        Returns:
            Tuple of (obs_list, rewards, terminations, truncations, infos_list).
        """
        obs_list, rewards, terminations, truncations, infos_list = self.env.chunk_step(
            chunk_actions
        )

        if self.enabled:
            # Use the last obs and infos for recording (compatible with old interface)
            last_obs = obs_list[-1] if isinstance(obs_list, (list, tuple)) else obs_list
            last_infos = (
                infos_list[-1] if isinstance(infos_list, (list, tuple)) else infos_list
            )
            self._record_chunk_data(
                chunk_actions, last_obs, rewards, terminations, truncations, last_infos
            )
            self._check_and_flush_episodes(terminations, truncations)

        return obs_list, rewards, terminations, truncations, infos_list

    def _record_step_data(self, action, obs, reward, terminated, truncated, info):
        """Record transition data for each environment.

        Handles auto-reset by extracting final_observation from info when available.

        Args:
            action: Action taken.
            obs: Resulting observation (may be reset obs if auto-reset occurred).
            reward: Reward received.
            terminated: Termination flag.
            truncated: Truncation flag.
            info: Info dictionary (may contain final_observation).
        """

        for env_idx in range(self.num_envs):
            env_action = self._extract_env_data(action, env_idx)
            env_reward = self._extract_env_data(reward, env_idx)
            env_terminated = self._extract_env_data(terminated, env_idx)
            env_truncated = self._extract_env_data(truncated, env_idx)
            env_info = self._extract_env_data(info, env_idx)

            has_final_obs = isinstance(info, dict) and "final_observation" in info
            if has_final_obs:
                final_obs = info["final_observation"]
                env_obs = self._extract_env_data(final_obs, env_idx)
                self._next_episode_obs[env_idx] = self._extract_env_data(obs, env_idx)
            else:
                env_obs = self._extract_env_data(obs, env_idx)

            self._buffers[env_idx]["actions"].append(env_action)
            self._buffers[env_idx]["observations"].append(env_obs)
            self._buffers[env_idx]["rewards"].append(env_reward)
            self._buffers[env_idx]["terminated"].append(env_terminated)
            self._buffers[env_idx]["truncated"].append(env_truncated)
            self._buffers[env_idx]["infos"].append(env_info)

            self._update_success_status(env_idx, info)

    def _record_chunk_data(
        self, chunk_actions, obs, rewards, terminations, truncations, infos
    ):
        """Record chunk transition data for each environment.

        Args:
            chunk_actions: Actions executed in the chunk.
            obs: Resulting observation.
            rewards: Rewards received.
            terminations: Termination flags.
            truncations: Truncation flags.
            infos: Info dictionaries.
        """
        for env_idx in range(self.num_envs):
            env_actions = self._extract_env_data(chunk_actions, env_idx)
            env_rewards = self._extract_env_data(rewards, env_idx)
            env_terminated = self._extract_env_data(terminations, env_idx)
            env_truncated = self._extract_env_data(truncations, env_idx)
            env_info = self._extract_env_data(infos, env_idx)

            has_final_obs = isinstance(infos, dict) and "final_observation" in infos
            if has_final_obs:
                final_obs = infos["final_observation"]
                env_obs = self._extract_env_data(final_obs, env_idx)
                self._next_episode_obs[env_idx] = self._extract_env_data(obs, env_idx)
            else:
                env_obs = self._extract_env_data(obs, env_idx)

            self._buffers[env_idx]["actions"].append(env_actions)
            self._buffers[env_idx]["observations"].append(env_obs)
            self._buffers[env_idx]["rewards"].append(env_rewards)
            self._buffers[env_idx]["terminated"].append(env_terminated)
            self._buffers[env_idx]["truncated"].append(env_truncated)
            self._buffers[env_idx]["infos"].append(env_info)

            self._update_success_status(env_idx, env_info)

    def _update_success_status(self, env_idx: int, info):
        """Update episode success status based on info dictionary.

        Checks for success indicators in the following priority order:
        1. info["episode"]["success_once"] - cumulative success flag
        2. info["episode"]["success_at_end"] - end-of-episode success
        3. info["success"] - current step success

        Args:
            env_idx: Index of the environment to update.
            info: Info dictionary from the environment step.
        """
        import torch

        success = None

        if isinstance(info, dict) and "episode" in info:
            episode_info = info["episode"]
            if isinstance(episode_info, dict) and "success_once" in episode_info:
                success = episode_info["success_once"]

        if success is None and isinstance(info, dict) and "episode" in info:
            episode_info = info["episode"]
            if isinstance(episode_info, dict) and "success_at_end" in episode_info:
                success = episode_info["success_at_end"]

        if success is None and isinstance(info, dict) and "success" in info:
            success = info["success"]

        if success is not None:
            if isinstance(success, torch.Tensor):
                if success.dim() == 0:
                    self._episode_success[env_idx] = bool(success.item())
                elif len(success.shape) > 0 and success.shape[0] == self.num_envs:
                    self._episode_success[env_idx] = bool(success[env_idx].item())
            else:
                self._episode_success[env_idx] = bool(success)

    def _get_episode_success(self, buffer: dict, env_idx: int) -> bool:
        """Determine episode success from buffer data.

        Searches through info dictionaries to find success indicators,
        handling auto-reset scenarios where the final info may be in final_info.

        Args:
            buffer: Episode buffer containing recorded data.
            env_idx: Index of the environment.

        Returns:
            True if the episode was successful, False otherwise.
        """
        import torch

        if buffer["infos"]:
            last_info = buffer["infos"][-1]
            if isinstance(last_info, dict) and "final_info" in last_info:
                final_info = last_info["final_info"]
                if isinstance(final_info, dict) and "episode" in final_info:
                    episode = final_info["episode"]
                    if "success_once" in episode:
                        val = episode["success_once"]
                        return bool(
                            val.item() if isinstance(val, torch.Tensor) else val
                        )

        if len(buffer["infos"]) >= 2:
            prev_info = buffer["infos"][-2]
            if isinstance(prev_info, dict) and "episode" in prev_info:
                episode = prev_info["episode"]
                if "success_once" in episode:
                    val = episode["success_once"]
                    return bool(val.item() if isinstance(val, torch.Tensor) else val)

        for info in reversed(buffer["infos"]):
            if isinstance(info, dict) and "episode" in info:
                episode = info["episode"]
                if "success_once" in episode:
                    val = episode["success_once"]
                    return bool(val.item() if isinstance(val, torch.Tensor) else val)

        return self._episode_success[env_idx]

    def _extract_env_data(self, data, env_idx: int):
        """Extract data for a specific environment from batched data.

        Args:
            data: Potentially batched data (tensor, array, dict, or list).
            env_idx: Index of the environment to extract.

        Returns:
            Data corresponding to the specified environment index.
        """
        import numpy as np
        import torch

        if isinstance(data, torch.Tensor):
            if data.shape[0] == self.num_envs:
                return self._copy_data(data[env_idx])
            return self._copy_data(data)
        elif isinstance(data, np.ndarray):
            if data.shape[0] == self.num_envs:
                return self._copy_data(data[env_idx])
            return self._copy_data(data)
        elif isinstance(data, dict):
            return {k: self._extract_env_data(v, env_idx) for k, v in data.items()}
        elif isinstance(data, list):
            if len(data) == self.num_envs:
                return self._copy_data(data[env_idx])
            return self._copy_data(data)
        else:
            return self._copy_data(data)

    def _check_and_flush_episodes(self, terminated, truncated):
        """Check termination status and save completed episodes.

        Args:
            terminated: Termination flags for each environment.
            truncated: Truncation flags for each environment.
        """
        import torch

        for env_idx in range(self.num_envs):
            if isinstance(terminated, torch.Tensor):
                if terminated.dim() > 1:
                    env_term = terminated[env_idx, -1].item()
                    env_trunc = truncated[env_idx, -1].item()
                else:
                    env_term = terminated[env_idx].item()
                    env_trunc = truncated[env_idx].item()
            else:
                env_term = bool(terminated)
                env_trunc = bool(truncated)

            if env_term or env_trunc:
                self._flush_env_episode(env_idx)

    def _flush_env_episode(self, env_idx: int):
        """Save episode data to disk and reset the buffer.

        Applies sampling rate filtering based on episode success status.

        Args:
            env_idx: Index of the environment whose episode to save.
        """
        buffer = self._buffers[env_idx]
        if not buffer["actions"]:
            return

        is_success = self._get_episode_success(buffer, env_idx)
        sample_rate = self.sample_rate_success if is_success else self.sample_rate_fail

        if random.random() < sample_rate:
            label = "success" if is_success else "fail"
            filename = (
                f"rank_{self.rank}_env_{env_idx}_"
                f"episode_{self._episode_ids[env_idx]}_{label}.pkl"
            )
            save_path = os.path.join(self.save_dir, filename)

            episode_data = {
                "mode": self.env_mode,
                "rank": self.rank,
                "env_idx": env_idx,
                "episode_id": self._episode_ids[env_idx],
                "success": is_success,
                "observations": buffer["observations"],
                "actions": buffer["actions"],
                "rewards": buffer["rewards"],
                "terminated": buffer["terminated"],
                "truncated": buffer["truncated"],
                "infos": buffer["infos"],
            }

            with open(save_path, "wb") as f:
                pickle.dump(episode_data, f)

        self._episode_ids[env_idx] += 1
        self._buffers[env_idx] = self._create_empty_buffer()
        self._episode_success[env_idx] = False

        if self._next_episode_obs[env_idx] is not None:
            self._buffers[env_idx]["observations"].append(
                self._next_episode_obs[env_idx]
            )
            self._next_episode_obs[env_idx] = None

    def _copy_data(self, data):
        """Create a deep copy of data to avoid reference issues.

        Args:
            data: Data to copy (tensor, array, dict, list, tuple, or scalar).

        Returns:
            Deep copy of the input data.
        """
        import numpy as np
        import torch

        if isinstance(data, torch.Tensor):
            return data.detach().cpu().clone()
        elif isinstance(data, np.ndarray):
            return data.copy()
        elif isinstance(data, dict):
            return {k: self._copy_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._copy_data(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._copy_data(item) for item in data)
        else:
            return data

    def set_mode(self, mode: str):
        """Change the collection mode.

        Args:
            mode: New collection mode - "train", "eval", or "all".
        """
        self.collect_mode = mode
        should_collect = (self.collect_mode == "all") or (
            self.collect_mode == self.env_mode
        )
        self.enabled = should_collect

    def set_enabled(self, enabled: bool):
        """Enable or disable data collection.

        Args:
            enabled: Whether to enable data collection.
        """
        self.enabled = enabled
        if enabled:
            self._setup_save_dir()
