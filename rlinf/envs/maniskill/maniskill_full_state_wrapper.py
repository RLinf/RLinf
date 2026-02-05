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

import gymnasium as gym


class ManiskillFullStateWrapper(gym.Wrapper):
    """Wrapper that replaces partial states with full states in rgb mode.

    In rgb mode, ManiSkill returns partial state in obs["states"]. This wrapper
    replaces it with full state by querying robot and object poses from the env.

    Args:
        env: The ManiSkill environment to wrap.
        num_envs: Number of parallel environments.
    """

    def __init__(self, env: gym.Env, num_envs: int = 1):
        super().__init__(env)
        self.num_envs = num_envs
        self._unwrapped = self._get_unwrapped_env()

    def _get_unwrapped_env(self):
        """Get the innermost ManiSkill env."""
        unwrapped = self.env
        while hasattr(unwrapped, "env"):
            unwrapped = unwrapped.env
        if hasattr(unwrapped, "unwrapped"):
            unwrapped = unwrapped.unwrapped
        return unwrapped

    def _get_full_state(self):
        """Get full state observation by temporarily switching to state mode."""
        from mani_skill.utils import common

        env = self._unwrapped
        original_mode = env._obs_mode
        env._obs_mode = "state"
        try:
            state_obs = env.get_obs()
        finally:
            env._obs_mode = original_mode

        if isinstance(state_obs, dict):
            return common.flatten_state_dict(
                state_obs, use_torch=True, device=env.device
            )
        return state_obs

    def _replace_states(self, obs):
        """Replace partial states with full states."""

        if not isinstance(obs, dict):
            return obs

        # Only process if this is rgb mode (has main_images)
        if "main_images" not in obs:
            return obs

        # Get full state and replace
        full_state = self._get_full_state()
        obs["states"] = full_state

        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._replace_states(obs)
        return obs, info

    def step(self, action, **kwargs):
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)
        obs = self._replace_states(obs)
        return obs, reward, terminated, truncated, info

    def chunk_step(self, chunk_actions):
        obs_list, rewards, terminations, truncations, infos_list = self.env.chunk_step(
            chunk_actions
        )
        # Process obs_list - replace states in each observation
        if isinstance(obs_list, (list, tuple)):
            obs_list = [self._replace_states(obs) for obs in obs_list]
        else:
            obs_list = self._replace_states(obs_list)

        # Handle final_observation in the last infos (from auto_reset)
        if isinstance(infos_list, (list, tuple)) and len(infos_list) > 0:
            last_infos = infos_list[-1]
            if isinstance(last_infos, dict) and "final_observation" in last_infos:
                last_infos["final_observation"] = self._replace_states(
                    last_infos["final_observation"]
                )
        elif isinstance(infos_list, dict) and "final_observation" in infos_list:
            infos_list["final_observation"] = self._replace_states(
                infos_list["final_observation"]
            )
        return obs_list, rewards, terminations, truncations, infos_list
