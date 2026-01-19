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
        show_goal_site: Whether to show green goal site visualization.
    """

    def __init__(self, env: gym.Env, num_envs: int = 1, show_goal_site: bool = True):
        super().__init__(env)
        self.num_envs = num_envs
        self._unwrapped = self._get_unwrapped_env()
        self.show_goal_site = show_goal_site

        # Store main_images for reward model during rollout
        self.rollout_images = []

        # Store episode-ending images for terminal reward mode
        # This tracks the ACTUAL final observation for each env when episode ends
        self.episode_final_images = (
            None  # Shape: (num_envs, H, W, C) - latest final image per env
        )
        self.episode_final_step = (
            None  # Shape: (num_envs,) - which step each env's episode ended
        )

        # Show goal site visualization (green dot)
        self._show_goal_site_visual()

    def _get_unwrapped_env(self):
        """Get the innermost ManiSkill env."""
        unwrapped = self.env
        while hasattr(unwrapped, "env"):
            unwrapped = unwrapped.env
        if hasattr(unwrapped, "unwrapped"):
            unwrapped = unwrapped.unwrapped
        return unwrapped

    def _show_goal_site_visual(self):
        """Make goal site visualization visible (green dot for target position)."""
        if not self.show_goal_site:
            return

        env = self._unwrapped

        if not hasattr(env, "goal_site"):
            return

        goal_site = env.goal_site

        # Remove from hidden objects if present
        if hasattr(env, "_hidden_objects"):
            while goal_site in env._hidden_objects:
                env._hidden_objects.remove(goal_site)

        # Show visual if method exists
        if hasattr(goal_site, "show_visual"):
            goal_site.show_visual()

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
        self._show_goal_site_visual()  # Ensure goal site is visible after reset
        obs = self._replace_states(obs)
        return obs, info

    def step(self, action, **kwargs):
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)
        obs = self._replace_states(obs)
        return obs, reward, terminated, truncated, info

    def chunk_step(self, chunk_actions):
        import torch

        obs, rewards, terminations, truncations, infos = self.env.chunk_step(
            chunk_actions
        )
        obs = self._replace_states(obs)
        # Handle final_observation in infos (from auto_reset)
        if isinstance(infos, dict) and "final_observation" in infos:
            infos["final_observation"] = self._replace_states(
                infos["final_observation"]
            )

        # Collect images for reward model
        # We need to handle two scenarios:
        # 1. Per-step mode: collect every frame
        # 2. Terminal mode: collect the ACTUAL episode-ending frame for each env

        if isinstance(obs, dict) and "main_images" in obs:
            current_step = len(self.rollout_images)

            # Track episode-final images when episodes end (auto_reset)
            if (
                isinstance(infos, dict)
                and "final_observation" in infos
                and isinstance(infos["final_observation"], dict)
                and "main_images" in infos["final_observation"]
            ):
                reset_mask = infos.get("_final_observation", None)
                if reset_mask is not None and reset_mask.any():
                    final_images = infos["final_observation"]["main_images"]

                    # Initialize episode_final_images on first use
                    if self.episode_final_images is None:
                        self.episode_final_images = final_images.clone().cpu()
                        # Use -1 to indicate "not set yet" (0 could be a valid step)
                        self.episode_final_step = torch.full(
                            (self.num_envs,), -1, dtype=torch.long
                        )

                    # Update episode_final_images for envs that just finished
                    # reset_mask shape: (n_envs,) or (n_envs, chunk_size)
                    if reset_mask.dim() > 1:
                        # chunk_step returns dones with shape (n_envs, chunk_size)
                        done_mask = reset_mask.any(dim=-1)  # Any done in chunk
                    else:
                        done_mask = reset_mask

                    for env_idx in range(self.num_envs):
                        if done_mask[env_idx]:
                            self.episode_final_images[env_idx] = final_images[
                                env_idx
                            ].cpu()
                            self.episode_final_step[env_idx] = current_step

            # Always collect current obs for per-step mode
            self.rollout_images.append(obs["main_images"].cpu())

        return obs, rewards, terminations, truncations, infos

    def get_rollout_images(self):
        """Get collected main_images from the rollout.

        Returns:
            torch.Tensor: Stacked images with shape (n_steps, n_envs, H, W, C)
        """
        import torch

        if len(self.rollout_images) == 0:
            return None
        # Each item has shape (n_envs, H, W, C)
        # Stack to (n_steps, n_envs, H, W, C)
        return torch.stack(self.rollout_images, dim=0)

    def get_episode_final_images(self):
        """Get the actual episode-ending images for terminal reward mode.

        For each env, returns the image from when its episode actually ended,
        NOT the last collected image (which might be from a new episode after reset).

        Returns:
            torch.Tensor: Images with shape (n_envs, H, W, C) or None if no episodes ended.
        """
        if self.episode_final_images is None:
            # No episodes ended during this rollout, use last collected images
            if len(self.rollout_images) == 0:
                return None
            return self.rollout_images[-1]

        # For envs that had episodes end, use their final images
        # For envs that never ended (ran full rollout), use the last collected image
        if len(self.rollout_images) == 0:
            return self.episode_final_images

        result = self.rollout_images[-1].clone()

        # episode_final_step tracks which step each env's episode ended
        # If an env's episode ended (step >= 0 means it was set, -1 means not set), use episode_final_images
        for env_idx in range(self.num_envs):
            if (
                self.episode_final_step is not None
                and self.episode_final_step[env_idx] >= 0
            ):
                result[env_idx] = self.episode_final_images[env_idx]

        return result

    def get_episode_final_steps(self):
        """Get the step at which each env's episode ended.

        Returns:
            torch.Tensor: Step indices with shape (n_envs,), -1 means no episode ended.
        """
        import torch

        if self.episode_final_step is None:
            # No episodes ended, return all -1 (will use last step)
            return torch.full((self.num_envs,), -1, dtype=torch.long)

        return self.episode_final_step.clone()

    def clear_rollout_images(self):
        """Clear collected images (call at start of each rollout)."""
        self.rollout_images = []
        self.episode_final_images = None
        self.episode_final_step = None
