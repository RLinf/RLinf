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

"""Locomotion tasks from the Unitree Go2 benchmark."""

from functools import partial
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch

from ..isaaclab_env import IsaaclabProcessEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

FLAT = "MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0"
ROUGH = "MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0"
STAIRS = "MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0"


def terrain_base_height_l2(
    env: "ManagerBasedRLEnv",
    target_height: float,
    asset_cfg: "SceneEntityCfg",
    sensor_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Penalize squared base-height error over the ground seen by the Go2 scanner.

    Isaac Lab encodes a missed ray as three positive infinities. When an entire
    scan misses, the height term contributes zero for that environment.
    """
    root_pos = env.scene[asset_cfg.name].data.root_pos_w
    hits = env.scene[sensor_cfg.name].data.ray_hits_w
    hit_z = hits[..., 2]
    valid = torch.isfinite(hits).all(dim=-1)
    root_valid = torch.isfinite(root_pos).all()
    # Use Isaac Lab's reduction order on the strided height tensor for complete scans.
    target_z = target_height + torch.mean(hit_z, dim=1)
    penalty = torch.square(root_pos[:, 2] - target_z)
    if valid.all() & root_valid & torch.isfinite(penalty).all():
        return penalty
    if not root_valid:
        raise FloatingPointError("Nonfinite Go2 root position")
    missed = torch.isposinf(hits).all(dim=-1)
    if not (valid | missed).all():
        raise FloatingPointError(
            "Expected a finite terrain hit or a (+inf, +inf, +inf) ray miss"
        )
    count = valid.sum(dim=1)
    observed_z = hit_z.masked_fill(missed, 0.0).sum(dim=1) / count.clamp_min(1)
    target_z = torch.where(count < hit_z.shape[1], target_height + observed_z, target_z)
    target_z = torch.where(count == 0, root_pos[:, 2], target_z)
    penalty = torch.square(root_pos[:, 2] - target_z)
    if not torch.isfinite(penalty).all():
        raise FloatingPointError("Nonfinite Go2 terrain height penalty")
    return penalty


def _make_go2_env(params, seed: int, num_envs: int, max_episode_steps: int):
    from isaaclab.app import AppLauncher

    args = []
    if params.get("asset_root"):
        args.append(f"--/persistent/isaac/asset_root/cloud={params.asset_root}")
    if params.get("disable_kit_telemetry", False):
        args.append("--/app/extensions/excluded/0=omni.kit.telemetry")
    app = AppLauncher(headless=True, enable_cameras=False, kit_args=" ".join(args)).app
    try:
        from envs.registry import register_tasks
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_tasks.utils import load_cfg_from_registry

        class TerminalObservationEnv(ManagerBasedRLEnv):
            capture_terminal = False
            terminal_observation = None

            @property
            def max_episode_length(self):
                # Converting seconds back to steps in Isaac Lab can round the horizon up.
                return max_episode_steps

            def _reset_idx(self, env_ids):
                if self.capture_terminal and len(env_ids):
                    # Sensing the pre-reset state must preserve the rollout's RNG and history.
                    device = torch.device(self.device)
                    with torch.random.fork_rng(
                        devices=[] if device.type == "cpu" else [device.index],
                        device_type=device.type,
                    ):
                        obs = self.observation_manager.compute(update_history=False)
                        self.terminal_observation = obs["policy"].clone()
                return super()._reset_idx(env_ids)

        register_tasks()
        cfg = load_cfg_from_registry(params.id, "env_cfg_entry_point")
        cfg.rewards.base_height.func = terrain_base_height_l2
        cfg.seed = seed
        cfg.scene.num_envs = num_envs
        cfg.episode_length_s = max_episode_steps * (cfg.sim.dt * cfg.decimation)
        task = TerminalObservationEnv(cfg=cfg)
        return _Go2Task(task, params, num_envs), app
    except BaseException:
        app.close()
        raise


class _Go2Task:
    """Translate manager observations and episode boundaries inside the simulator."""

    def __init__(self, task, params, num_envs: int):
        self.task = task
        self.device = task.device
        self.task_id = params.id
        obs_dim = task.observation_manager.group_obs_dim["policy"]
        action_dim = task.action_manager.total_action_dim
        if tuple(obs_dim) != (235,) or action_dim != 12:
            raise ValueError(
                f"Expected Go2 obs=(235,), actions=12; got {obs_dim}, {action_dim}"
            )
        self.returns = torch.zeros(num_envs, device=self.device)
        self.lengths = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.initial_reset = True

    @staticmethod
    def _wrap_obs(obs):
        states = obs["policy"].clone()
        return {"states": states, "critic_states": states}

    def reset(self, seed=None, env_ids=None):
        obs, _ = self.task.reset(env_ids=env_ids)
        if self.initial_reset:
            ages = self.task.episode_length_buf
            ages.copy_(torch.randint_like(ages, high=self.task.max_episode_length))
            self.initial_reset = False
        idx = slice(None) if env_ids is None else env_ids
        self.returns[idx] = 0
        self.lengths[idx] = 0
        return self._wrap_obs(obs), {}

    def step(self, actions):
        actions = actions.to(self.device)
        if self.task_id == ROUGH:
            if not torch.isfinite(actions).all():
                raise FloatingPointError(
                    "Nonfinite Go2 rough-terrain action before simulator step"
                )
            actions = actions.clamp(-100.0, 100.0)

        self.task.terminal_observation = None
        self.task.capture_terminal = True
        try:
            obs, rewards, terminated, truncated, _ = self.task.step(actions)
        finally:
            self.task.capture_terminal = False
        terminated, truncated = terminated.clone(), truncated.clone()
        done = terminated | truncated
        self.returns += rewards
        self.lengths += 1
        episode = {"return": self.returns.clone(), "episode_len": self.lengths.clone()}
        info = {"episode": episode}
        if done.any():
            # Rollout-end value prediction uses the full batch, including continuing rows.
            final_states = torch.where(
                done[:, None], self.task.terminal_observation, obs["policy"]
            )
            info.update(
                final_observation={
                    "states": final_states,
                    "critic_states": final_states,
                },
                final_info={"episode": episode},
                _final_observation=done,
                _final_info=done,
            )
        self.returns[done] = 0
        self.lengths[done] = 0
        return self._wrap_obs(obs), rewards.clone(), terminated, truncated, info

    def close(self):
        self.task.close()


class Go2IsaacLabEnv(IsaaclabProcessEnv):
    """Run one benchmark instance with its own terrain RNG and native auto-reset."""

    def __init__(
        self, cfg, num_envs, seed_offset=0, total_num_processes=1, worker_info=None
    ):
        if seed_offset != 0 or total_num_processes != 1:
            raise ValueError(
                "Go2 locomotion requires one unpartitioned environment worker"
            )
        if not cfg.auto_reset or cfg.ignore_terminations:
            raise ValueError(
                "Go2 Isaac Lab tasks reset at every episode boundary; set "
                "auto_reset=true and ignore_terminations=false."
            )
        self.cfg = cfg
        self.num_envs = num_envs
        self.auto_reset = True
        self.is_start = True
        self.observation_space = gym.spaces.Dict(
            {
                name: gym.spaces.Box(-np.inf, np.inf, (235,), np.float32)
                for name in ("states", "critic_states")
            }
        )
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (12,), np.float32)
        super().__init__(timeout=180)

    def _make_env_function(self):
        return partial(
            _make_go2_env,
            self.cfg.init_params,
            self.cfg.seed,
            self.num_envs,
            self.cfg.max_episode_steps,
        )

    def reset(self, env_idx=None):
        ids = None if env_idx is None else torch.as_tensor(env_idx, dtype=torch.long)
        result = self.env.reset(env_ids=ids)
        self.is_start = True
        return result

    def step(self, actions):
        result = self.env.step(actions)
        self.is_start = False
        return result

    def chunk_step(self, chunk_actions):
        """Apply one 12-joint action per policy step."""
        if tuple(chunk_actions.shape) != (self.num_envs, 1, 12):
            raise ValueError(
                f"Expected ({self.num_envs}, 1, 12) actions, got {tuple(chunk_actions.shape)}"
            )
        obs, rewards, terminated, truncated, info = self.step(chunk_actions[:, 0])
        return [obs], rewards[:, None], terminated[:, None], truncated[:, None], [info]
