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

"""DROID rollouts in Arena's Python interpreter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import gymnasium as gym
    from isaaclab.envs import ManagerBasedRLEnv


def _step_limit(env: ManagerBasedRLEnv, max_steps: int) -> torch.Tensor:
    return env.episode_length_buf >= max_steps


def _make_env(
    environment: str,
    environment_args: dict,
    num_envs: int,
    seed: int,
    device: str,
    max_episode_steps: int,
    language_instruction: str | None,
) -> gym.Env:
    import isaaclab_arena_environments
    from isaaclab.managers import DatasetExportMode, TerminationTermCfg
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_openpi.policy.droid_adapter import Pi0DroidAdapter

    path = Path(environment)
    if path.suffix in {".yaml", ".yml"}:
        if environment_args:
            raise ValueError(f"Set graph parameters in {path}, got {environment_args}")
        root = Path(isaaclab_arena_environments.__file__).resolve().parent.parent
        definition = ArenaEnvGraphSpec.from_yaml(root / path).to_arena_env(
            enable_cameras=True
        )
    else:
        registry = EnvironmentRegistry()
        factory = registry.get_component_by_name(environment)
        config = registry.get_environment_cfg_type(factory)
        definition = factory().build(config(enable_cameras=True, **environment_args))

    embodiment = definition.embodiment
    if embodiment is None or embodiment.registry_name != "droid_abs_joint_pos":
        raise ValueError(
            f"The DROID policy requires droid_abs_joint_pos, got "
            f"{embodiment.registry_name if embodiment else None!r} in {environment!r}"
        )
    embodiment.camera_config.fit_resolution(*Pi0DroidAdapter.target_image_size)
    builder = ArenaEnvBuilder(
        definition,
        ArenaEnvBuilderCfg(
            num_envs=num_envs,
            seed=seed,
            placement_seed=seed,
            device=device,
            language_instruction=language_instruction,
            env_cfg_override={"compute_final_obs": True},
        ),
    )
    cfg, kwargs = builder.build_env_cfg()
    cfg.num_rerenders_on_reset = max(1, cfg.num_rerenders_on_reset)
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    cfg.terminations.rlinf_step_limit = TerminationTermCfg(
        func=_step_limit, time_out=True, params={"max_steps": max_episode_steps}
    )
    return builder.make_registered(cfg, kwargs)


def _array(tensor: torch.Tensor) -> np.ndarray:
    # Isaac Lab reuses buffers on the next step, including on CPU.
    return tensor.detach().to(device="cpu", copy=True).numpy()


def _observation(
    raw: dict, rows: torch.Tensor | slice = slice(None)
) -> dict[str, np.ndarray]:
    cameras = raw["camera_obs"]
    state = raw["policy"]
    return {
        "main_images": _array(cameras["external_camera_rgb"][rows]),
        "wrist_images": _array(cameras["wrist_camera_rgb"][rows]),
        "states": _array(
            torch.cat((state["joint_pos"][rows], state["gripper_pos"][rows]), dim=-1)
        ),
    }


class ArenaRollout:
    """Track the episode assigned to each lane of an Arena vector environment."""

    def __init__(self, env: gym.Env, auto_reset: bool, all_frames: bool = False):
        self.env = env
        self.auto_reset = auto_reset
        self.all_frames = all_frames
        base = env.unwrapped
        if not auto_reset:
            base.configure_episode_limit(base.num_envs)
        self.num_envs = base.num_envs
        self.instruction = base.get_language_instruction()
        if not self.instruction:
            raise ValueError("DROID evaluation requires a nonempty task instruction")
        self.has_success = "success" in base.termination_manager.active_terms
        self.has_reward = bool(base.reward_manager.active_terms)
        self.done = np.zeros(self.num_envs, dtype=bool)
        self.success = np.zeros_like(self.done)
        self.length = np.zeros(self.num_envs, dtype=np.int64)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)
        self.obs: dict = {}

    def _policy_obs(self) -> dict:
        return {
            **self.obs,
            "task_descriptions": [self.instruction] * self.num_envs,
            "active_mask": ~self.done,
        }

    def reset(self, seed: int | None, options: dict | None = None) -> tuple[dict, dict]:
        """Start a new episode in every lane."""
        raw, _ = self.env.reset(seed=seed, options=options)
        self.obs = _observation(raw)
        self.done.fill(False)
        self.success.fill(False)
        self.length.fill(0)
        self.returns.fill(0)
        return self._policy_obs(), {}

    def _step(self, actions: torch.Tensor, capture: bool) -> tuple:
        active = ~self.done
        terminated = np.zeros_like(active)
        truncated = np.zeros_like(active)
        reward = np.zeros_like(self.returns)
        info = {}
        if active.any():
            base = self.env.unwrapped
            raw, native_reward, term, trunc, extras = self.env.step(actions)
            terminated = _array(term) & active
            # A task can finish exactly at its time limit. RLinf bootstraps
            # truncations, so a genuine terminal state takes precedence.
            truncated = _array(trunc) & active & ~terminated
            ended = terminated | truncated
            success = (
                _array(base.termination_manager.get_term("success")) & terminated
                if self.has_success
                else np.zeros_like(active)
            )
            if self.has_reward:
                reward = np.where(active, _array(native_reward), 0)
            else:
                reward = success.astype(np.float32)
            self.returns += reward
            self.success |= success
            self.length += active
            if ended.any():
                if self.auto_reset:
                    info["final_observation"] = {
                        **_observation(extras["final_obs"]),
                        "task_descriptions": [self.instruction] * self.num_envs,
                        "active_mask": np.ones_like(active),
                    }
                    info["_final_observation"] = ended
                else:
                    # Preserve only the newly finished rows, before native reset.
                    rows = torch.as_tensor(ended, device=term.device)
                    final = _observation(extras["final_obs"], rows)
                    self.obs = {key: value.copy() for key, value in self.obs.items()}
                    for key, value in final.items():
                        self.obs[key][ended] = value
            if capture:
                obs = _observation(raw)
                frozen = self.done | ended if not self.auto_reset else self.done
                if frozen.any():
                    for key in obs:
                        obs[key][frozen] = self.obs[key][frozen]
                self.obs = obs
            if not self.auto_reset:
                self.done |= ended

        episode = {
            "return": self.returns.copy(),
            "episode_len": self.length.copy(),
            "success_once": self.success.copy(),
        }
        info["episode"] = episode
        if self.auto_reset and (ended := terminated | truncated).any():
            info["final_info"] = {"episode": episode}
            self.returns[ended] = 0
            self.length[ended] = 0
            self.success[ended] = False
        return (
            self._policy_obs() if capture else None,
            reward,
            terminated,
            truncated,
            info,
        )

    def chunk_step(self, actions: np.ndarray) -> tuple:
        """Execute a policy chunk and preserve terminal frames and per-step rewards."""
        if not self.obs:
            raise RuntimeError("Call reset before stepping Isaac Lab-Arena")
        if (
            actions.ndim != 3
            or actions.shape[0] != self.num_envs
            or actions.shape[1] == 0
            or actions.shape[2] != 8
        ):
            raise ValueError(
                f"Expected [{self.num_envs}, positive horizon, 8] DROID actions, got {actions.shape}"
            )
        if self.auto_reset and actions.shape[1] != 1:
            raise ValueError(
                "Use num_action_chunks=1 with auto_reset=true so each new episode starts with a fresh policy action"
            )
        chunk = torch.as_tensor(
            actions, device=self.env.unwrapped.device, dtype=torch.float32
        )
        steps = []
        executed = []
        for i, action in enumerate(chunk.unbind(1)):
            executed.append(~self.done)
            steps.append(self._step(action, self.all_frames or i == chunk.shape[1] - 1))
        obs, rewards, terms, truncs, infos = zip(*steps, strict=True)
        infos[-1]["executed_mask"] = np.stack(executed, axis=1)
        return (
            list(obs) if self.all_frames else [obs[-1]],
            np.stack(rewards, axis=1),
            np.stack(terms, axis=1),
            np.stack(truncs, axis=1),
            list(infos) if self.all_frames else [infos[-1]],
        )
