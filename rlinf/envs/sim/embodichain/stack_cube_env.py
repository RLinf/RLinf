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

"""RLinf episode ownership for the native EmbodiChain Stack Cube task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from rlinf.envs.sim.embodichain.embodichain_env import (
    EmbodiChainEnv,
    _cfg_get,
    _clone_nested,
    _resolve_gym_config_path,
)


class StackCubeEnv(EmbodiChainEnv):
    """Run frozen Stack Cube scenes with synchronized reset and native rewards."""

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ) -> None:
        self._stack_ppo = False
        self._stack_requests = []
        self._stack_request_index = 0
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

    def _build_env(self):
        gym_config_path = _resolve_gym_config_path(
            str(_cfg_get(self.cfg, "gym_config_path"))
        )
        from rlinf.envs.sim.embodichain.stack_cube import (
            STACK_PROMPT,
            build_stack_cube_env,
        )

        protocol = _cfg_get(self.cfg, "stack_cube_protocol", "evaluation")
        if protocol not in {"evaluation", "isaaclab"}:
            raise ValueError("Stack Cube protocol must be evaluation or isaaclab.")
        self._stack_ppo = protocol == "isaaclab"
        if self.num_envs != 1 and not self._stack_ppo:
            raise ValueError(
                "Stack Cube frozen-manifest evaluation requires num_envs=1."
            )
        if self._stack_ppo and self.auto_reset and not self.ignore_terminations:
            raise ValueError(
                "Stack Cube PPO requires auto_reset=false; selective physical "
                "resets cannot safely clear contacts in the shared world."
            )
        if self.clip_actions:
            raise ValueError(
                "Stack Cube requires relative EEF control without clipping."
            )
        requests_path = _cfg_get(self.cfg, "request_manifest")
        if not requests_path:
            raise ValueError("Stack Cube requires a frozen request_manifest.")
        self._stack_requests = json.loads(Path(requests_path).read_text())["requests"]
        if not self._stack_requests:
            raise ValueError("Stack Cube request manifest must not be empty.")
        if self._stack_ppo and len(self._stack_requests) < self.num_envs:
            raise ValueError("Stack Cube needs at least one request per scene.")
        if self._stack_ppo:
            self._stack_request_index = self.seed_offset * self.num_envs
        self.task_prompt = STACK_PROMPT
        env = build_stack_cube_env(
            _cfg_get(self.cfg, "panda_urdf"),
            seed=self.seed,
            device=str(self.device),
            max_episode_steps=self.max_episode_steps,
            num_envs=self.num_envs,
            protocol=protocol,
            gym_config_path=gym_config_path,
            ignore_terminations=self.ignore_terminations,
        )
        env.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        return env

    def _wrap_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        return self.env.policy_observation(raw_obs)

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset a complete scene batch to its frozen requests."""
        options = {} if options is None else dict(options)
        env_idx = options.pop("env_idx", None)
        if self._stack_ppo:
            return self._reset_stack_cube_ppo(env_idx, options)
        if env_idx is not None and list(env_idx) != [0]:
            raise ValueError("Single-scene Stack Cube reset accepts only env_idx=[0].")
        request = options.pop("request", None)
        if request is None:
            request = self._stack_requests[
                self._stack_request_index % len(self._stack_requests)
            ]
            self._stack_request_index += 1
        raw_obs, infos = self.env.reset_request(
            request, initial_state=options.pop("initial_state", None)
        )
        self._reset_metrics()
        self._is_start = True
        return self._wrap_obs(raw_obs), infos

    def _reset_stack_cube_ppo(self, env_idx, options):
        """Reset synchronized scenes without advancing unrelated episodes."""
        if env_idx is not None and list(env_idx) != list(range(self.num_envs)):
            raise NotImplementedError(
                "Stack Cube supports synchronized full-batch reset only; "
                "partial reset would advance other scenes while clearing contacts."
            )
        requests = options.pop("requests", None)
        initial_states = options.pop("initial_states", None)
        if options:
            raise ValueError(
                f"Unsupported Stack Cube PPO reset options: {list(options)}"
            )
        if requests is None:
            start = self._stack_request_index
            requests = [
                self._stack_requests[(start + row) % len(self._stack_requests)]
                for row in range(self.num_envs)
            ]
            self._stack_request_index += self.num_envs * self.total_num_processes
        raw_obs, infos = self.env.reset_requests(
            requests, initial_states=initial_states
        )
        infos = dict(
            infos,
            reset_request_ids=[request["request_id"] for request in requests],
            reset_request_seeds=[request["seed"] for request in requests],
        )
        self._reset_metrics()
        self._is_start = True
        return self._wrap_obs(raw_obs), infos

    def step(
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        """Return native rewards, first-episode done signals, and physical metrics.

        Training retains the first success/drop and resets at rollout boundaries.
        Evaluation runs to its time limit, including any collapse after success.
        """
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0).repeat(self.num_envs, 1)
        actions = actions.reshape(self.num_envs, -1)
        if not torch.isfinite(actions).all():
            raise ValueError("Stack Cube actions must contain only finite values.")
        if (self._elapsed_steps >= self.max_episode_steps).any():
            raise RuntimeError("Reset Stack Cube after its terminal transition.")
        raw_obs, rewards, terminations, truncations, step_info = self.env.step_relative(
            actions
        )
        metrics = step_info["metrics"]["stack_metrics"]
        stable = step_info["metrics"]["stable_success"]
        self._elapsed_steps += 1
        step_metrics = {
            "success": stable,
            "stack_metrics": metrics,
            "rewards": step_info["rewards"],
        }
        if self._stack_ppo:
            step_metrics["fail"] = metrics["failure"]
        infos = self._record_metrics(rewards, step_metrics)
        infos["episode"]["success_at_end"] = stable.clone()
        if self._stack_ppo:
            infos["episode"]["baseline_success_at_end"] = metrics[
                "baseline_success"
            ].clone()
        infos["applied_action"] = actions.clone()
        observation = self._wrap_obs(raw_obs)
        dones = terminations | truncations if self._stack_ppo else truncations
        if dones.any() and (self._stack_ppo or self.auto_reset):
            infos.update(
                final_info=_clone_nested(infos),
                final_observation=_clone_nested(observation),
                _final_info=dones.clone(),
                _final_observation=dones.clone(),
            )
            if self.auto_reset:
                if not dones.all():
                    raise RuntimeError("Stack Cube cannot auto-reset a partial batch.")
                observation, _ = self.reset()
        self._is_start = False
        return observation, rewards, terminations, truncations, infos

    def chunk_step(self, chunk_actions: np.ndarray | torch.Tensor):
        """Preserve the first terminal observation at its action-chunk position."""
        return self._chunk_step(
            chunk_actions,
            collect_final_info=self.auto_reset or self._stack_ppo,
            preserve_step_dones=self._stack_ppo
            and not self.auto_reset
            and not self.ignore_terminations,
        )

    def update_reset_state_ids(self) -> None:
        """Rewind fixed cohorts while leaving training requests rotating."""
        if not self._stack_ppo or _cfg_get(
            self.cfg, "use_fixed_reset_state_ids", False
        ):
            self._stack_request_index = (
                self.seed_offset * self.num_envs if self._stack_ppo else 0
            )
