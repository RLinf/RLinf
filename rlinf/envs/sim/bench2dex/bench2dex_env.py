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

"""RLinf transitions for configurable Bench2Dex tasks and embodiments."""

from __future__ import annotations

import copy
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.data.schema.embodied_types import JointPositionTargets
from rlinf.envs.sim.isaaclab.venv import SubProcIsaacLabEnv

from .evaluation import Bench2DexResults, normalize_profile
from .runtime import make_bench2dex_runtime
from .spec import Bench2DexSpec


class Bench2DexEnv(gym.Env):
    """Own one simulator process; actions are absolute active-joint targets."""

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info,
    ):
        if num_envs != 1:
            raise ValueError(
                f"Bench2Dex needs one scene per process, got num_envs={num_envs}"
            )
        if cfg.get("ignore_terminations", False):
            raise ValueError(
                "Use ignore_terminations=false; early_stop selects the Bench2Dex evaluation protocol"
            )
        self.cfg, self.num_envs = cfg, num_envs
        self.auto_reset = bool(cfg.auto_reset)
        self.reward_coef = float(cfg.get("reward_coef", 1.0))
        self.is_start = True
        self.policy_queries = 0
        self.query_at_success = None
        self._episode_recorded = False
        self.results = None
        self.device = torch.device("cpu")
        self._elapsed_steps = torch.zeros(1, dtype=torch.int32)
        params = OmegaConf.to_container(cfg.init_params, resolve=True)
        self.env = SubProcIsaacLabEnv(
            partial(
                make_bench2dex_runtime,
                params,
                int(cfg.seed),
                seed_offset,
                total_num_processes,
            )
        )
        try:
            spec = Bench2DexSpec.from_config(params)
            self.spec = {
                name: getattr(spec, name)
                for name in (
                    "action_dim",
                    "full_dof",
                    "main_camera",
                    "extra_cameras",
                    "wrist_cameras",
                )
            }
            if cfg.get("is_eval", False) and params.get("output_dir"):
                profile = normalize_profile(
                    params.get("generalization_profile", "none")
                )
                channel = {"cov_only": "cov", "inv_only": "inv"}.get(profile, profile)
                self.results = Bench2DexResults(
                    spec.source_path,
                    Path(params["output_dir"]).expanduser().resolve(),
                    f"{params['policy_name']}:{channel}",
                )
            self.action_space = gym.spaces.Box(
                -np.inf, np.inf, shape=(self.spec["action_dim"],), dtype=np.float32
            )
        except BaseException:
            self.env.close()
            raise

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    def _wrap_obs(self, raw: dict) -> dict:
        """Return RLinf fields; OpenPI's repack_env_obs maps them to policy keys."""
        images = raw["images"]
        obs = {
            "main_images": torch.from_numpy(images[self.spec["main_camera"]]).unsqueeze(
                0
            ),
            "states": torch.from_numpy(raw["state"]).unsqueeze(0),
            "task_descriptions": [raw["instruction"]],
        }
        tactile = raw["tactile"]
        obs["extra_obs"] = {
            "bench2dex": {
                "full_joint_positions": torch.from_numpy(
                    raw["full_joint_positions"]
                ).unsqueeze(0),
                "active_joint_names": [list(raw["active_joint_names"])],
                "full_joint_names": [list(raw["full_joint_names"])],
                "robot_keys": [raw["robot_key"]],
                "tactile": {
                    "site_names": [list(tactile["site_names"])],
                    **{
                        key: torch.as_tensor(value).unsqueeze(0).clone()
                        for key, value in tactile.items()
                        if key != "site_names"
                    },
                },
            }
        }
        for field, cameras in (
            ("extra_view_images", self.spec["extra_cameras"]),
            ("wrist_images", self.spec["wrist_cameras"]),
        ):
            if cameras:
                obs[field] = torch.from_numpy(
                    np.stack([images[name] for name in cameras])
                ).unsqueeze(0)
        return obs

    def _wrap_info(self, raw: dict) -> dict:
        info = dict(raw)
        count = int(raw["control_steps"])
        self._elapsed_steps.fill_(count)
        # Episode metrics use the same reward scale as the emitted transitions.
        total = float(raw["episode_return"]) * self.reward_coef
        episode = {
            "success_once": torch.tensor([raw["success_once"]], dtype=torch.bool),
            "return": torch.tensor([total], dtype=torch.float32),
            "episode_len": self.elapsed_steps.clone(),
            "reward": torch.tensor([total / max(count, 1)], dtype=torch.float32),
        }
        for key, value in raw.get("metrics", {}).items():
            if isinstance(value, (bool, int, float)):
                episode[key] = torch.tensor([value])
        info["episode"] = episode
        return info

    def reset(self, seed: int | None = None, env_ids=None) -> tuple[dict, dict]:
        """Reset selected environment 0 using the next manifest episode."""
        if env_ids is not None:
            ids = torch.as_tensor(env_ids).cpu().reshape(-1)
            if ids.tolist() != [0]:
                raise ValueError(f"Expected env_ids=[0], got {ids.tolist()}")
        raw_obs, raw_info = self.env.reset(seed=seed)
        self.is_start = False
        self.policy_queries = 0
        self.query_at_success = None
        self._episode_recorded = False
        return self._wrap_obs(raw_obs), self._wrap_info(raw_info)

    def _step_once(self, command, *, new_query: bool):
        obs, reward, terminated, truncated, info = self.env.step(command)
        if new_query and info["action_steps"] > 0:
            self.policy_queries += 1
        if self.query_at_success is None and info["success_once"]:
            self.query_at_success = self.policy_queries
        if "metrics" in info:
            info = dict(info, metrics=dict(info["metrics"]))
            info["metrics"]["policy_query_count"] = self.policy_queries
            info["metrics"]["policy_queries_to_stable_success"] = self.query_at_success
        if (
            self.results is not None
            and not self._episode_recorded
            and (terminated or truncated)
        ):
            self.results.append(info, self.policy_queries, self.query_at_success)
            self._episode_recorded = True
        return (
            self._wrap_obs(obs),
            torch.tensor([reward * self.reward_coef], dtype=torch.float32),
            torch.tensor([terminated], dtype=torch.bool),
            torch.tensor([truncated], dtype=torch.bool),
            self._wrap_info(info),
        )

    def step(self, actions, auto_reset: bool = True) -> tuple:
        """Execute one control action through SubProcIsaacLabEnv.step."""
        packet = actions if isinstance(actions, JointPositionTargets) else None
        positions = (
            torch.as_tensor(packet.joint_positions if packet is not None else actions)
            .detach()
            .cpu()
            .numpy()
        )
        layout = packet.joint_layout if packet is not None else "active"
        if layout not in ("active", "full"):
            raise ValueError(f"Expected active or full layout, got {layout!r}")
        width = self.spec["action_dim"] if layout == "active" else self.spec["full_dof"]
        if positions.shape != (1, width):
            raise ValueError(f"Expected actions [1,{width}], got {positions.shape}")
        if self.is_start:
            self.reset()
        command = (
            positions[0]
            if packet is None
            else JointPositionTargets(positions, packet.joint_names, layout)
        )
        obs, reward, terminated, truncated, info = self._step_once(
            command, new_query=True
        )
        valid = info["action_steps"] > 0
        done = bool(terminated.any() or truncated.any())
        step_info = {
            "valid_action_mask": torch.tensor([[valid]], dtype=torch.bool),
            "executed_action_count": torch.tensor([int(valid)]),
            "terminal_index": 0 if done and valid else None,
        }
        info = dict(info, **step_info)
        if done and auto_reset and self.auto_reset:
            final_obs, final_info = obs, copy.deepcopy(info)
            obs, reset_info = self.reset()
            info = dict(
                reset_info,
                final_observation=final_obs,
                final_info=final_info,
                _final_observation=torch.ones(1, dtype=torch.bool),
                _final_info=torch.ones(1, dtype=torch.bool),
                **step_info,
            )
        return obs, reward, terminated, truncated, info

    def chunk_step(self, actions) -> tuple:
        """Assemble the RLinf chunk protocol using only existing step calls."""
        packet = actions if isinstance(actions, JointPositionTargets) else None
        positions = (
            torch.as_tensor(packet.joint_positions if packet is not None else actions)
            .detach()
            .cpu()
            .numpy()
        )
        layout = packet.joint_layout if packet is not None else "active"
        if layout not in ("active", "full"):
            raise ValueError(f"Expected active or full layout, got {layout!r}")
        width = self.spec["action_dim"] if layout == "active" else self.spec["full_dof"]
        if (
            positions.ndim != 3
            or positions.shape[0] != 1
            or positions.shape[2] != width
            or positions.shape[1] < 1
        ):
            raise ValueError(
                f"Expected actions [1,C,{width}] with C>0, got {positions.shape}"
            )
        if self.is_start:
            self.reset()
        count = positions.shape[1]
        observations, infos = [], []
        rewards = torch.zeros((1, count), dtype=torch.float32)
        terminated = torch.zeros((1, count), dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        valid = torch.zeros_like(terminated)
        for i in range(count):
            command = (
                positions[0, i]
                if packet is None
                else JointPositionTargets(positions[:, i], packet.joint_names, layout)
            )
            obs, reward, term, trunc, info = self._step_once(command, new_query=i == 0)
            observations.append(obs)
            infos.append(info)
            rewards[:, i], terminated[:, i], truncated[:, i] = reward, term, trunc
            valid[0, i] = info["action_steps"] > 0
            if term.any() or trunc.any():
                break
        executed = len(observations)
        auto_reset = self.auto_reset
        done = bool(terminated.any() or truncated.any())
        while len(observations) < count:
            observations.append(observations[-1])
            infos.append(infos[-1])
            terminated[0, len(observations) - 1] = terminated[0, executed - 1]
            truncated[0, len(observations) - 1] = truncated[0, executed - 1]
        chunk_info = {
            "valid_action_mask": valid,
            "executed_action_count": valid.sum(dim=1),
            "terminal_index": executed - 1 if done and valid.any() else None,
        }
        infos[-1] = dict(infos[-1], **chunk_info)
        if done and auto_reset and self.auto_reset:
            final_obs, final_info = observations[-1], copy.deepcopy(infos[-1])
            observations[-1], reset_info = self.reset()
            infos[-1] = dict(
                reset_info,
                final_observation=final_obs,
                final_info=final_info,
                _final_observation=torch.ones(1, dtype=torch.bool),
                _final_info=torch.ones(1, dtype=torch.bool),
                **chunk_info,
            )
            terminated[:, -1] = terminated.any(dim=1)
            truncated[:, -1] = truncated.any(dim=1)
            terminated[:, :-1] = False
            truncated[:, :-1] = False
        return observations, rewards, terminated, truncated, infos

    def update_reset_state_ids(self) -> None:
        """Episode selection advances only when reset consumes a manifest entry."""

    def close(self) -> None:
        self.env.close()
