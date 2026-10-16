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
from rlinf.envs.venv import BaseVectorEnv
from rlinf.utils.nested_dict_process import cat_list_of_dict_tensor

from .evaluation import Bench2DexResults, normalize_profile
from .runtime import make_bench2dex_runtime
from .spec import Bench2DexSpec
from .venv import Bench2DexSubprocEnvWorker


class Bench2DexEnv(gym.Env):
    """Batch independent scene processes using RLinf's vector environment."""

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info,
    ):
        if num_envs < 1 or not 0 <= seed_offset < total_num_processes:
            raise ValueError(
                f"Expected positive num_envs and 0<=seed_offset<total_num_processes; "
                f"got {num_envs}, {seed_offset}, {total_num_processes}"
            )
        if cfg.get("ignore_terminations", False):
            raise ValueError(
                "Use ignore_terminations=false; early_stop selects the Bench2Dex evaluation protocol"
            )
        self.cfg, self.num_envs = cfg, num_envs
        self.auto_reset = bool(cfg.auto_reset)
        self.reward_coef = float(cfg.get("reward_coef", 1.0))
        self.is_start = True
        self.device = torch.device("cpu")
        self.policy_queries = np.zeros(num_envs, dtype=np.int64)
        self.episode_seed = np.zeros(num_envs, dtype=np.int64)
        self.query_at_success = [None] * num_envs
        self._episode_recorded = np.zeros(num_envs, dtype=bool)
        self._initialized = np.zeros(num_envs, dtype=bool)
        self._elapsed_steps = torch.zeros(num_envs, dtype=torch.int32)
        self._terminated = torch.zeros(num_envs, dtype=torch.bool)
        self._truncated = torch.zeros_like(self._terminated)
        self._observations = [None] * num_envs
        self._infos = [None] * num_envs
        self.results = None
        params = OmegaConf.to_container(cfg.init_params, resolve=True)
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
        self.action_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(spec.action_dim,), dtype=np.float32
        )
        env_fns = [
            partial(
                make_bench2dex_runtime,
                params,
                int(cfg.seed),
                seed_offset * num_envs + env_id,
                total_num_processes * num_envs,
            )
            for env_id in range(num_envs)
        ]
        workers = []

        def worker_fn(env_fn):
            worker = Bench2DexSubprocEnvWorker(env_fn)
            workers.append(worker)
            return worker

        try:
            self.env = BaseVectorEnv(env_fns, worker_fn)
            if cfg.get("is_eval", False) and params.get("output_dir"):
                profile = normalize_profile(
                    params.get("generalization_profile", "none")
                )
                channel = {"cov_only": "cov", "inv_only": "inv"}.get(profile, profile)
                output = Path(params["output_dir"]).expanduser().resolve()
                if total_num_processes > 1:
                    output = output / "shards" / f"{seed_offset:06d}"
                self.results = Bench2DexResults(
                    spec.source_path, output, f"{params['policy_name']}:{channel}"
                )
        except BaseException:
            for worker in workers:
                worker.close()
            raise

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    def _wrap_obs(self, raw: dict, env_id: int) -> dict:
        """Wrap one scene's observation with its own policy sampling context."""
        images = raw["images"]
        tactile = raw["tactile"]
        obs = {
            "main_images": torch.from_numpy(images[self.spec["main_camera"]]).unsqueeze(
                0
            ),
            "states": torch.from_numpy(raw["state"]).unsqueeze(0),
            "task_descriptions": [raw["instruction"]],
            "extra_obs": {
                "policy_sampling": {
                    "episode_seed": torch.tensor([self.episode_seed[env_id]]),
                    "query_index": torch.tensor([self.policy_queries[env_id]]),
                },
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
                },
            },
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

    def _batch_obs(self, env_ids) -> dict:
        return cat_list_of_dict_tensor(
            [self._wrap_obs(self._observations[i], i) for i in env_ids]
        )

    def _wrap_info(self, env_ids) -> dict:
        raw = [self._infos[i] for i in env_ids]
        info = {key: [item.get(key) for item in raw] for key in set().union(*raw)}
        counts = torch.tensor(
            [item["control_steps"] for item in raw], dtype=torch.int32
        )
        self._elapsed_steps[env_ids] = counts
        # Episode metrics use the same reward scale as the emitted transitions.
        totals = (
            torch.tensor([item["episode_return"] for item in raw], dtype=torch.float32)
            * self.reward_coef
        )
        episode = {
            "success_once": torch.tensor([item["success_once"] for item in raw]),
            "return": totals,
            "episode_len": counts,
            "reward": totals / counts.clamp_min(1),
        }
        metrics = [item.get("metrics", {}) for item in raw]
        for key in set().union(*metrics):
            values = [item.get(key, 0) for item in metrics]
            if all(isinstance(value, (bool, int, float)) for value in values):
                episode[key] = torch.tensor(values)
        info["episode"] = episode
        return info

    def reset(self, seed: int | None = None, env_ids=None) -> tuple[dict, dict]:
        """Reset selected scenes and return their batch in the requested order."""
        ids = (
            list(range(self.num_envs))
            if env_ids is None
            else torch.as_tensor(env_ids).cpu().reshape(-1).tolist()
        )
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(not isinstance(i, int) or not 0 <= i < self.num_envs for i in ids)
        ):
            raise ValueError(
                f"Expected unique env_ids in [0,{self.num_envs}), got {ids}"
            )
        observations, infos = self.env.reset(id=ids, seed=seed)
        for i, obs, info in zip(ids, observations, infos, strict=True):
            self._observations[i], self._infos[i] = obs, info
            self.episode_seed[i] = int(info["seed"])
            self.policy_queries[i] = 0
            self.query_at_success[i] = None
            self._episode_recorded[i] = False
            self._terminated[i] = self._truncated[i] = False
            self._initialized[i] = True
        self.is_start = not self._initialized.all()
        return self._batch_obs(ids), self._wrap_info(ids)

    def _step_batch(self, commands, env_ids, *, new_query: bool):
        rewards = torch.zeros(self.num_envs, dtype=torch.float32)
        valid = torch.zeros(self.num_envs, dtype=torch.bool)
        if len(env_ids):
            observations, raw_rewards, terminated, truncated, infos = self.env.step(
                commands, id=env_ids
            )
            for row, i in enumerate(env_ids):
                info = infos[row]
                self._observations[i], self._infos[i] = observations[row], info
                self._terminated[i], self._truncated[i] = (
                    bool(terminated[row]),
                    bool(truncated[row]),
                )
                rewards[i] = float(raw_rewards[row]) * self.reward_coef
                valid[i] = info["action_steps"] > 0
                if new_query and valid[i]:
                    self.policy_queries[i] += 1
                if self.query_at_success[i] is None and info["success_once"]:
                    self.query_at_success[i] = int(self.policy_queries[i])
                if "metrics" in info:
                    info["metrics"] = dict(
                        info["metrics"],
                        policy_query_count=int(self.policy_queries[i]),
                        policy_queries_to_stable_success=self.query_at_success[i],
                    )
                if (
                    self.results is not None
                    and not self._episode_recorded[i]
                    and (terminated[row] or truncated[row])
                ):
                    self.results.append(
                        info, int(self.policy_queries[i]), self.query_at_success[i]
                    )
                    self._episode_recorded[i] = True
        ids = list(range(self.num_envs))
        info = self._wrap_info(ids)
        info["action_steps"] = valid.to(torch.int64)
        return (
            self._batch_obs(ids),
            rewards,
            self._terminated.clone(),
            self._truncated.clone(),
            info,
            valid,
        )

    def step(self, actions, auto_reset: bool = True) -> tuple:
        """Execute absolute targets [N,J], then optionally reset completed scenes."""
        if isinstance(actions, JointPositionTargets):
            actions = JointPositionTargets(
                actions.joint_positions[:, None],
                actions.joint_names,
                actions.joint_layout,
            )
        else:
            actions = torch.as_tensor(actions).unsqueeze(1)
        obs, reward, terminated, truncated, info = self._execute_chunk(
            actions, auto_reset=auto_reset and self.auto_reset
        )
        return obs[-1], reward[:, 0], terminated[:, 0], truncated[:, 0], info[-1]

    def chunk_step(self, actions) -> tuple:
        """Return C post-action observation batches and outcome tensors [N,C]."""
        return self._execute_chunk(actions, auto_reset=self.auto_reset)

    def _execute_chunk(self, actions, *, auto_reset: bool) -> tuple:
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
            or positions.shape[0] != self.num_envs
            or positions.shape[2] != width
            or positions.shape[1] < 1
        ):
            raise ValueError(
                f"Expected actions [{self.num_envs},C,{width}] with C>0, got {positions.shape}"
            )
        if not np.isfinite(positions).all():
            raise ValueError("Expected finite Bench2Dex joint targets")
        if self.is_start:
            ids = np.flatnonzero(~self._initialized)
            self.reset(env_ids=ids if len(ids) else None)
        count = positions.shape[1]
        observations, infos = [], []
        rewards = torch.zeros((self.num_envs, count), dtype=torch.float32)
        terminated = torch.zeros((self.num_envs, count), dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        valid = torch.zeros_like(terminated)
        for t in range(count):
            ids = torch.where(~(self._terminated | self._truncated))[0].tolist()
            commands = [
                positions[i, t]
                if packet is None
                else JointPositionTargets(
                    positions[i : i + 1, t], packet.joint_names[i : i + 1], layout
                )
                for i in ids
            ]
            obs, reward, term, trunc, info, executed = self._step_batch(
                commands, ids, new_query=t == 0
            )
            observations.append(obs)
            infos.append(info)
            rewards[:, t], terminated[:, t], truncated[:, t], valid[:, t] = (
                reward,
                term,
                trunc,
                executed,
            )
        done = terminated.any(dim=1) | truncated.any(dim=1)
        lengths = valid.sum(dim=1)
        chunk_info = {
            "valid_action_mask": valid,
            "executed_action_count": lengths,
            "terminal_index": torch.where(done & (lengths > 0), lengths - 1, -1),
        }
        infos[-1] = dict(infos[-1], **chunk_info)
        if auto_reset and done.any():
            final_obs, final_info = observations[-1], copy.deepcopy(infos[-1])
            self.reset(env_ids=torch.where(done)[0])
            ids = list(range(self.num_envs))
            observations[-1] = self._batch_obs(ids)
            infos[-1] = dict(
                self._wrap_info(ids),
                final_observation=final_obs,
                final_info=final_info,
                _final_observation=done.clone(),
                _final_info=done.clone(),
                **chunk_info,
            )
            terminated[:, -1], truncated[:, -1] = (
                terminated.any(dim=1),
                truncated.any(dim=1),
            )
            terminated[:, :-1] = False
            truncated[:, :-1] = False
        return observations, rewards, terminated, truncated, infos

    def update_reset_state_ids(self) -> None:
        """Episode selection advances only when reset consumes a manifest entry."""

    def close(self) -> None:
        if not self.env.is_closed:
            self.env.close()
