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
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import cat_list_of_dict_tensor

from .evaluation import Bench2DexResults, normalize_profile
from .runtime import make_bench2dex_runtime
from .spec import Bench2DexSpec
from .venv import Bench2DexSubprocEnvWorker, Bench2DexVectorEnv


class Bench2DexEnv(gym.Env):
    """Expose batched transitions from independently owned simulator processes."""

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
        self.seed = int(cfg.seed) + seed_offset
        self.auto_reset = bool(cfg.auto_reset)
        self.skip_intermediate_renders = bool(
            cfg.get("skip_intermediate_renders", False)
        )
        self.group_size = int(cfg.get("group_size", 1))
        self.use_fixed_reset_state_ids = bool(cfg.use_fixed_reset_state_ids)
        if self.group_size < 1 or num_envs % self.group_size:
            raise ValueError(
                f"Expected num_envs={num_envs} divisible by positive "
                f"group_size, got {self.group_size}"
            )
        if self.group_size > 1 and (
            not self.use_fixed_reset_state_ids or self.auto_reset
        ):
            raise ValueError(
                "Grouped Bench2Dex rollouts require "
                "use_fixed_reset_state_ids=true and auto_reset=false"
            )
        self.reward_coef = float(cfg.get("reward_coef", 1.0))
        self.is_start = True
        self.device = torch.device("cpu")
        self.policy_queries = np.zeros(num_envs, dtype=np.int64)
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
        num_groups = num_envs // self.group_size
        self._next_reset_state_id = (
            int(params.get("start_episode", 1)) - 1 + seed_offset * num_groups
        )
        self._reset_state_stride = total_num_processes * num_groups
        self.reset_state_ids = None
        self.update_reset_state_ids()
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
        self.env = None
        self._closed = False
        self._episode_indices = None
        self._env_fns = [
            partial(
                make_bench2dex_runtime,
                params,
                int(cfg.seed),
                seed_offset * num_envs + env_id,
                total_num_processes * num_envs,
            )
            for env_id in range(num_envs)
        ]
        if not cfg.get("enable_offload", False):
            self._start_envs()
        if cfg.get("is_eval", False) and params.get("output_dir"):
            try:
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
                self.close()
                raise

    def _start_envs(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot restart a closed Bench2Dex environment")
        if self.env is not None:
            return
        workers = []

        def worker_fn(env_fn):
            worker = Bench2DexSubprocEnvWorker(env_fn)
            workers.append(worker)
            return worker

        try:
            self.env = Bench2DexVectorEnv(self._env_fns, worker_fn)
            if self._episode_indices is not None:
                for env_id, episode_index in enumerate(self._episode_indices):
                    self.env.set_env_attr("episode_index", episode_index, id=env_id)
            get_logger().info(
                "Bench2Dex simulator pool started: pids=%s",
                [worker.process.pid for worker in workers],
            )
        except BaseException:
            if self.env is not None:
                self.env.close()
            else:
                for worker in workers:
                    worker.close()
            self.env = None
            raise

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    def _wrap_obs(self, raw: dict) -> dict:
        """Convert one scene observation to the policy input format."""
        images = raw["images"]
        tactile = raw["tactile"]
        obs = {
            "main_images": torch.as_tensor(images[self.spec["main_camera"]]).unsqueeze(
                0
            ),
            "states": torch.as_tensor(raw["state"]).unsqueeze(0),
            "task_descriptions": [raw["instruction"]],
            "extra_obs": {
                "bench2dex": {
                    "full_joint_positions": torch.as_tensor(
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
                obs[field] = torch.stack(
                    [torch.as_tensor(images[name]) for name in cameras]
                ).unsqueeze(0)
        return obs

    def _batch_obs(self, env_ids) -> dict:
        return cat_list_of_dict_tensor(
            [self._wrap_obs(self._observations[i]) for i in env_ids]
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

    def reset(
        self, seed: int | None = None, env_ids=None, reset_state_ids=None
    ) -> tuple[dict, dict]:
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
        if reset_state_ids is None and self.use_fixed_reset_state_ids:
            reset_state_ids = self.reset_state_ids[ids]
        self._start_envs()
        observations, infos = self.env.reset(
            id=ids, seed=seed, reset_state_ids=reset_state_ids
        )
        for i, obs, info in zip(ids, observations, infos, strict=True):
            self._observations[i], self._infos[i] = obs, info
            self.policy_queries[i] = 0
            self.query_at_success[i] = None
            self._episode_recorded[i] = False
            self._terminated[i] = self._truncated[i] = False
            self._initialized[i] = True
        self.is_start = not self._initialized.all()
        return self._batch_obs(ids), self._wrap_info(ids)

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
        """Return post-action observations and outcome tensors [N,C].

        With ``skip_intermediate_renders``, return one observation batch at
        the chunk boundary; otherwise return C batches, one per action.
        """
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
        ids = torch.where(~(self._terminated | self._truncated))[0].tolist()
        commands = [
            positions[i]
            if packet is None
            else JointPositionTargets(
                positions[i : i + 1], packet.joint_names[i : i + 1], layout
            )
            for i in ids
        ]
        transitions = self.env.chunk_step(
            commands,
            env_ids=ids,
            capture_intermediate_observations=not self.skip_intermediate_renders,
        )
        all_ids = list(range(self.num_envs))
        for t in range(count):
            for i, scene_transitions in zip(ids, transitions, strict=True):
                if t >= len(scene_transitions):
                    continue
                obs, reward, term, trunc, info = scene_transitions[t]
                if obs is not None:
                    self._observations[i] = obs
                self._infos[i] = info
                self._terminated[i], self._truncated[i] = bool(term), bool(trunc)
                rewards[i, t] = float(reward) * self.reward_coef
                valid[i, t] = info["action_steps"] > 0
                if t == 0 and valid[i, t]:
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
                    and (term or trunc)
                ):
                    self.results.append(
                        info, int(self.policy_queries[i]), self.query_at_success[i]
                    )
                    self._episode_recorded[i] = True
            if not self.skip_intermediate_renders or t == count - 1:
                observations.append(self._batch_obs(all_ids))
            info = self._wrap_info(all_ids)
            info["action_steps"] = valid[:, t].to(torch.int64)
            infos.append(info)
            terminated[:, t] = self._terminated
            truncated[:, t] = self._truncated
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
        """Select the next scene per group for the following rollout."""
        if not self.use_fixed_reset_state_ids:
            return
        num_groups = self.num_envs // self.group_size
        self.reset_state_ids = np.repeat(
            np.arange(num_groups, dtype=np.int64) + self._next_reset_state_id,
            self.group_size,
        )
        self._next_reset_state_id += self._reset_state_stride

    def offload(self) -> None:
        """Release simulator GPU resources after all initialized episodes end.

        Close and join this environment's simulator children, retaining episode
        cursors and parent-side results on CPU. The next reset creates a new
        pool with the same shard assignments and next-episode indices.

        Raises:
            RuntimeError: An initialized episode is still in progress. Live
                physics state cannot be restored after process shutdown.
        """
        if self.env is None:
            return
        active = self._initialized & ~(
            self._terminated.numpy() | self._truncated.numpy()
        )
        if active.any():
            raise RuntimeError(
                "Bench2Dex offload requires completed episodes; "
                "use auto_reset=false and a full episode rollout budget"
            )
        self._episode_indices = self.env.get_env_attr("episode_index")
        pids = [worker.process.pid for worker in self.env.workers]
        self.env.close()
        self.env = None
        self._initialized[:] = False
        self.is_start = True
        get_logger().info("Bench2Dex simulator pool offloaded: exited_pids=%s", pids)

    def close(self) -> None:
        """Permanently close owned simulators; subsequent reset is invalid."""
        self._closed = True
        if self.env is not None and not self.env.is_closed:
            self.env.close()
        self.env = None
