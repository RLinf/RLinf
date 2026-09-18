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

import functools
import json
import os
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from PIL import Image

from rlinf.envs.sim.robotwin.seed_utils import partition_success_seeds
from rlinf.envs.utils import center_crop_image, list_of_dict_to_dict_of_list

__all__ = ["RoboTwinEnv"]


OPENWAM_ROBOTWIN_REPRESENTATIONS = ("absolute_eef20",)
_ACTION_TYPE_MARKER = "_rlinf_robotwin_action_type"


def robotwin_task_eef20_proprio(task: Any) -> np.ndarray:
    """Read OpenWAM's 20-D dual-arm EEF proprio straight from a RoboTwin task.

    Matches ``RoboTwinDataset._read_eef_actions`` / ``get_obs()["endpose"]``:
    ``[l_xyz(3), l_rot6d(6), l_grip(1), r_xyz(3), r_rot6d(6), r_grip(1)]`` with
    RoboTwin's xyzw quaternion turned into the first two rotation-matrix
    columns and the gripper kept as the raw ``[0, 1]`` opening (1 = open).
    ``VectorEnv.update_obs`` drops the ``endpose`` block, so the pose is read
    from the task object instead of the observation dict.
    """
    from rlinf.utils.rot6d import quat_xyzw_to_rot6d

    left = np.asarray(task.get_arm_pose("left"), dtype=np.float32).reshape(-1)
    right = np.asarray(task.get_arm_pose("right"), dtype=np.float32).reshape(-1)
    if left.shape[0] != 7 or right.shape[0] != 7:
        raise ValueError(
            "RoboTwin endpose must be 7-D xyz+quat_xyzw per arm, "
            f"got left={left.shape}, right={right.shape}"
        )
    grippers = (
        float(np.asarray(task.robot.get_left_gripper_val()).reshape(-1)[0]),
        float(np.asarray(task.robot.get_right_gripper_val()).reshape(-1)[0]),
    )
    proprio = np.concatenate(
        [
            left[:3],
            quat_xyzw_to_rot6d(left[3:7]),
            [grippers[0]],
            right[:3],
            quat_xyzw_to_rot6d(right[3:7]),
            [grippers[1]],
        ]
    ).astype(np.float32)
    if not np.isfinite(proprio).all():
        raise ValueError("RoboTwin EEF proprio contains non-finite values")
    return proprio


def execute_robotwin_ee_chunk(task: Any, chunk_actions: Any):
    """Run a chunk of 16-D ``ee`` actions on a RoboTwin task, one target at a time.

    ``gen_sparse_reward_data`` accepts an ``action_type`` argument but always
    slices the chunk as joint targets (6 + 1 + 6 + 1) and TOPPs the joint path,
    so 16-D end-effector chunks would drive the arms to garbage. ``take_action``
    does handle ``action_type="ee"`` (7-D pose + gripper per arm, planned with
    the arm's motion planner), so replay the chunk through it and reproduce the
    sparse-reward bookkeeping of ``gen_sparse_reward_data``: success on the
    task's ``eval_success`` flag, truncation when ``take_action_cnt`` reaches
    ``step_lim``. Returns ``(reward, termination, truncation, infos)`` shaped
    like the original so ``VectorEnv.step`` can consume it unchanged.
    """
    infos = {"success": False}
    reward = np.zeros(1, dtype=np.float32)
    termination = np.zeros(1, dtype=np.int32)
    truncation = np.zeros(1, dtype=np.int32)

    def _finish():
        if getattr(task, "eval_success", False):
            infos["success"] = True
            reward[:] = 1
            termination[:] = 1
        elif task.take_action_cnt >= task.step_lim:
            truncation[:] = 1
        return reward, termination, truncation, infos

    if getattr(task, "eval_success", False) or task.take_action_cnt >= task.step_lim:
        return _finish()
    actions = np.asarray(chunk_actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None]
    if actions.shape[-1] != 16:
        raise ValueError(
            f"RoboTwin ee control expects 16-D actions per step, got {actions.shape}"
        )
    for action in actions:
        if (
            getattr(task, "eval_success", False)
            or task.take_action_cnt >= task.step_lim
        ):
            break
        task.take_action(action, action_type="ee")
    return _finish()


def step_robotwin_venv(venv: Any, actions: Any, timeout_s: float | None):
    """``VectorEnv.step`` with a configurable per-sub-environment timeout.

    RoboTwin's ``VectorEnv.step`` waits ``future.result(timeout=120)`` on each
    sub-environment. End-effector chunks are planned target by target, and a
    few hard targets make the motion planner retry for tens of seconds, so a
    32-step chunk can legitimately exceed two minutes; the resulting
    ``TimeoutError`` surfaces as an empty "SubEnv i step error". Re-implement the
    fan-out with the caller's budget (``None`` waits indefinitely). Falls back
    to ``venv.step`` when the VectorEnv does not expose its thread pool.
    """
    envs = getattr(venv, "envs", None)
    pool = getattr(venv, "env_thread_pool", None)
    transform = getattr(venv, "transform", None)
    if not envs or pool is None or transform is None:
        return venv.step(actions)
    futures = [pool.submit(env.step, actions[i]) for i, env in enumerate(envs)]
    results = []
    for index, future in enumerate(futures):
        try:
            results.append(future.result(timeout=timeout_s))
        except Exception as exc:  # noqa: BLE001 - mirror VectorEnv's reporting
            raise RuntimeError(
                f"SubEnv {index} step error: {type(exc).__name__}: {exc}"
            ) from exc
    return transform(results)


def bind_robotwin_action_type(venv: Any, action_type: str) -> int:
    """Make every RoboTwin sub-environment execute chunks as ``action_type``.

    RoboTwin's ``VectorEnv.step`` forwards a chunk to
    ``task.gen_sparse_reward_data(chunk_actions)``, which only understands
    14-D joint targets. For ``"ee"`` that entry point is replaced on each task
    instance by :func:`execute_robotwin_ee_chunk`; ``"qpos"`` restores the
    original method. Idempotent; returns how many tasks were (re)bound.
    """
    if action_type not in ("qpos", "ee"):
        raise ValueError(f"Unsupported RoboTwin action_type {action_type!r}")
    bound = 0
    for sub_env in getattr(venv, "envs", []) or []:
        task = getattr(sub_env, "task", None)
        if task is None or getattr(task, _ACTION_TYPE_MARKER, None) == action_type:
            continue
        original = getattr(task, "_rlinf_original_gen_sparse_reward_data", None)
        if original is None:
            original = task.gen_sparse_reward_data
            task._rlinf_original_gen_sparse_reward_data = original
        if action_type == "ee":
            task.gen_sparse_reward_data = functools.partial(
                execute_robotwin_ee_chunk, task
            )
        else:
            task.gen_sparse_reward_data = original
        setattr(task, _ACTION_TYPE_MARKER, action_type)
        bound += 1
    return bound


class RoboTwinEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        env_seed = cfg.seed
        self.seed = env_seed + seed_offset
        self.base_seed = env_seed
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations

        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.use_custom_reward = cfg.use_custom_reward

        self.video_cfg = cfg.video_cfg

        self.cfg = cfg
        self.record_metrics = record_metrics
        self._is_start = True

        self.task_name = cfg.task_config.task_name

        self.center_crop = cfg.get("center_crop", False)
        # OpenWAM RoboTwin checkpoints act in the 20-D absolute EEF space of the
        # dataset's ``endpose`` fields. ``absolute_eef20`` switches the
        # sub-environments to RoboTwin's ``ee`` controller and adds a matching
        # ``native_proprio`` observation; joint-space policies leave it unset.
        self.openwam_action_representation = cfg.get(
            "openwam_action_representation", None
        )
        if self.openwam_action_representation not in (
            None,
            *OPENWAM_ROBOTWIN_REPRESENTATIONS,
        ):
            raise ValueError(
                "RoboTwin openwam_action_representation must be one of "
                f"{OPENWAM_ROBOTWIN_REPRESENTATIONS} or null, "
                f"got {self.openwam_action_representation!r}"
            )
        self.robotwin_action_type = (
            "ee" if self.openwam_action_representation is not None else "qpos"
        )
        # Per-chunk wait for one sub-environment; null waits indefinitely. Only
        # used for ee control, joint chunks keep VectorEnv's own 120 s.
        timeout = cfg.get("robotwin_step_timeout_s", 1800)
        self.robotwin_step_timeout_s = None if timeout is None else float(timeout)
        self._init_reset_state_ids()

        self._init_env()

        self.prev_step_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        if self.record_metrics:
            self._init_metrics()
            self._elapsed_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )

    def _init_env(self):
        mp.set_start_method("spawn", force=True)
        os.environ["ASSETS_PATH"] = self.cfg.assets_path

        from robotwin.envs.vector_env import VectorEnv

        env_seeds = self.reset_state_ids.tolist()

        self.venv = VectorEnv(
            task_config=OmegaConf.to_container(self.cfg.task_config, resolve=True),
            n_envs=self.num_envs,
            env_seeds=env_seeds,
        )
        self._bind_action_type()

    def _venv_step(self, actions):
        if self.robotwin_action_type == "ee":
            return step_robotwin_venv(self.venv, actions, self.robotwin_step_timeout_s)
        return self.venv.step(actions)

    def _bind_action_type(self) -> None:
        """(Re)bind the controller mode; sub-envs are rebuilt after ``close()``."""
        if self.robotwin_action_type != "qpos":
            bind_robotwin_action_type(self.venv, self.robotwin_action_type)

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
                self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0
                self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            if isinstance(infos["success"], list):
                infos["success"] = torch.as_tensor(
                    np.array(infos["success"]).reshape(-1), device=self.device
                )
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def center_and_crop(self, image, center_crop=False):
        image = np.array(image)

        image = Image.fromarray(image).convert("RGB")
        if center_crop:
            image = center_crop_image(image)
        return np.array(image)

    def _extract_obs_image(self, raw_obs):
        batch_images = []
        batch_wrist_images = []
        batch_states = []
        batch_instructions = []
        for obs in raw_obs:
            batch_images.append(
                self.center_and_crop(obs["full_image"], center_crop=self.center_crop)
            )
            wrist_images = []
            if "left_wrist_image" in obs and obs["left_wrist_image"] is not None:
                wrist_images.append(
                    self.center_and_crop(
                        obs["left_wrist_image"], center_crop=self.center_crop
                    )
                )
            if "right_wrist_image" in obs and obs["right_wrist_image"] is not None:
                wrist_images.append(
                    self.center_and_crop(
                        obs["right_wrist_image"], center_crop=self.center_crop
                    )
                )
            if len(wrist_images) > 0:
                batch_wrist_images.append(
                    torch.stack([torch.from_numpy(img) for img in wrist_images])
                )
            batch_states.append(obs["state"])
            batch_instructions.append(obs["instruction"])

        batch_images = torch.stack([torch.from_numpy(img) for img in batch_images])
        if len(batch_wrist_images) > 0:
            batch_wrist_images = torch.stack(batch_wrist_images)
        else:
            batch_wrist_images = None
        batch_states = torch.stack([torch.from_numpy(state) for state in batch_states])

        extracted_obs = {
            "main_images": batch_images,
            "wrist_images": batch_wrist_images,
            "states": batch_states,
            "task_descriptions": batch_instructions,
        }
        if self.openwam_action_representation == "absolute_eef20":
            extracted_obs["native_proprio"] = self._extract_native_proprio(raw_obs)

        return extracted_obs

    def _extract_native_proprio(self, raw_obs) -> torch.Tensor:
        """20-D EEF proprio per env, from ``endpose`` when present else the task."""
        proprios = []
        sub_envs = list(getattr(self.venv, "envs", []) or [])
        for index, obs in enumerate(raw_obs):
            endpose = obs.get("endpose") if isinstance(obs, dict) else None
            if endpose and all(
                key in endpose
                for key in (
                    "left_endpose",
                    "right_endpose",
                    "left_gripper",
                    "right_gripper",
                )
            ):
                from rlinf.utils.rot6d import quat_xyzw_to_rot6d

                left = np.asarray(endpose["left_endpose"], dtype=np.float32).reshape(-1)
                right = np.asarray(endpose["right_endpose"], dtype=np.float32).reshape(
                    -1
                )
                proprio = np.concatenate(
                    [
                        left[:3],
                        quat_xyzw_to_rot6d(left[3:7]),
                        np.asarray(endpose["left_gripper"], np.float32).reshape(-1)[:1],
                        right[:3],
                        quat_xyzw_to_rot6d(right[3:7]),
                        np.asarray(endpose["right_gripper"], np.float32).reshape(-1)[
                            :1
                        ],
                    ]
                ).astype(np.float32)
            else:
                if index >= len(sub_envs):
                    raise RuntimeError(
                        "RoboTwin observation has no endpose and the VectorEnv exposes "
                        f"only {len(sub_envs)} sub-environments for {len(raw_obs)} observations"
                    )
                proprio = robotwin_task_eef20_proprio(sub_envs[index].task)
            proprios.append(torch.from_numpy(proprio))
        return torch.stack(proprios)

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations

        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _cal_chunk_rewards(self, step_reward, chunk_step, terminations, infos):
        n_steps_to_run = np.array(
            [[0] for i in range(self.num_envs)]
        )  # infos.get("n_steps_to_run", np.array([[0] for i in range(self.num_envs)]))

        n_steps_to_run = torch.as_tensor(
            np.array(n_steps_to_run).reshape(-1), device=self.device
        )
        chunk_rewards = torch.zeros(self.num_envs, chunk_step, device=self.device)
        for env_id in range(self.num_envs):
            steps_left = n_steps_to_run[env_id]
            reward = step_reward[env_id]
            start_idx = chunk_step - steps_left - 1

            if terminations[env_id] and start_idx > 0:
                if self.use_rel_reward:
                    chunk_rewards[env_id, start_idx] = reward
                else:
                    chunk_rewards[env_id, start_idx:] = reward

        return chunk_rewards

    def reset(
        self,
        env_idx: Optional[Union[int, list[int]]] = None,
        env_seeds=None,
    ):
        if self._is_start:
            self._is_start = False

        env_seeds = self.reset_state_ids.tolist() if env_seeds is None else env_seeds

        self.venv.reset(env_idx=env_idx, env_seeds=env_seeds)
        self._bind_action_type()
        raw_obs = self.venv.get_obs()
        infos = {}

        self._reset_metrics(env_idx)

        extracted_obs = self._extract_obs_image(raw_obs)

        return extracted_obs, infos

    def step(
        self, actions: Union[torch.Tensor, np.ndarray, dict] = None, auto_reset=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        elif isinstance(actions, dict):
            actions = actions.get("actions", actions)

        # [n_envs, horizon, action_dim]
        if len(actions.shape) == 2:
            # [n_envs, action_dim] -> [n_envs, 1, action_dim]
            actions = actions[:, None, :]

        self._bind_action_type()
        raw_obs, step_reward, terminations, truncations, info_list = self._venv_step(
            actions
        )
        extracted_obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)

        if isinstance(terminations, list):
            terminations = torch.as_tensor(
                np.array(terminations).reshape(-1), device=self.device
            )
        if isinstance(truncations, list):
            truncations = torch.as_tensor(
                np.array(truncations).reshape(-1), device=self.device
            )

        if self.use_custom_reward:
            step_reward = self._calc_step_reward(terminations)
        else:
            if isinstance(step_reward, list):
                step_reward = torch.as_tensor(
                    np.array(step_reward, dtype=np.float32).reshape(-1),
                    device=self.device,
                )

        self._elapsed_steps += actions.shape[1]
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        if truncated.any():
            truncations = torch.logical_or(truncated, truncations)

        infos = self._record_metrics(step_reward, infos)

        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)

        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.cpu().numpy()

        # chunk_actions: [num_envs, chunk_step, action_dim]
        num_envs = chunk_actions.shape[0]
        chunk_step = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        self._bind_action_type()
        raw_obs, step_reward, terminations, truncations, info_list = self._venv_step(
            chunk_actions
        )
        extracted_obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)
        obs_list.append(extracted_obs)
        infos_list.append(infos)
        if isinstance(terminations, list):
            terminations = torch.as_tensor(
                np.array(terminations).reshape(-1), device=self.device
            )
        if isinstance(truncations, list):
            truncations = torch.as_tensor(
                np.array(truncations).reshape(-1), device=self.device
            )

        if self.use_custom_reward:
            step_reward = self._calc_step_reward(terminations)
        else:
            if isinstance(step_reward, list):
                step_reward = torch.as_tensor(
                    np.array(step_reward, dtype=np.float32).reshape(-1),
                    device=self.device,
                )

        chunk_rewards = self._cal_chunk_rewards(
            step_reward, chunk_step, terminations, infos
        )

        self._elapsed_steps += chunk_actions.shape[1]
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        if truncated.any():
            truncations = torch.logical_or(truncated, truncations)

        infos = self._record_metrics(step_reward, infos)

        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()

        past_dones = torch.logical_or(terminations, truncations)
        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros((num_envs, chunk_step), dtype=bool)
        chunk_terminations[:, -1] = terminations

        chunk_truncations = torch.zeros((num_envs, chunk_step), dtype=bool)
        chunk_truncations[:, -1] = truncations

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs.copy()
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        final_info = infos.copy()
        if self.cfg.is_eval:
            self.update_reset_state_ids(env_idx=env_idx)

        extracted_obs, infos = self.reset(env_idx=env_idx.tolist())
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def offload(self, clear_cache=True):
        if hasattr(self, "venv"):
            self.venv.close(clear_cache)

    def sample_action_space(self):
        action_dim = 16 if self.robotwin_action_type == "ee" else 14
        return np.random.randn(self.num_envs, self.horizon, action_dim)

    def _init_reset_state_ids(self):
        if self.cfg.get("seeds_path", None) is not None and os.path.exists(
            self.cfg.seeds_path
        ):
            with open(self.cfg.seeds_path, "r") as f:
                data = json.load(f)
            success_seeds = data[self.task_name].get("success_seeds", None)
            if success_seeds is not None:
                success_seeds = torch.as_tensor(success_seeds, dtype=torch.long)
                self.success_seeds = partition_success_seeds(
                    success_seeds,
                    base_seed=self.base_seed,
                    seed_offset=self.seed_offset,
                    total_num_processes=self.total_num_processes,
                    num_group=self.num_group,
                )
                self._current_seed_index = 0
            else:
                self.success_seeds = None
                self._current_seed_index = 0
        else:
            self.success_seeds = None
            self._current_seed_index = 0

        if not hasattr(self, "_generator"):
            self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self, env_idx=None):
        if self.use_fixed_reset_state_ids and hasattr(self, "reset_state_ids"):
            return

        if env_idx is not None and hasattr(self, "reset_state_ids"):
            if self.success_seeds is not None:
                total_seeds = self.success_seeds.numel()
                indices = (
                    torch.arange(self.num_group, device=self.success_seeds.device)
                    + self._current_seed_index
                ) % total_seeds
                reset_state_ids = self.success_seeds[indices]
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
                self._current_seed_index = (
                    self._current_seed_index + self.num_group
                ) % total_seeds
            else:
                reset_state_ids = torch.randint(
                    low=10000,
                    high=200000,
                    size=(self.num_group,),
                    generator=self._generator,
                )
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
            for idx in env_idx:
                self.reset_state_ids[idx] = reset_state_ids[idx]
        else:
            if self.success_seeds is not None:
                total_seeds = self.success_seeds.numel()
                indices = (
                    torch.arange(self.num_group, device=self.success_seeds.device)
                    + self._current_seed_index
                ) % total_seeds
                reset_state_ids = self.success_seeds[indices]
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
                self._current_seed_index = (
                    self._current_seed_index + self.num_group
                ) % total_seeds
            else:
                reset_state_ids = torch.randint(
                    low=10000,
                    high=200000,
                    size=(self.num_group,),
                    generator=self._generator,
                )
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
            self.reset_state_ids = reset_state_ids

    def check_seeds(self, seeds):
        resutls = self.venv.check_seeds(seeds)

        return resutls
