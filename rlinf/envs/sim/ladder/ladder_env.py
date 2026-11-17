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

"""RLinf environment wrapper for the LADDER long-horizon manipulation benchmark."""

from __future__ import annotations

import atexit
import contextlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

DEFAULT_LADDER_ROOT = "/app/ladder"
_CANONICAL_TO_WIDOWX_ROTATION = np.array(
    [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
)

SIMPLER_TASKS: tuple[str, ...] = (
    "all10_plate",
    "all10_towel",
    "all1_plate",
    "all1_towel",
    "all2_plate",
    "all2_towel",
    "all3_plate",
    "all3_towel",
    "all4_plate",
    "all4_towel",
    "all5_plate",
    "all5_towel",
    "all6_plate",
    "all6_towel",
    "all7_plate",
    "all7_towel",
    "all8_plate",
    "all8_towel",
    "all9_plate",
    "all9_towel",
    "except_eggplant_2",
    "except_eggplant_3",
    "except_eggplant_4",
    "except_eggplant_5",
    "move_plate_to_towel_2",
    "move_plate_to_towel_3",
    "move_plate_to_towel_4",
    "move_plate_to_towel_5",
    "move_towel_to_plate_3",
    "ordered_plate_2",
    "ordered_plate_3",
    "relay10e",
    "relay2e",
    "relay3e",
    "relay4e",
    "relay5e",
    "relay6e",
    "relay7e",
    "relay8e",
    "relay9e",
    "shuttle10",
    "shuttle7",
    "shuttle8",
    "shuttle9",
    "split2",
    "split3",
    "split4",
    "split5",
    "split6",
    "split7",
    "spoons2",
    "spoons3",
    "spoons4",
    "spoons5",
)

ROBOT_ARENA_TASKS: tuple[str, ...] = (
    "scene36_1goal",
    "scene36_2goal",
    "scene45_1goal",
    "scene45_2goal",
    "scene45_3goal",
    "scene45_4goal",
    "scene45_5goal",
    "scene45_6goal",
    "scene45_all_2",
    "scene45_all_3",
    "scene45_all_4",
    "scene45_all_5",
    "scene45_all_6",
    "scene45_except_2",
    "scene45_except_3",
    "scene45_except_4",
    "scene45_except_5",
    "scene45_except_6",
    "scene50_3goal",
    "scene54_1goal",
    "scene54_2goal",
    "scene54_2goal_ordered",
    "scene54_3goal",
    "scene54_3goal_ordered",
    "scene54_duplicates_2",
    "scene54_duplicates_3",
    "scene54_except",
    "scene54_except_3",
    "scene54_relay_2",
    "scene54_relay_4",
    "scene54_revisit_10",
    "scene54_revisit_3",
    "scene54_revisit_6",
    "scene54_revisit_7",
    "scene54_revisit_8",
    "scene54_revisit_9",
    "scene54_route_2",
    "scene54_route_4",
    "scene54_route_6",
    "scene54_route_7",
    "scene54_route_8",
    "scene54_split_2",
    "scene54_split_3",
    "scene54_transfer_2",
    "scene76_1goal",
    "scene76_2goal",
    "scene76_2goal_ordered",
)

ROBOLAB_TASKS: tuple[str, ...] = (
    "comp_bin_1",
    "comp_bin_10",
    "comp_bin_2",
    "comp_bin_2_distractor",
    "comp_bin_4_distractor",
    "comp_bin_6",
    "comp_identical_2",
    "comp_identical_4",
    "comp_ordered_bin_10",
    "comp_ordered_bin_2",
    "comp_ordered_bin_3",
    "comp_ordered_bin_4",
    "comp_ordered_bin_5",
    "comp_ordered_bin_6",
    "comp_ordered_bin_7",
    "comp_ordered_bin_8",
    "comp_ordered_bin_9",
    "comp_plate_1",
    "comp_relay_2",
    "comp_relay_3",
    "comp_relay_4",
    "comp_relay_5",
    "comp_revisit_1",
    "comp_revisit_2",
    "comp_revisit_3",
    "comp_route_10",
    "comp_route_2",
    "comp_route_2_binfirst",
    "comp_route_3",
    "comp_route_4",
    "comp_route_5",
    "comp_route_6",
    "comp_route_7",
    "comp_route_8",
    "comp_route_9",
    "comp_split_plate_2",
    "comp_split_plate_3",
    "comp_split_plate_4",
    "comp_split_plate_5",
    "comp_split_plate_6",
    "comp_split_plate_7",
    "comp_split_plate_8",
    "comp_transfer_2",
    "comp_transfer_3",
)

ROBOT_ARENA_K8_OFFSETS: tuple[tuple[float, float], ...] = (
    (-0.03, -0.02),
    (-0.03, +0.02),
    (-0.01, -0.02),
    (-0.01, +0.02),
    (+0.01, -0.02),
    (+0.01, +0.02),
    (+0.03, -0.02),
    (+0.03, +0.02),
)

SUITE_TASKS: dict[str, tuple[str, ...]] = {
    "simpler": SIMPLER_TASKS,
    "robot_arena": ROBOT_ARENA_TASKS,
    "robolab": ROBOLAB_TASKS,
}
LADDER_SUITES: tuple[str, ...] = tuple(SUITE_TASKS)


def _axisangle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    kx, ky, kz = rv / theta
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * k
        + (1.0 - np.cos(theta)) * (k @ k)
    )


def _matrix_to_axisangle(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    theta = float(np.arccos(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)))
    if theta < 1e-7:
        return np.zeros(3, dtype=np.float32)
    axis = np.array(
        [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]], dtype=np.float64
    )
    norm = float(np.linalg.norm(axis))
    return (
        ((axis / norm) * theta).astype(np.float32)
        if norm >= 1e-12
        else np.zeros(3, dtype=np.float32)
    )


def _intrinsic_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    x, y, z = float(euler[0]), float(euler[1]), float(euler[2])
    cx, sx, cy, sy, cz, sz = (
        np.cos(x),
        np.sin(x),
        np.cos(y),
        np.cos(y),
        np.cos(z),
        np.sin(z),
    )
    return np.array(
        [
            [cy * cz, -cy * sz, sy],
            [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
            [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy],
        ],
        dtype=np.float64,
    )


def _ensure_ladder_on_path(root: Path) -> None:
    for p in (
        root,
        root / "build_support" / "python",
        root / "robolab_src",
        root / "openpi_src" / "packages" / "openpi-client" / "src",
    ):
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
    try:
        import ladder.components  # noqa: F401

        return
    except ImportError:
        sys.modules.pop("ladder", None)
    if not (root / "ladder").is_dir():
        raise FileNotFoundError(
            f"LADDER source tree not found at {root}. Set LADDER_ROOT or pass init_params.ladder_root."
        )


def _clone_obs_dict(obs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in obs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        elif isinstance(v, list):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = _clone_obs_dict(v)
        else:
            out[k] = v
    return out


class _RoboLabKit:
    app: Any = None
    thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ladder-robolab")
    env: Any = None


class LadderEnv(gym.Env):
    """RLinf-compatible vectorized environment wrapper for LADDER (`simpler`, `robot_arena`, `robolab`)."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any = None,
        record_metrics: bool = True,
    ) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed = int(getattr(cfg, "seed", 0)) + int(seed_offset)
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info

        self.group_size = int(getattr(cfg, "group_size", 1))
        self.num_group = max(self.num_envs // max(self.group_size, 1), 1)
        self.auto_reset = bool(getattr(cfg, "auto_reset", True))
        self.use_rel_reward = bool(getattr(cfg, "use_rel_reward", True))
        self.ignore_terminations = bool(getattr(cfg, "ignore_terminations", False))
        self.use_fixed_reset_state_ids = bool(
            getattr(cfg, "use_fixed_reset_state_ids", False)
        )
        self.reward_coef = float(getattr(cfg, "reward_coef", 1.0))
        self.reward_mode = str(getattr(cfg, "reward_mode", "subgoal_progress"))
        self.video_cfg = getattr(cfg, "video_cfg", None)
        self.record_metrics = bool(record_metrics)
        self._is_start = True

        init_params = getattr(cfg, "init_params", None)
        if init_params is not None and OmegaConf.is_config(init_params):
            init_dict = OmegaConf.to_container(init_params, resolve=True) or {}
        elif isinstance(init_params, dict):
            init_dict = dict(init_params)
        else:
            init_dict = {}

        self.suite = str(init_dict.get("suite", getattr(cfg, "suite", "simpler")))
        if self.suite not in SUITE_TASKS:
            raise ValueError(
                f"Unknown LADDER suite {self.suite!r}. Expected one of {LADDER_SUITES}."
            )

        default_task = SUITE_TASKS[self.suite][0]
        self.task_name = str(
            init_dict.get("task_name", getattr(cfg, "task_name", default_task))
        )
        if self.task_name not in SUITE_TASKS[self.suite]:
            raise ValueError(
                f"Unknown task {self.task_name!r} for LADDER suite {self.suite!r}."
            )

        self.prompt_mode = str(init_dict.get("prompt_mode", "flat"))
        self.action_mode = str(
            init_dict.get(
                "action_mode", "joint_vel" if self.suite == "robolab" else "delta_ee"
            )
        )
        self.control_mode = init_dict.get("control_mode")
        if self.control_mode and "target_base_pose" in str(self.control_mode):
            self.action_mode = "abs_ee"

        root_arg = init_dict.get("ladder_root") or os.environ.get(
            "LADDER_ROOT", DEFAULT_LADDER_ROOT
        )
        self._root = Path(str(root_arg)).expanduser().resolve()
        assets_arg = init_dict.get("assets_dir")
        self._assets_dir = (
            Path(str(assets_arg)).expanduser().resolve() if assets_arg else None
        )
        default_h, default_w = (224, 224) if self.suite == "robolab" else (480, 640)
        self._obs_h = int(init_dict.get("camera_height", default_h))
        self._obs_w = int(init_dict.get("camera_width", default_w))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_episode_steps = int(
            getattr(cfg, "max_episode_steps", None)
            or (1500 if self.suite == "robolab" else 1000)
        )

        self._env: Any = None
        self._env_cfg: Any = None
        self._converter: Any = None
        self._raw_obs: Any = None
        self._oracle: Any = None
        self._padded_single = False
        self._rollout_idx = 0
        self._last_eef_poses = [
            np.eye(4, dtype=np.float32) for _ in range(self.num_envs)
        ]
        self._last_target_poses = [
            np.eye(4, dtype=np.float32) for _ in range(self.num_envs)
        ]
        self._last_joints: list[np.ndarray | None] = [
            None for _ in range(self.num_envs)
        ]

        self._build_simulator(init_dict)

        action_dim = 8 if self.suite == "robolab" else 7
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(action_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({})

        self._init_reset_state_ids()
        self.prev_step_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        if self.record_metrics:
            self._init_metrics()

    def _build_simulator(self, init_dict: dict[str, Any]) -> None:
        # Simulator is initialized on first reset (matching SimplerEnv / RobotArenaEnv / RoboLab lifecycle).
        pass

    @property
    def total_num_group_envs(self) -> int:
        return int(np.iinfo(np.uint8).max // 2)

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    def _init_reset_state_ids(self) -> None:
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self) -> None:
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    def _init_metrics(self) -> None:
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.max_subgoal_progress = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx: Optional[torch.Tensor] = None) -> None:
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self._elapsed_steps[mask] = 0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.max_subgoal_progress[mask] = 0.0
                self.returns[mask] = 0.0
        else:
            self.prev_step_reward[:] = 0.0
            self._elapsed_steps[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.max_subgoal_progress[:] = 0.0
                self.returns[:] = 0.0

    def _calc_step_reward(
        self, progress_t: torch.Tensor, success_t: torch.Tensor
    ) -> torch.Tensor:
        base_reward = (
            success_t.float() if self.reward_mode == "only_success" else progress_t
        )
        reward = torch.maximum(self.reward_coef * base_reward, self.prev_step_reward)
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward.clone()
        return reward_diff if self.use_rel_reward else reward

    def _record_metrics(
        self,
        step_reward: torch.Tensor,
        success_t: torch.Tensor,
        progress_t: torch.Tensor,
        infos: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.record_metrics:
            return infos
        episode_info: dict[str, Any] = {}
        self.returns += step_reward
        self.success_once = self.success_once | success_t
        self.max_subgoal_progress = torch.maximum(
            self.max_subgoal_progress,
            torch.where(self.success_once, torch.ones_like(progress_t), progress_t),
        )
        episode_info["success_once"] = self.success_once.clone()
        episode_info["subgoal_progress"] = self.max_subgoal_progress.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / torch.clamp(
            episode_info["episode_len"].float(), min=1.0
        )
        infos["episode"] = episode_info
        return infos

    def _pack_raw_obs_list(
        self, raw_obs_list: list[Any]
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        main_imgs: list[np.ndarray] = []
        wrist_imgs: list[np.ndarray] = []
        states: list[np.ndarray] = []
        instructions: list[str] = []
        success_arr = np.zeros(self.num_envs, dtype=bool)
        trunc_arr = np.zeros(self.num_envs, dtype=bool)
        progress_arr = np.zeros(self.num_envs, dtype=np.float32)

        for i, raw_obs in enumerate(raw_obs_list):
            if isinstance(raw_obs, dict):
                images = {
                    k: np.asarray(v, dtype=np.uint8)
                    for k, v in (raw_obs.get("images") or {}).items()
                }
                st = raw_obs.get("state") or {}
                eef_pose = np.asarray(st.get("eef_pose", np.eye(4)), dtype=np.float32)
                target_pose = np.asarray(
                    st.get("eef_pose_from_target", eef_pose), dtype=np.float32
                )
                gripper = float(
                    np.asarray(st.get("gripper_state", [1.0])).reshape(-1)[0]
                )
                joints = st.get("joints")
                metadata = raw_obs.get("metadata") or {}
            else:
                images = {
                    k: np.asarray(v, dtype=np.uint8)
                    for k, v in (raw_obs.images or {}).items()
                }
                eef_pose = np.asarray(raw_obs.state.eef_pose, dtype=np.float32)
                target_raw = getattr(raw_obs.state, "eef_pose_from_target", None)
                target_pose = np.asarray(
                    target_raw if target_raw is not None else eef_pose, dtype=np.float32
                )
                gripper = float(np.asarray(raw_obs.state.gripper_state).reshape(-1)[0])
                joints = getattr(raw_obs.state, "joints", None)
                metadata = raw_obs.metadata or {}

            self._last_eef_poses[i] = eef_pose
            self._last_target_poses[i] = target_pose
            self._last_joints[i] = (
                np.asarray(joints, dtype=np.float32).reshape(-1)[:7]
                if joints is not None
                else None
            )

            main_img = images.get("main")
            if main_img is None:
                main_img = next(
                    iter(images.values()),
                    np.zeros((self._obs_h, self._obs_w, 3), dtype=np.uint8),
                )
            main_imgs.append(main_img)
            if self.suite == "robolab":
                wrist_imgs.append(images.get("wrist", np.zeros_like(main_img)))

            if self.suite == "robolab" and joints is not None:
                j = np.asarray(joints, dtype=np.float32).reshape(-1)[:7]
                states.append(
                    np.concatenate([j, np.array([1.0 - gripper], dtype=np.float32)])
                )
            else:
                rotvec = _matrix_to_axisangle(eef_pose[:3, :3])
                states.append(
                    np.concatenate(
                        [eef_pose[:3, 3], rotvec, np.array([gripper], dtype=np.float32)]
                    )
                )

            step_done = bool(metadata.get("done", False))
            oracle = (
                metadata.get("oracle")
                if isinstance(metadata.get("oracle"), dict)
                else None
            )
            achieved = [bool(v) for v in oracle.get("achieved", [])] if oracle else []
            subtasks = [str(s) for s in oracle.get("subtasks", [])] if oracle else []
            n_goals = len(achieved)
            completed = int(metadata.get("num_subgoals_complete", sum(achieved)))
            if step_done and n_goals > 0:
                completed = max(completed, n_goals)

            instr = str(metadata.get("instruction", self.task_name))
            if self.prompt_mode == "oracle_hlc" and subtasks:
                active_idx = next(
                    (idx for idx, d in enumerate(achieved) if not d), len(subtasks) - 1
                )
                if 0 <= active_idx < len(subtasks) and subtasks[active_idx]:
                    instr = subtasks[active_idx]
            instructions.append(instr)

            success_arr[i] = step_done
            trunc_arr[i] = bool(metadata.get("truncated", False))
            progress_arr[i] = (
                float(completed) / float(n_goals) if n_goals > 0 else float(step_done)
            )

        obs_dict = {
            "main_images": torch.from_numpy(
                np.ascontiguousarray(np.stack(main_imgs, axis=0))
            ).to(self.device, dtype=torch.uint8),
            "wrist_images": (
                torch.from_numpy(np.ascontiguousarray(np.stack(wrist_imgs, axis=0))).to(
                    self.device, dtype=torch.uint8
                )
                if self.suite == "robolab"
                else None
            ),
            "states": torch.from_numpy(np.stack(states, axis=0).astype(np.float32)).to(
                self.device, dtype=torch.float32
            ),
            "task_descriptions": instructions,
        }
        return (
            obs_dict,
            torch.from_numpy(success_arr).to(self.device, dtype=torch.bool),
            torch.from_numpy(trunc_arr).to(self.device, dtype=torch.bool),
            torch.from_numpy(progress_arr).to(self.device, dtype=torch.float32),
        )

    def _decode_action(self, env_i: int, raw: np.ndarray) -> Any:
        from ladder.components.inference.api.core.action import (
            EndEffectorPoseAction,
            JointAnglesAction,
            ManipulatorAction,
        )

        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        if raw.size == 8 and (
            self.action_mode in ("joint_pos", "joint_vel") or self.suite == "robolab"
        ):
            arm = raw[:7].astype(np.float32)
            cur_j = self._last_joints[env_i]
            joints = (
                (cur_j + np.clip(arm, -1.0, 1.0) * 0.2).astype(np.float32)
                if (self.action_mode == "joint_vel" and cur_j is not None)
                else arm
            )
            return ManipulatorAction(
                arm_action=JointAnglesAction(angles=joints),
                gripper_action=0.0 if float(raw[7]) > 0.5 else 1.0,
            )

        if raw.size < 7:
            raise ValueError(
                f"LadderEnv expected at least a 7-D action, got shape {raw.shape}"
            )
        pos = raw[:3].astype(np.float32)
        rot_in = raw[3:6].astype(np.float64)
        gripper = (
            float(1.0 if raw[6] > 0.0 else 0.0)
            if raw[6] < 0.0
            else float(np.clip(raw[6], 0.0, 1.0))
        )
        target_eef = self._last_target_poses[env_i]
        if self.action_mode == "delta_ee":
            translation = (target_eef[:3, 3] + pos).astype(np.float32)
            rotation = (_axisangle_to_matrix(rot_in) @ target_eef[:3, :3]).astype(
                np.float32
            )
            new_target = np.eye(4, dtype=np.float32)
            new_target[:3, 3], new_target[:3, :3] = translation, rotation
            self._last_target_poses[env_i] = new_target
        elif self.control_mode and "target_base_pose" in str(self.control_mode):
            translation = pos
            rotation = (
                _intrinsic_xyz_to_matrix(rot_in) @ _CANONICAL_TO_WIDOWX_ROTATION.T
            ).astype(np.float32)
        else:
            translation = pos
            rotation = _intrinsic_xyz_to_matrix(rot_in).astype(np.float32)
        return ManipulatorAction(
            arm_action=EndEffectorPoseAction(
                translation=translation, rotation=rotation
            ),
            gripper_action=gripper,
        )

    def _reset_backend(self, episode_ids: list[int]) -> list[Any]:
        if self.suite == "simpler":
            return self._reset_simpler(episode_ids)
        if self.suite == "robot_arena":
            return self._reset_robot_arena(episode_ids)
        return _RoboLabKit.thread.submit(self._reset_robolab).result()

    def _step_backend(self, actions: list[Any]) -> list[Any]:
        assert self._env is not None
        if self.suite == "simpler":
            return self._env.step(actions)
        if self.suite == "robot_arena":
            step_actions = [actions[0], actions[0]] if self._padded_single else actions
            obs_list = self._env.step(step_actions)
            return obs_list[:1] if self._padded_single else obs_list
        return _RoboLabKit.thread.submit(self._step_robolab, actions).result()

    def _reset_simpler(self, episode_ids: list[int]) -> list[Any]:
        _ensure_ladder_on_path(self._root)
        if "MS_ASSET_DIR" not in os.environ:
            for cand in (
                self._assets_dir,
                Path(os.environ["LADDER_SIMPLER_ASSETS"])
                if "LADDER_SIMPLER_ASSETS" in os.environ
                else None,
                self._root
                / "ladder"
                / "components"
                / "inference"
                / "robots"
                / "simpler"
                / "maniskill3",
            ):
                if cand is not None and cand.is_dir():
                    os.environ["MS_ASSET_DIR"] = str(cand)
                    break

        from ladder.components.inference.robots.simpler.env_builder import (
            ABS_CONTROL_MODE,
        )
        from ladder.components.inference.robots.simpler.simpler_env import (
            SimplerEnv,
            SimplerEnvConfig,
        )

        if self._env is None:
            self._env = SimplerEnv(
                SimplerEnvConfig(
                    env_name=f"{self.task_name}-v0",
                    robot_name="widowx",
                    scene_name="bridge_table_1_v1",
                    rgb_overlay_path="",
                    sim_freq=500,
                    control_mode=ABS_CONTROL_MODE,
                    control_freq=5,
                    max_episode_steps=0,
                    raytracing=False,
                    n_envs=self.num_envs,
                    episode_layouts=max(24, max(episode_ids, default=0) + 1),
                )
            )
        return self._env.reset(episode_ids)

    def _reset_robot_arena(self, episode_ids: list[int]) -> list[Any]:
        _ensure_ladder_on_path(self._root)
        from ladder.components.inference.robots.robot_arena.composition.registry import (
            COMPOSITION_TASKS,
        )
        from ladder.components.inference.robots.robot_arena.robot_arena_env import (
            RobotArenaEnv,
            RobotArenaEnvConfig,
        )
        from ladder.components.inference.robots.robot_arena.scene_builder import (
            set_genesis_seed,
        )
        from ladder.core.compute.seeder import SEEDER

        data_root = next(
            (
                str(c)
                for c in (
                    self._assets_dir,
                    Path(os.environ["LADDER_ROBOT_ARENA_ASSETS"])
                    if "LADDER_ROBOT_ARENA_ASSETS" in os.environ
                    else None,
                    self._root
                    / "ladder/components/inference/robots/robot_arena/robot_arena_data/examples/data",
                )
                if c is not None
                and ((c / "bridge").is_dir() or (c / "scenes").is_dir())
            ),
            None,
        )
        if data_root is None:
            raise FileNotFoundError(
                "RobotArena assets not found. Set LADDER_ROBOT_ARENA_ASSETS."
            )

        SEEDER.seed_everything(self.seed)
        set_genesis_seed(self.seed)
        comp_task = COMPOSITION_TASKS[self.task_name]
        offsets = [
            ROBOT_ARENA_K8_OFFSETS[ep % len(ROBOT_ARENA_K8_OFFSETS)]
            for ep in episode_ids
        ]
        self._padded_single = len(offsets) == 1
        if self._padded_single:
            offsets = [offsets[0], ROBOT_ARENA_K8_OFFSETS[0]]

        cfg = RobotArenaEnvConfig(
            data_root=data_root,
            scene_name=comp_task.scene,
            robot_name="widowx",
            max_episode_steps=comp_task.max_episode_steps,
            control_freq=5,
            sim_freq=200,
            sim_substeps=1,
            rigid_constraint_timeconst=0.02,
            camera_width=self._obs_w,
            camera_height=self._obs_h,
            per_env_object_offsets=offsets,
            composition_task=self.task_name,
        )
        if self._env is not None and self._env.can_reuse_for(cfg):
            self._env.reconfigure(cfg)
        else:
            if self._env is not None:
                self._env.close()
            self._env = RobotArenaEnv(config=cfg)
        obs_list = self._env.reset()
        return obs_list[:1] if self._padded_single else obs_list

    def _reset_robolab(self) -> list[Any]:
        _ensure_ladder_on_path(self._root)
        if "ROBOLAB_ASSET_DIR" not in os.environ:
            for cand in (
                self._assets_dir,
                Path(os.environ["LADDER_ROBOLAB_ASSETS"])
                if "LADDER_ROBOLAB_ASSETS" in os.environ
                else None,
                self._root
                / "ladder"
                / "components"
                / "inference"
                / "robots"
                / "robolab"
                / "assets",
            ):
                if cand is not None and cand.is_dir():
                    os.environ["ROBOLAB_ASSET_DIR"] = str(cand)
                    break

        from ladder.components.inference.robots.robolab.composition.specs import (
            SPECS,
            task_class_name,
            task_target,
        )
        from ladder.components.inference.robots.robolab.droid_conversions import (
            DroidJointposConverter,
        )
        from ladder.components.inference.robots.robolab.oracle import build_oracle

        if _RoboLabKit.app is None:
            import signal
            import threading

            import cv2  # noqa: F401
            from isaaclab.app import AppLauncher

            os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            orig_signal = signal.signal
            if threading.current_thread() is not threading.main_thread():
                signal.signal = lambda *a, **kw: None  # type: ignore[assignment]
            try:
                _RoboLabKit.app = AppLauncher(headless=True, enable_cameras=True).app
            finally:
                signal.signal = orig_signal  # type: ignore[assignment]
            atexit.register(
                lambda: (
                    (sys.stdout.flush(), sys.stderr.flush(), os._exit(0))
                    if getattr(sys, "last_exc", None) is None
                    and getattr(sys, "last_value", None) is None
                    else None
                )
            )

            import carb.settings
            import robolab.constants

            robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
            carb.settings.get_settings().set(
                "/persistent/app/usd/muteUsdDiagnostics", False
            )

        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_jointpos import (
            auto_register_droid_envs,
        )

        auto_register_droid_envs(
            task_dirs=None,
            task=[
                task_target(self.task_name)
                if self.task_name in SPECS
                else self.task_name
            ],
            cameras=None,
        )
        cls_name = (
            task_class_name(self.task_name)
            if self.task_name in SPECS
            else self.task_name
        )
        if self._env is None:
            if _RoboLabKit.env is not None:
                _RoboLabKit.env.close()
                _RoboLabKit.env = None
            self._env, self._env_cfg = create_env(
                cls_name,
                device="cuda:0",
                num_envs=self.num_envs,
                instruction_type="default",
                policy="ladder_runner",
                renderer="realtime",
                rendering_mode="balanced",
            )
            _RoboLabKit.env = self._env
            self._oracle = build_oracle(
                getattr(self._env.cfg, "subtasks", None), self._env_cfg.instruction
            )
        else:
            self._env.reset_eval_state()

        obs, _ = self._env.reset()
        obs, _ = self._env.reset()
        self._raw_obs = obs
        self._converter = DroidJointposConverter(
            num_envs=self.num_envs, image_size=(self._obs_h, self._obs_w)
        )
        return self._build_robolab_obs_list(cls_name)

    def _step_robolab(self, actions: list[Any]) -> list[Any]:
        import omni.kit.app
        import omni.timeline

        assert self._env is not None and self._raw_obs is not None
        joints = self._raw_obs["proprio_obs"]["arm_joint_pos"].detach().cpu().numpy()
        commands = self._converter.actions_to_jointpos(actions, joints)
        timeline = omni.timeline.get_timeline_interface()
        kit_app = omni.kit.app.get_app()
        while not timeline.is_playing():
            kit_app.update()
        obs, _, _, _, _ = self._env.step(
            torch.as_tensor(commands, device=self._env.device)
        )
        self._raw_obs = obs
        return self._build_robolab_obs_list(
            self.task_name if self._env_cfg is None else type(self._env_cfg).__name__
        )

    def _build_robolab_obs_list(self, cls_name: str) -> list[dict[str, Any]]:
        from ladder.components.inference.robots.robolab.composition.specs import (
            SPECS,
            decomp_cache_entries,
            task_class_name,
        )

        image_obs = self._raw_obs["image_obs"]
        proprio = self._raw_obs["proprio_obs"]
        results = self._env.get_env_results()
        active = set(self._env.active_env_ids)
        plan_entries = decomp_cache_entries()
        ordered_map = {task_class_name(k): s.ordered for k, s in SPECS.items()}
        revisit_map = {task_class_name(k): bool(s.via_legs) for k, s in SPECS.items()}

        achieved_all: list[list[bool]] | None = None
        plan_subtasks: list[str] | None = None
        if self._oracle:
            commands = [g.command for g in self._oracle]
            cached = (plan_entries.get(cls_name) or {}).get("subtasks")
            plan_subtasks = (
                list(cached) if cached and len(cached) == len(commands) else commands
            )
            with contextlib.suppress(Exception):
                achieved_all = [
                    [bool(g.check(self._env, env_id=i)) for g in self._oracle]
                    for i in range(self._env.num_envs)
                ]

        out: list[dict[str, Any]] = []
        for i in range(self._env.num_envs):
            success = bool(results[i]["success"])
            truncated = i not in active
            oracle_block = (
                {
                    "subtasks": plan_subtasks,
                    "achieved": achieved_all[i],
                    "ordered": ordered_map.get(cls_name, False),
                    "revisits": revisit_map.get(cls_name, False),
                }
                if achieved_all is not None
                else {"subtasks": [self._env_cfg.instruction], "achieved": [success]}
            )
            metadata = {
                "env_name": cls_name,
                "robot_name": "franka",
                "instruction": self._env_cfg.instruction,
                "done": success,
                "truncated": truncated,
                "oracle": oracle_block,
            }
            out.append(
                self._converter.observation_dict(
                    env_i=i,
                    main_image=image_obs["over_shoulder_left_camera"][i]
                    .detach()
                    .cpu()
                    .numpy(),
                    wrist_image=image_obs["wrist_cam"][i].detach().cpu().numpy(),
                    joints=proprio["arm_joint_pos"][i].detach().cpu().numpy(),
                    gripper_position=proprio["gripper_pos"][i].detach().cpu().numpy(),
                    metadata=metadata,
                )
            )
        return out

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        episode_ids = [
            self._rollout_idx * self.num_envs + i for i in range(self.num_envs)
        ]
        self._rollout_idx += 1
        raw_obs_list = self._reset_backend(episode_ids)
        if options is not None and "env_idx" in options:
            self._reset_metrics(options["env_idx"])
        else:
            self._reset_metrics()
        self._is_start = False
        extracted_obs, _, _, _ = self._pack_raw_obs_list(raw_obs_list)
        return extracted_obs, {}

    def step(
        self,
        actions: Union[torch.Tensor, np.ndarray, None] = None,
        auto_reset: bool = True,
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        if actions is None:
            raise ValueError("Actions must be provided to LadderEnv.step().")

        act_np = (
            actions.detach().cpu().numpy().astype(np.float32)
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions, dtype=np.float32)
        )
        if act_np.ndim == 1:
            act_np = act_np[None, :]
        self._elapsed_steps += 1

        decoded = [self._decode_action(i, act_np[i]) for i in range(self.num_envs)]
        raw_obs_list = self._step_backend(decoded)
        extracted_obs, success_t, trunc_raw_t, progress_t = self._pack_raw_obs_list(
            raw_obs_list
        )

        terminations = success_t.clone()
        truncations = trunc_raw_t | (self._elapsed_steps >= self.max_episode_steps)
        step_reward = self._calc_step_reward(progress_t, success_t)

        infos: dict[str, Any] = {
            "success": success_t,
            "subgoal_progress": progress_t,
        }
        infos = self._record_metrics(step_reward, success_t, progress_t, infos)

        if self.ignore_terminations:
            if self.record_metrics:
                infos["episode"]["success_at_end"] = success_t.clone()
            terminations = torch.zeros_like(terminations)

        dones = terminations | truncations
        if dones.any() and auto_reset and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)

        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(
        self, chunk_actions: Union[torch.Tensor, np.ndarray]
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []

        for i in range(chunk_size):
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                chunk_actions[:, i], auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)
            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards_t = torch.stack(chunk_rewards, dim=1)
        raw_term_t = torch.stack(raw_chunk_terminations, dim=1)
        raw_trunc_t = torch.stack(raw_chunk_truncations, dim=1)

        past_terminations = raw_term_t.any(dim=1)
        past_truncations = raw_trunc_t.any(dim=1)
        past_dones = past_terminations | past_truncations

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros_like(raw_term_t)
        chunk_terminations[:, -1] = past_terminations
        chunk_truncations = torch.zeros_like(raw_trunc_t)
        chunk_truncations[:, -1] = past_truncations

        return (
            obs_list,
            chunk_rewards_t,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(
        self,
        dones: torch.Tensor,
        extracted_obs: dict[str, Any],
        infos: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        final_obs = _clone_obs_dict(extracted_obs)
        final_info = _clone_obs_dict(infos)
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options: dict[str, Any] = {"env_idx": env_idx}
        if self.use_fixed_reset_state_ids:
            options["episode_id"] = self.reset_state_ids[env_idx]
        extracted_obs, new_infos = self.reset(options=options)
        new_infos["final_observation"] = final_obs
        new_infos["final_info"] = final_info
        new_infos["_final_info"] = dones
        new_infos["_final_observation"] = dones
        new_infos["_elapsed_steps"] = dones
        return extracted_obs, new_infos

    def close(self) -> None:
        if self._env is not None and self.suite != "robolab":
            with contextlib.suppress(Exception):
                self._env.close()
        self._env = None
