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

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

PEG_INSERTION_SIDE_WIDE_ENV_ID = "PegInsertionSideWideClearance-v1"
PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID = (
    "PegInsertionSideWideClearanceObserverWideWrist-v1"
)
PEG_INSERTION_SIDE_HARD_ENV_ID = "PegInsertionSideHardClearance-v1"
PEG_INSERTION_SIDE_HARD_OBSERVER_WIDE_WRIST_ENV_ID = (
    "PegInsertionSideHardClearanceObserverWideWrist-v1"
)
PANDA_WIDE_WRISTCAM_UID = "panda_wristcam_wide"
PEG_INSERTION_SIDE_BASE_CLEARANCE = 0.003
PEG_INSERTION_SIDE_WIDE_CLEARANCE = 0.02
PEG_INSERTION_SIDE_HARD_CLEARANCE = 0.01
RLT_OPENPI_JOINT_WRAP_MODE = "rlt_openpi_joint"
OPENPI_EEF_WRAP_MODE = "openpi_eef_rot6d"

_RLT_JOINT_STATE_DIM = 9
_RLT_MAIN_CAMERA_KEY = "3rd_view_camera"
_RLT_WRIST_CAMERA_KEY = "wide_hand_camera"
_RLT_DEFAULT_PROMPT = "insert the peg in the hole"
_RLT_LEGACY_PEG_PROMPT = "insert the peg into the hole"

_PEG_VARIANTS_REGISTERED = False


def is_peg_insertion_side_env_id(env_id: str | None) -> bool:
    return env_id in {
        PEG_INSERTION_SIDE_WIDE_ENV_ID,
        PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID,
        PEG_INSERTION_SIDE_HARD_ENV_ID,
        PEG_INSERTION_SIDE_HARD_OBSERVER_WIDE_WRIST_ENV_ID,
    }


def get_joint_observer_env_id(env_id: str | None) -> str | None:
    if env_id in {
        PEG_INSERTION_SIDE_WIDE_ENV_ID,
        PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID,
    }:
        return PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID
    if env_id in {
        PEG_INSERTION_SIDE_HARD_ENV_ID,
        PEG_INSERTION_SIDE_HARD_OBSERVER_WIDE_WRIST_ENV_ID,
    }:
        return PEG_INSERTION_SIDE_HARD_OBSERVER_WIDE_WRIST_ENV_ID
    return None


def patch_rlt_openpi_joint_env_args(
    env_args: dict[str, Any],
    *,
    wrap_obs_mode: str,
) -> dict[str, Any]:
    """Patch ManiSkill env args for the RLT OpenPI joint observer variant."""
    if wrap_obs_mode not in (RLT_OPENPI_JOINT_WRAP_MODE, OPENPI_EEF_WRAP_MODE):
        return env_args
    if not is_peg_insertion_side_env_id(env_args.get("id")):
        return env_args

    if wrap_obs_mode == OPENPI_EEF_WRAP_MODE:
        if env_args.get("sim_backend") not in {"gpu", "physx_cuda"}:
            raise ValueError(
                "EEF rot6d collection, evaluation and rollout require GPU simulation and GPU IK."
            )
        if env_args.get("control_mode") != "pd_ee_delta_pose":
            raise ValueError("EEF rot6d requires control_mode=pd_ee_delta_pose.")
    register_rlinf_peg_insertion_side_variants()
    observer_env_id = get_joint_observer_env_id(env_args.get("id"))
    if observer_env_id is not None:
        env_args["id"] = observer_env_id
    env_args.setdefault("robot_uids", PANDA_WIDE_WRISTCAM_UID)
    sensor_configs = env_args.setdefault("sensor_configs", {})
    sensor_configs.setdefault("width", 384)
    sensor_configs.setdefault("height", 384)
    return env_args


def validate_eef_gpu_controller(env: Any) -> dict[str, Any]:
    """Check the shared EEF controller contract and return its provenance."""
    task = env.unwrapped
    if not task.scene.gpu_sim_enabled or task.device.type != "cuda":
        raise ValueError(
            "EEF rot6d requires GPU simulation; CPU execution is incompatible with GPU demonstrations."
        )
    if task.control_mode != "pd_ee_delta_pose":
        raise ValueError(f"Expected pd_ee_delta_pose, got {task.control_mode}.")
    arm = task.agent.controller.controllers["arm"]
    cfg = arm.config
    if not arm.kinematics.use_gpu_ik:
        raise ValueError("EEF rot6d requires the GPU kinematics solver.")
    if (
        not cfg.use_delta
        or cfg.use_target
        or not cfg.normalize_action
        or cfg.interpolate
        or cfg.frame != "root_translation:root_aligned_body_rotation"
        or cfg.delta_solver_config != {"type": "levenberg_marquardt", "alpha": 1.0}
        or cfg.rot_lower != -0.1
        or cfg.rot_upper != 0.1
        or cfg.pos_lower != -0.1
        or cfg.pos_upper != 0.1
    ):
        raise ValueError(
            "Expected normalized root-frame EEF deltas, GPU LM alpha=1, no target accumulation or interpolation, and +/-0.1 bounds."
        )
    return {
        "sim_backend": "gpu",
        "ik_backend": "gpu_pytorch_kinematics_lm",
        "control_mode": task.control_mode,
        "frame": cfg.frame,
        "delta_solver_config": dict(cfg.delta_solver_config),
        "rotation_scale": cfg.rot_lower,
        "position_bounds": [cfg.pos_lower, cfg.pos_upper],
        "use_target": cfg.use_target,
        "interpolate": cfg.interpolate,
    }


def normalize_peg_instruction(instruction):
    if isinstance(instruction, str) and instruction == _RLT_LEGACY_PEG_PROMPT:
        return _RLT_DEFAULT_PROMPT
    return instruction


def normalize_peg_instructions(instructions, *, num_envs: int):
    if isinstance(instructions, str):
        return [normalize_peg_instruction(instructions) for _ in range(num_envs)]
    return [normalize_peg_instruction(item) for item in instructions]


def default_peg_instruction(*, num_envs: int) -> list[str]:
    return [_RLT_DEFAULT_PROMPT for _ in range(num_envs)]


def resolve_maniskill_task_descriptions(
    env,
    *,
    num_envs: int,
    is_peg_insertion_side: bool,
):
    if hasattr(env, "get_language_instruction"):
        instruction = env.get_language_instruction()
        if instruction is not None:
            return _format_task_descriptions(
                instruction,
                num_envs=num_envs,
                is_peg_insertion_side=is_peg_insertion_side,
            )

    for attr in ("task_descriptions", "task_description", "task_prompt", "instruction"):
        if not hasattr(env, attr):
            continue
        instruction = getattr(env, attr)
        if instruction is None:
            continue
        return _format_task_descriptions(
            instruction,
            num_envs=num_envs,
            is_peg_insertion_side=is_peg_insertion_side,
        )

    if is_peg_insertion_side:
        return default_peg_instruction(num_envs=num_envs)
    return ["" for _ in range(num_envs)]


def _format_task_descriptions(
    instruction,
    *,
    num_envs: int,
    is_peg_insertion_side: bool,
):
    if isinstance(instruction, str):
        instruction = [instruction for _ in range(num_envs)]
    if is_peg_insertion_side:
        return normalize_peg_instructions(instruction, num_envs=num_envs)
    return instruction


def wrap_rlt_openpi_joint_obs(
    raw_obs: dict[str, Any],
    *,
    infos: dict[str, Any] | None,
    task_descriptions,
    num_envs: int,
    device: torch.device,
    is_peg_insertion_side: bool,
    eef_state: bool = False,
) -> dict[str, Any]:
    sensor_data = raw_obs.pop("sensor_data")
    raw_obs.pop("sensor_param")

    main_images = sensor_data[_RLT_MAIN_CAMERA_KEY]["rgb"]
    wrist_images = sensor_data[_RLT_WRIST_CAMERA_KEY]["rgb"]

    if infos is not None and "prompt" in infos:
        task_descriptions = infos["prompt"]
        if is_peg_insertion_side:
            task_descriptions = normalize_peg_instructions(
                task_descriptions,
                num_envs=num_envs,
            )

    return {
        "main_images": main_images,
        "wrist_images": wrist_images,
        "extra_view_images": None,
        "states": (
            extract_openpi_eef_states if eef_state else extract_rlt_joint_states
        )(
            raw_obs,
            batch_size=main_images.shape[0],
            device=device,
        ),
        "task_descriptions": task_descriptions,
    }


def extract_rlt_joint_states(raw_obs: dict[str, Any], *, batch_size: int, device):
    qpos = raw_obs["agent"]["qpos"]
    return torch.stack(
        [_extract_rlt_joint_state(qpos[index], device) for index in range(batch_size)],
        dim=0,
    )


def extract_openpi_eef_states(
    raw_obs: dict[str, Any], *, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Return world TCP xyz, column-based rot6d and total finger opening (m)."""
    from rlinf.utils.torch_rot6d import quaternion_wxyz_to_rot6d

    pose = torch.as_tensor(raw_obs["extra"]["tcp_pose"], device=device).float()
    qpos = torch.as_tensor(raw_obs["agent"]["qpos"], device=device).float()
    if pose.shape != (batch_size, 7) or qpos.shape != (batch_size, 9):
        raise ValueError(
            f"Expected TCP [{batch_size},7] and Panda qpos [{batch_size},9], "
            f"got {tuple(pose.shape)} and {tuple(qpos.shape)}."
        )
    quat = pose[:, 3:]
    norm = quat.norm(dim=-1, keepdim=True)
    if (
        not torch.isfinite(pose).all()
        or not torch.isfinite(qpos).all()
        or (norm < 1e-8).any()
    ):
        raise ValueError("Expected finite EEF state and a nonzero TCP quaternion.")
    rotation = quaternion_wxyz_to_rot6d(quat / norm)
    return torch.cat(
        (pose[:, :3], rotation, qpos[:, 7:9].sum(-1, keepdim=True)), dim=-1
    )


def init_peg_insertion_event_state(*, num_envs: int, device) -> dict[str, torch.Tensor]:
    return {
        "grasp_count": torch.zeros(num_envs, device=device, dtype=torch.int32),
        "consecutive_grasp_once": torch.zeros(
            num_envs,
            device=device,
            dtype=torch.bool,
        ),
        "prealign_once": torch.zeros(num_envs, device=device, dtype=torch.bool),
        "partial_insert_once": torch.zeros(num_envs, device=device, dtype=torch.bool),
        "success_once": torch.zeros(num_envs, device=device, dtype=torch.bool),
    }


def reset_peg_insertion_event_state(
    state: dict[str, torch.Tensor],
    *,
    env_idx=None,
) -> None:
    if env_idx is None:
        for value in state.values():
            value.zero_()
        return

    for value in state.values():
        value[env_idx] = 0


def snapshot_peg_insertion_event_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in state.items()}


def restore_peg_insertion_event_state(
    state: dict[str, torch.Tensor],
    snapshot: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> None:
    for key, value in snapshot.items():
        state[key][mask] = value[mask]


def augment_peg_insertion_info(
    *,
    env,
    infos: dict[str, Any],
    event_state: dict[str, torch.Tensor],
    device,
) -> dict[str, Any]:
    peg_head_pos_at_hole = infos.get("peg_head_pos_at_hole")
    if peg_head_pos_at_hole is None:
        peg_head_pos_at_hole = (env.box_hole_pose.inv() * env.peg_head_pose).p
    peg_head_pos_at_hole = peg_head_pos_at_hole.to(device, dtype=torch.float32)

    peg_head_wrt_goal = env.goal_pose.inv() * env.peg_head_pose
    peg_wrt_goal = env.goal_pose.inv() * env.peg.pose
    peg_head_goal_yz_dist = torch.linalg.norm(
        peg_head_wrt_goal.p[:, 1:],
        dim=1,
    ).to(torch.float32)
    peg_body_goal_yz_dist = torch.linalg.norm(
        peg_wrt_goal.p[:, 1:],
        dim=1,
    ).to(torch.float32)
    tcp_pos = env.agent.tcp.pose.p.to(device, dtype=torch.float32)
    peg_pos = env.peg.pose.p.to(device, dtype=torch.float32)
    tcp_peg_dist = torch.linalg.norm(tcp_pos - peg_pos, dim=1).to(torch.float32)

    is_grasped_current = env.agent.is_grasping(env.peg, max_angle=20)
    event_state["grasp_count"] = torch.where(
        is_grasped_current,
        event_state["grasp_count"] + 1,
        torch.zeros_like(event_state["grasp_count"]),
    )
    consecutive_grasp_current = event_state["grasp_count"] >= 5

    prealigned_current = (peg_head_goal_yz_dist < 0.01) & (peg_body_goal_yz_dist < 0.01)

    hole_radii = env.box_hole_radii.to(device, dtype=torch.float32)
    peg_head_hole_x = peg_head_pos_at_hole[:, 0]
    peg_head_hole_abs_y = torch.abs(peg_head_pos_at_hole[:, 1])
    peg_head_hole_abs_z = torch.abs(peg_head_pos_at_hole[:, 2])
    partial_insert_current = (
        prealigned_current
        & (peg_head_hole_x >= -0.05)
        & (peg_head_hole_abs_y <= 1.25 * hole_radii)
        & (peg_head_hole_abs_z <= 1.25 * hole_radii)
    )

    success_current = infos.get("success")
    if success_current is None:
        success_current = (
            (peg_head_hole_x >= -0.015)
            & (peg_head_hole_abs_y <= hole_radii)
            & (peg_head_hole_abs_z <= hole_radii)
        )
    success_current = success_current.to(device, dtype=torch.bool)

    consecutive_grasp_event = consecutive_grasp_current & (
        ~event_state["consecutive_grasp_once"]
    )
    prealign_event = prealigned_current & (~event_state["prealign_once"])
    partial_insert_event = partial_insert_current & (
        ~event_state["partial_insert_once"]
    )
    success_event = success_current & (~event_state["success_once"])

    event_state["consecutive_grasp_once"] |= consecutive_grasp_current
    event_state["prealign_once"] |= prealigned_current
    event_state["partial_insert_once"] |= partial_insert_current
    event_state["success_once"] |= success_current

    infos.update(
        {
            "peg_head_pos_at_hole": peg_head_pos_at_hole,
            "peg_head_hole_x": peg_head_hole_x,
            "peg_head_hole_abs_y": peg_head_hole_abs_y,
            "peg_head_hole_abs_z": peg_head_hole_abs_z,
            "peg_head_goal_yz_dist": peg_head_goal_yz_dist,
            "peg_body_goal_yz_dist": peg_body_goal_yz_dist,
            "hole_radii": hole_radii,
            "tcp_peg_dist": tcp_peg_dist,
            "is_grasped_current": is_grasped_current,
            "consecutive_grasp_current": consecutive_grasp_current,
            "prealigned_current": prealigned_current,
            "partial_insert_current": partial_insert_current,
            "success_current": success_current,
            "consecutive_grasp_event": consecutive_grasp_event,
            "prealign_event": prealign_event,
            "partial_insert_event": partial_insert_event,
            "success_event": success_event,
            "consecutive_grasp_once": event_state["consecutive_grasp_once"].clone(),
            "prealign_once": event_state["prealign_once"].clone(),
            "partial_insert_once": event_state["partial_insert_once"].clone(),
            "success_once": event_state["success_once"].clone(),
            "success": success_current,
        }
    )
    return infos


def maybe_augment_peg_insertion_info(
    *,
    env,
    infos: dict[str, Any],
    event_state: dict[str, torch.Tensor] | None,
    device,
    is_peg_insertion_side: bool,
) -> dict[str, Any]:
    if not is_peg_insertion_side or event_state is None:
        return infos
    return augment_peg_insertion_info(
        env=env,
        infos=infos,
        event_state=event_state,
        device=device,
    )


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _extract_rlt_joint_state(qpos, device):
    qpos = _to_numpy(qpos).astype(np.float32)
    if qpos.shape[0] < _RLT_JOINT_STATE_DIM:
        raise ValueError(
            f"Expected Panda qpos with at least {_RLT_JOINT_STATE_DIM} dims, "
            f"got {qpos.shape}"
        )
    return torch.as_tensor(
        qpos[:_RLT_JOINT_STATE_DIM],
        device=device,
        dtype=torch.float32,
    )


def register_rlinf_peg_insertion_side_variants() -> None:
    global _PEG_VARIANTS_REGISTERED
    if _PEG_VARIANTS_REGISTERED:
        return
    required_env_ids = {
        PEG_INSERTION_SIDE_WIDE_ENV_ID,
        PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID,
        PEG_INSERTION_SIDE_HARD_ENV_ID,
        PEG_INSERTION_SIDE_HARD_OBSERVER_WIDE_WRIST_ENV_ID,
    }
    if all(env_id in gym.registry for env_id in required_env_ids):
        _PEG_VARIANTS_REGISTERED = True
        return

    import sapien
    import torch
    from mani_skill.agents.registration import register_agent
    from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
    from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import common, sapien_utils
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.scene_builder.table import TableSceneBuilder
    from mani_skill.utils.structs.actor import Actor
    from mani_skill.utils.structs.pose import Pose

    def _build_box_with_hole(
        scene,
        inner_radius: float,
        outer_radius: float,
        depth: float,
        center=(0, 0),
    ):
        builder = scene.create_actor_builder()
        thickness = (outer_radius - inner_radius) * 0.5
        # x-axis is hole direction
        half_center = [x * 0.5 for x in center]
        half_sizes = [
            [depth, thickness - half_center[0], outer_radius],
            [depth, thickness + half_center[0], outer_radius],
            [depth, outer_radius, thickness - half_center[1]],
            [depth, outer_radius, thickness + half_center[1]],
        ]
        offset = thickness + inner_radius
        poses = [
            sapien.Pose([0, offset + half_center[0], 0]),
            sapien.Pose([0, -offset + half_center[0], 0]),
            sapien.Pose([0, 0, offset + half_center[1]]),
            sapien.Pose([0, 0, -offset + half_center[1]]),
        ]
        mat = sapien.render.RenderMaterial(
            base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
        )
        for half_size, pose in zip(half_sizes, poses):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)
        return builder

    @register_agent()
    class PandaWristWideCam(PandaWristCam):  # type: ignore[unused-ignore]
        uid = PANDA_WIDE_WRISTCAM_UID

        @property
        def _sensor_configs(self):
            configs = list(super()._sensor_configs)
            tilt = np.deg2rad(-15.0)
            configs.append(
                CameraConfig(
                    uid="wide_hand_camera",
                    pose=sapien.Pose(
                        p=[-0.055, 0.0, 0.035],
                        q=[float(np.cos(tilt / 2)), 0.0, float(np.sin(tilt / 2)), 0.0],
                    ),
                    width=384,
                    height=384,
                    fov=1.45,
                    near=0.01,
                    far=100,
                    mount=self.robot.links_map["camera_link"],
                )
            )
            return configs

    def _observer_sensor_configs(base_env) -> list[CameraConfig]:
        configs = list(base_env._default_sensor_configs)
        pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        configs.append(
            CameraConfig(
                "3rd_view_camera",
                pose,
                384,
                384,
                1.0,
                0.01,
                100,
            )
        )
        return configs

    class _PegInsertionSideWideSceneMixin:
        hole_clearance = PEG_INSERTION_SIDE_WIDE_CLEARANCE
        success_clearance = PEG_INSERTION_SIDE_WIDE_CLEARANCE

        def __init__(
            self,
            *args,
            fixed_peg_half_length: float | None = None,
            fixed_peg_half_width: float | None = None,
            hole_clearance: float | None = None,
            preinsert: dict[str, float] | None = None,
            **kwargs,
        ):
            if hole_clearance is not None:
                if not np.isfinite(hole_clearance) or hole_clearance <= 0:
                    raise ValueError(
                        "Expected a finite positive per-side hole clearance."
                    )
                self.hole_clearance = float(hole_clearance)
                self.success_clearance = float(hole_clearance)
            if (fixed_peg_half_length is None) != (fixed_peg_half_width is None):
                raise ValueError(
                    "Set both fixed peg half-length and half-width, or neither."
                )
            if fixed_peg_half_length is not None and (
                not np.isfinite([fixed_peg_half_length, fixed_peg_half_width]).all()
                or fixed_peg_half_width <= 0
                or fixed_peg_half_length <= fixed_peg_half_width + self.hole_clearance
            ):
                raise ValueError(
                    "Expected finite positive peg dimensions with half-length > half-width + clearance."
                )
            from .peg_preinsert import PegPreinsertConfig

            self.preinsert = (
                None if preinsert is None else PegPreinsertConfig(**preinsert)
            )
            self.fixed_peg_half_length = fixed_peg_half_length
            self.fixed_peg_half_width = fixed_peg_half_width
            super().__init__(*args, **kwargs)

        @property
        def _default_human_render_camera_configs(self):
            if self.preinsert is None:
                return super()._default_human_render_camera_configs
            return CameraConfig(
                "render_camera",
                sapien_utils.look_at([0.45, -0.35, 0.35], [0, 0.15, 0.11]),
                768,
                512,
                1.0,
                0.01,
                100,
            )

        def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
            super()._initialize_episode(env_idx, options)
            if self.preinsert is not None:
                from .peg_preinsert import initialize_peg_preinsert

                initialize_peg_preinsert(self, env_idx, self.preinsert)

        def _load_scene(self, options: dict):
            del options
            with torch.device(self.device):
                self.table_scene = TableSceneBuilder(self)
                self.table_scene.build()
                lengths = self._batched_episode_rng.uniform(0.085, 0.125)
                radii = self._batched_episode_rng.uniform(0.015, 0.025)
                if self.fixed_peg_half_length is not None:
                    lengths = np.full(self.num_envs, self.fixed_peg_half_length)
                    radii = np.full(self.num_envs, self.fixed_peg_half_width)
                centers = np.zeros((self.num_envs, 2), dtype=np.float32)
                self.peg_half_sizes = common.to_tensor(
                    np.vstack([lengths, radii, radii])
                ).T
                peg_head_offsets = torch.zeros((self.num_envs, 3))
                peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
                self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)
                box_hole_offsets = torch.zeros((self.num_envs, 3))
                box_hole_offsets[:, 1:] = common.to_tensor(centers)
                self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
                self.box_hole_radii = common.to_tensor(radii + self.success_clearance)

                pegs = []
                boxes = []
                for i in range(self.num_envs):
                    scene_idxs = [i]
                    length = lengths[i]
                    radius = radii[i]

                    builder = self.scene.create_actor_builder()
                    builder.add_box_collision(half_size=[length, radius, radius])
                    mat = sapien.render.RenderMaterial(
                        base_color=sapien_utils.hex2rgba("#EC7357"),
                        roughness=0.5,
                        specular=0.5,
                    )
                    builder.add_box_visual(
                        sapien.Pose([length / 2, 0, 0]),
                        half_size=[length / 2, radius, radius],
                        material=mat,
                    )
                    mat = sapien.render.RenderMaterial(
                        base_color=sapien_utils.hex2rgba("#EDF6F9"),
                        roughness=0.5,
                        specular=0.5,
                    )
                    builder.add_box_visual(
                        sapien.Pose([-length / 2, 0, 0]),
                        half_size=[length / 2, radius, radius],
                        material=mat,
                    )
                    builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                    builder.set_scene_idxs(scene_idxs)
                    peg = builder.build(f"peg_{i}")
                    self.remove_from_state_dict_registry(peg)

                    inner_radius, outer_radius, depth = (
                        radius + self.hole_clearance,
                        length,
                        length,
                    )
                    builder = _build_box_with_hole(
                        self.scene,
                        inner_radius,
                        outer_radius,
                        depth,
                        center=centers[i],
                    )
                    builder.initial_pose = sapien.Pose(p=[0, 1, 0.1])
                    builder.set_scene_idxs(scene_idxs)
                    box = builder.build_kinematic(f"box_with_hole_{i}")
                    self.remove_from_state_dict_registry(box)
                    pegs.append(peg)
                    boxes.append(box)

                self.peg = Actor.merge(pegs, "peg")
                self.box = Actor.merge(boxes, "box_with_hole")
                self.add_to_state_dict_registry(self.peg)
                self.add_to_state_dict_registry(self.box)

    class _PegInsertionSideHardSceneMixin(_PegInsertionSideWideSceneMixin):
        hole_clearance = PEG_INSERTION_SIDE_HARD_CLEARANCE
        success_clearance = PEG_INSERTION_SIDE_HARD_CLEARANCE

    @register_env(PEG_INSERTION_SIDE_WIDE_ENV_ID, max_episode_steps=100)
    class PegInsertionSideWideClearanceEnv(
        _PegInsertionSideWideSceneMixin, PegInsertionSideEnv
    ):  # type: ignore[unused-ignore]
        _clearance = PEG_INSERTION_SIDE_BASE_CLEARANCE

    @register_env(
        PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID,
        max_episode_steps=100,
    )
    class PegInsertionSideWideClearanceObserverWideWristEnv(
        PegInsertionSideWideClearanceEnv
    ):  # type: ignore[unused-ignore]
        SUPPORTED_ROBOTS = ["panda_wristcam", PANDA_WIDE_WRISTCAM_UID]

        def __init__(self, *args, robot_uids=PANDA_WIDE_WRISTCAM_UID, **kwargs):
            super().__init__(*args, robot_uids=robot_uids, **kwargs)

        @property
        def _default_sensor_configs(self):
            return _observer_sensor_configs(super())

    @register_env(PEG_INSERTION_SIDE_HARD_ENV_ID, max_episode_steps=100)
    class PegInsertionSideHardClearanceEnv(
        _PegInsertionSideHardSceneMixin, PegInsertionSideEnv
    ):  # type: ignore[unused-ignore]
        _clearance = PEG_INSERTION_SIDE_BASE_CLEARANCE

    @register_env(
        PEG_INSERTION_SIDE_HARD_OBSERVER_WIDE_WRIST_ENV_ID,
        max_episode_steps=100,
    )
    class PegInsertionSideHardClearanceObserverWideWristEnv(
        PegInsertionSideHardClearanceEnv
    ):  # type: ignore[unused-ignore]
        SUPPORTED_ROBOTS = ["panda_wristcam", PANDA_WIDE_WRISTCAM_UID]

        def __init__(self, *args, robot_uids=PANDA_WIDE_WRISTCAM_UID, **kwargs):
            super().__init__(*args, robot_uids=robot_uids, **kwargs)

        @property
        def _default_sensor_configs(self):
            return _observer_sensor_configs(super())

    _PEG_VARIANTS_REGISTERED = True
