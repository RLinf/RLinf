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

import copy
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.utils.ik_solver import process_gripper_command
from pytorch3d import transforms

from rlinf.utils.logging import get_logger


def infer_state_count(node):
    if isinstance(node, torch.Tensor) and node.ndim > 0:
        return int(node.shape[0])
    if isinstance(node, dict):
        for value in node.values():
            count = infer_state_count(value)
            if count is not None:
                return count
        return None
    if hasattr(node, "__dict__"):
        for value in vars(node).values():
            count = infer_state_count(value)
            if count is not None:
                return count
    return None


def slice_state_tree(node, indices: torch.Tensor, state_count: int):
    if isinstance(node, torch.Tensor):
        if node.ndim > 0 and node.shape[0] == state_count:
            return node.index_select(0, indices.to(node.device))
        return node

    if isinstance(node, dict):
        return {
            key: slice_state_tree(value, indices, state_count)
            for key, value in node.items()
        }

    if isinstance(node, list):
        return [slice_state_tree(value, indices, state_count) for value in node]

    if isinstance(node, tuple):
        return tuple(slice_state_tree(value, indices, state_count) for value in node)

    if hasattr(node, "__dict__"):
        cloned = copy.deepcopy(node)
        for key, value in vars(node).items():
            setattr(cloned, key, slice_state_tree(value, indices, state_count))
        return cloned

    return node


def select_initial_states(base_states, reset_state_ids):
    """Select per-env initial states for built-in and custom task state formats."""
    if base_states is None:
        return None

    reset_state_ids = np.asarray(reset_state_ids, dtype=np.int64).reshape(-1)
    if reset_state_ids.size == 0:
        return None

    if isinstance(base_states, (list, tuple)):
        if len(base_states) == 0:
            return None
        state_count = len(base_states)
        reset_state_ids = reset_state_ids % state_count
        return [copy.deepcopy(base_states[int(i)]) for i in reset_state_ids]

    state_count = infer_state_count(base_states)
    if state_count is None or state_count <= 0:
        return copy.deepcopy(base_states)

    reset_state_ids = reset_state_ids % state_count
    indices = torch.as_tensor(reset_state_ids, dtype=torch.long)
    return slice_state_tree(base_states, indices, state_count)


def get_valid_joint_names(
    env: Any, robot_name: str, last_obs: Any = None, robot_cfg: Any = None
):
    valid_joints = None
    try:
        if hasattr(env, "handler") and hasattr(env.handler, "get_joint_names"):
            joint_names = env.handler.get_joint_names(robot_name)
            if isinstance(joint_names, (list, tuple)):
                valid_joints = set(joint_names)

        if (
            valid_joints is None
            and last_obs is not None
            and hasattr(last_obs, "robots")
        ):
            rs = last_obs.robots.get(robot_name)
            if rs is not None:
                if hasattr(rs, "joint_names"):
                    valid_joints = set(rs.joint_names)

        if valid_joints is None and robot_cfg is not None:
            if hasattr(robot_cfg, "joint_names"):
                valid_joints = set(robot_cfg.joint_names)
            elif hasattr(robot_cfg, "actuators"):
                joint_names = []
                for actuator in robot_cfg.actuators:
                    if hasattr(actuator, "joint_name"):
                        joint_names.append(actuator.joint_name)
                if joint_names:
                    valid_joints = set(joint_names)
    except Exception:
        pass

    return valid_joints


def cfg_get(cfg: Any, init_params: Any, name: str, default: Any) -> Any:
    if init_params is not None and hasattr(init_params, name):
        return getattr(init_params, name)
    return getattr(cfg, name, default)


def build_roboverse_camera_cfgs(cfg: Any, task_cls: Any, robot_name: str):
    init_params = getattr(cfg, "init_params", None)

    main_camera_name = str(cfg_get(cfg, init_params, "camera_name", "main_camera"))
    wrist_camera_name = str(
        cfg_get(cfg, init_params, "wrist_camera_name", "robot0_eye_in_hand")
    )
    camera_width = int(
        cfg_get(
            cfg,
            init_params,
            "camera_widths",
            cfg_get(cfg, init_params, "camera_width", 256),
        )
    )
    camera_height = int(
        cfg_get(
            cfg,
            init_params,
            "camera_heights",
            cfg_get(cfg, init_params, "camera_height", 256),
        )
    )
    camera_pos = tuple(cfg_get(cfg, init_params, "camera_pos", (1.0, 0.0, 1.22)))
    camera_look_at = tuple(
        cfg_get(cfg, init_params, "camera_look_at", (-0.02, 0.0, 0.64))
    )
    camera_focal_length = float(cfg_get(cfg, init_params, "camera_focal_length", 40.0))
    camera_horizontal_aperture = float(
        cfg_get(cfg, init_params, "camera_horizontal_aperture", 20.955)
    )
    enable_wrist_camera = bool(cfg_get(cfg, init_params, "enable_wrist_camera", True))
    wrist_camera_width = int(
        cfg_get(cfg, init_params, "wrist_camera_widths", camera_width)
    )
    wrist_camera_height = int(
        cfg_get(cfg, init_params, "wrist_camera_heights", camera_height)
    )
    wrist_camera_focal_length = float(
        cfg_get(cfg, init_params, "wrist_camera_focal_length", 24.0)
    )
    wrist_mount_to = str(cfg_get(cfg, init_params, "wrist_camera_mount_to", robot_name))
    wrist_mount_link = str(
        cfg_get(cfg, init_params, "wrist_camera_mount_link", "panda_hand")
    )
    wrist_mount_pos = tuple(
        cfg_get(cfg, init_params, "wrist_camera_mount_pos", (0.1, 0.0, 0.09))
    )
    wrist_mount_quat = tuple(
        cfg_get(cfg, init_params, "wrist_camera_mount_quat", (0.5, 0.5, -0.5, -0.5))
    )

    cameras = list(getattr(task_cls.scenario, "cameras", []))

    def upsert_camera(cam_cfg: PinholeCameraCfg):
        cam_name = getattr(cam_cfg, "name", None)
        for idx, cam in enumerate(cameras):
            if getattr(cam, "name", None) == cam_name:
                updated = list(cameras)
                updated[idx] = cam_cfg
                return updated
        updated = list(cameras)
        updated.append(cam_cfg)
        return updated

    cameras = upsert_camera(
        PinholeCameraCfg(
            name=main_camera_name,
            width=camera_width,
            height=camera_height,
            pos=camera_pos,
            look_at=camera_look_at,
            focal_length=camera_focal_length,
            clipping_range=(0.01, 100.0),
            horizontal_aperture=camera_horizontal_aperture,
        )
    )

    if enable_wrist_camera:
        cameras = upsert_camera(
            PinholeCameraCfg(
                name=wrist_camera_name,
                width=wrist_camera_width,
                height=wrist_camera_height,
                focal_length=wrist_camera_focal_length,
                mount_to=wrist_mount_to,
                mount_link=wrist_mount_link,
                mount_pos=wrist_mount_pos,
                mount_quat=wrist_mount_quat,
                clipping_range=(0.01, 100.0),
            )
        )

    return main_camera_name, wrist_camera_name, cameras


def resolve_camera_rgb(raw_obs: Any, preferred_name: str, prefer_wrist: bool):
    cameras = getattr(raw_obs, "cameras", None)
    if not isinstance(cameras, dict) or len(cameras) == 0:
        msg = "No camera RGB found in observation."
        logger = get_logger()
        logger.error(msg)
        raise KeyError(msg)

    if preferred_name in cameras:
        return preferred_name, cameras[preferred_name].rgb

    camera_names = list(cameras.keys())
    name_lowers = {name: str(name).lower() for name in camera_names}
    wrist_keywords = ("wrist", "eye_in_hand", "hand", "gripper")

    if prefer_wrist:
        for name in camera_names:
            if any(keyword in name_lowers[name] for keyword in wrist_keywords):
                return name, cameras[name].rgb
        msg = f"Wrist camera '{preferred_name}' not found. Available cameras: {camera_names}"
        logger = get_logger()
        logger.error(msg)
        raise KeyError(msg)

    for name in camera_names:
        if not any(keyword in name_lowers[name] for keyword in wrist_keywords):
            return name, cameras[name].rgb
    return camera_names[0], cameras[camera_names[0]].rgb


def build_policy_states(
    raw_obs: Any,
    robot_name: str,
    ee_body_name: str,
    device,
    inverse_reorder_idx=None,
    ee_body_idx: Optional[int] = None,
    state_dim: Optional[int] = None,
):
    rs = raw_obs.robots[robot_name]

    def to_f32(x):
        return (
            (x if isinstance(x, torch.Tensor) else torch.tensor(x)).to(device).float()
        )

    body_state = to_f32(rs.body_state)
    joint_pos = to_f32(rs.joint_pos).reshape(body_state.shape[0], -1)

    if ee_body_idx is None:
        ee_body_idx = rs.body_names.index(ee_body_name)

    ee_pos_world = body_state[:, ee_body_idx, 0:3]
    ee_quat_world = body_state[:, ee_body_idx, 3:7]
    ee_rot_world = transforms.quaternion_to_axis_angle(ee_quat_world).to(
        dtype=torch.float32
    )

    if inverse_reorder_idx is None:
        msg = "inverse_reorder_idx must be provided by RoboVerseEnv to build policy states."
        logger = get_logger()
        logger.error(msg)
        raise ValueError(msg)
    joint_pos_reordered = joint_pos[:, inverse_reorder_idx]
    if joint_pos_reordered.shape[1] >= 2:
        gripper_state = joint_pos_reordered[:, -2:]
    elif joint_pos_reordered.shape[1] == 1:
        gripper_state = torch.cat(
            [joint_pos_reordered[:, -1:], joint_pos_reordered[:, -1:]], dim=1
        )
    else:
        gripper_state = torch.zeros(
            (body_state.shape[0], 2), dtype=torch.float32, device=device
        )

    gripper_state[:, 1] = -gripper_state[:, 1]
    states_tensor = torch.cat([ee_pos_world, ee_rot_world, gripper_state], dim=1).to(
        dtype=torch.float32
    )

    if state_dim is not None:
        current_dim = states_tensor.shape[1]
        if current_dim < state_dim:
            states_tensor = F.pad(states_tensor, (0, state_dim - current_dim))
        elif current_dim > state_dim:
            states_tensor = states_tensor[:, :state_dim]
    return states_tensor, ee_body_idx


def extract_roboverse_obs(
    raw_obs: Any,
    main_camera_name: str,
    wrist_camera_name: Optional[str],
    robot_name: str,
    ee_body_name: str,
    device,
    inverse_reorder_idx=None,
    ee_body_idx: Optional[int] = None,
    state_dim: Optional[int] = None,
    has_wrist_camera: bool = True,
):
    _, main_rgb = resolve_camera_rgb(raw_obs, main_camera_name, prefer_wrist=False)
    main_rgb = torch.flip(main_rgb, dims=[2])

    wrist_rgb = None
    if has_wrist_camera and wrist_camera_name is not None:
        _, wrist_rgb = resolve_camera_rgb(raw_obs, wrist_camera_name, prefer_wrist=True)
        wrist_rgb = torch.flip(wrist_rgb, dims=[2])

    states_tensor, ee_body_idx = build_policy_states(
        raw_obs=raw_obs,
        robot_name=robot_name,
        ee_body_name=ee_body_name,
        inverse_reorder_idx=inverse_reorder_idx,
        device=device,
        ee_body_idx=ee_body_idx,
        state_dim=state_dim,
    )

    return (
        {
            "full_image": main_rgb,
            "wrist_image": wrist_rgb,
            "state": states_tensor,
        },
        ee_body_idx,
    )


def apply_action_guard(
    action: torch.Tensor,
    action_clip: float,
    ee_pos_delta_scale: float,
    ee_rot_delta_scale: float,
):
    if action_clip > 0:
        action = torch.clamp(action, -action_clip, action_clip)

    ee_pos_delta_raw = action[:, :3]
    ee_rot_delta_raw = action[:, 3:6]
    gripper_open = action[:, -1]

    ee_pos_delta_raw = ee_pos_delta_raw * ee_pos_delta_scale
    ee_rot_delta_raw = ee_rot_delta_raw * ee_rot_delta_scale

    return ee_pos_delta_raw, ee_rot_delta_raw, gripper_open


def convert_roboverse_action(
    action: torch.Tensor,
    last_obs: Any,
    robot_name: str,
    ee_body_name: str,
    ee_body_idx: Optional[int],
    ik_solver: Any,
    robot_cfg: Any,
    action_coordinate_frame: str,
    action_clip: float,
    ee_pos_delta_scale: float,
    ee_rot_delta_scale: float,
    device,
    inverse_reorder_idx=None,
):
    num_envs = action.shape[0]

    rs = last_obs.robots[robot_name]
    if inverse_reorder_idx is None:
        msg = "inverse_reorder_idx must be provided by RoboVerseEnv"
        logger = get_logger()
        logger.error(msg)
        raise ValueError(msg)
    joint_pos_raw = (
        rs.joint_pos
        if isinstance(rs.joint_pos, torch.Tensor)
        else torch.tensor(rs.joint_pos)
    )
    curr_robot_q = joint_pos_raw[:, inverse_reorder_idx].to(device).float()
    robot_ee_state = (
        (
            rs.body_state
            if isinstance(rs.body_state, torch.Tensor)
            else torch.tensor(rs.body_state)
        )
        .to(device)
        .float()
    )
    robot_root_state = (
        (
            rs.root_state
            if isinstance(rs.root_state, torch.Tensor)
            else torch.tensor(rs.root_state)
        )
        .to(device)
        .float()
    )

    if ee_body_idx is None:
        ee_body_idx = rs.body_names.index(ee_body_name)
    ee_p_world = robot_ee_state[:, ee_body_idx, 0:3]
    ee_q_world = robot_ee_state[:, ee_body_idx, 3:7]

    robot_pos = robot_root_state[:, 0:3]
    robot_quat = robot_root_state[:, 3:7]
    inv_base_q = transforms.quaternion_invert(robot_quat).to(dtype=torch.float32)
    curr_ee_pos_local = transforms.quaternion_apply(
        inv_base_q, ee_p_world - robot_pos
    ).to(dtype=torch.float32)
    curr_ee_quat_local = transforms.quaternion_multiply(inv_base_q, ee_q_world).to(
        dtype=torch.float32
    )

    action = action.to(device).float()
    ee_pos_delta_raw, ee_rot_delta_raw, gripper_open = apply_action_guard(
        action[:num_envs],
        action_clip=action_clip,
        ee_pos_delta_scale=ee_pos_delta_scale,
        ee_rot_delta_scale=ee_rot_delta_scale,
    )

    if action_coordinate_frame == "world":
        ee_pos_delta = transforms.quaternion_apply(inv_base_q, ee_pos_delta_raw).to(
            dtype=torch.float32
        )
        ee_rot_matrix_world = transforms.euler_angles_to_matrix(ee_rot_delta_raw, "XYZ")
        ee_rot_quat_world = transforms.matrix_to_quaternion(ee_rot_matrix_world).to(
            dtype=torch.float32
        )
        ee_quat_delta = transforms.quaternion_multiply(
            inv_base_q, ee_rot_quat_world
        ).to(dtype=torch.float32)
    else:
        ee_pos_delta = ee_pos_delta_raw
        ee_quat_delta = transforms.matrix_to_quaternion(
            transforms.euler_angles_to_matrix(ee_rot_delta_raw, "XYZ")
        ).to(dtype=torch.float32)

    ee_pos_target = (curr_ee_pos_local + ee_pos_delta).to(
        device=device, dtype=torch.float32
    )
    ee_quat_target = transforms.quaternion_multiply(
        curr_ee_quat_local, ee_quat_delta
    ).to(device=device, dtype=torch.float32)
    curr_robot_q = curr_robot_q.to(device=device, dtype=torch.float32)

    q_solution, _ik_succ = ik_solver.solve_ik_batch(
        ee_pos_target, ee_quat_target, curr_robot_q
    )
    gripper_widths = process_gripper_command(gripper_open, robot_cfg, device)
    return ik_solver.compose_joint_action(
        q_solution, gripper_widths, current_q=curr_robot_q, return_dict=True
    )


def serialize_actions_dict(actions_dict, valid_joint_names=None):
    if not isinstance(actions_dict, dict):
        return actions_dict

    valid_joint_names = (
        set(valid_joint_names) if valid_joint_names is not None else None
    )

    def to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value

    def serialize_joint_values(joint_dict):
        # Batch tensor conversion to reduce many tiny tensor->numpy transfers.
        filtered_items = []
        for joint_name, joint_value in joint_dict.items():
            if valid_joint_names is not None and joint_name not in valid_joint_names:
                continue
            filtered_items.append((joint_name, joint_value))

        result = {}
        tensor_items = [
            item for item in filtered_items if isinstance(item[1], torch.Tensor)
        ]
        non_tensor_items = [
            item for item in filtered_items if not isinstance(item[1], torch.Tensor)
        ]

        if tensor_items:
            names = [name for name, _ in tensor_items]
            tensors = [tensor.detach() for _, tensor in tensor_items]
            try:
                stacked = torch.stack(tensors, dim=0).cpu().numpy()
                for idx, name in enumerate(names):
                    result[name] = stacked[idx]
            except Exception:
                for name, tensor in tensor_items:
                    result[name] = tensor.cpu().numpy()

        for name, value in non_tensor_items:
            result[name] = value

        return result

    serialized = {}
    for env_id, env_actions in actions_dict.items():
        serialized[env_id] = {}
        for obj_name, obj_actions in env_actions.items():
            serialized[env_id][obj_name] = {}
            for key, value in obj_actions.items():
                if isinstance(value, dict):
                    serialized[env_id][obj_name][key] = serialize_joint_values(value)
                else:
                    serialized[env_id][obj_name][key] = to_numpy(value)
    return serialized
