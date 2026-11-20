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

import json
import os
from importlib.util import find_spec
from pathlib import Path

import torch
import yaml
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.utils.data_iter_utils import ceildiv
from rlinf.utils.logging import get_logger

SUPPORTED_ENV_WRAPPERS = ("rgb", "default", "rgb_lowres", "rgbd_full_res")

R1PRO_PROPRIO_KEYS = [
    "joint_qpos",
    "joint_qpos_sin",
    "joint_qpos_cos",
    "joint_qvel",
    "joint_qeffort",
    "robot_pos",
    "robot_ori_cos",
    "robot_ori_sin",
    "robot_2d_ori",
    "robot_2d_ori_cos",
    "robot_2d_ori_sin",
    "robot_lin_vel",
    "robot_ang_vel",
    "arm_left_qpos",
    "arm_left_qpos_sin",
    "arm_left_qpos_cos",
    "arm_left_qvel",
    "eef_left_pos",
    "eef_left_quat",
    "gripper_left_qpos",
    "gripper_left_qvel",
    "arm_right_qpos",
    "arm_right_qpos_sin",
    "arm_right_qpos_cos",
    "arm_right_qvel",
    "eef_right_pos",
    "eef_right_quat",
    "gripper_right_qpos",
    "gripper_right_qvel",
    "trunk_qpos",
    "trunk_qvel",
    "base_qpos",
    "base_qpos_sin",
    "base_qpos_cos",
    "base_qvel",
]


def sync_robot_after_pose_override(robot) -> None:
    """Synchronize robot state after a direct pose override.

    Offline BEHAVIOR instance loading often teleports the robot base via
    ``set_position_orientation`` without restoring the robot articulation /
    controller state. Reset controller goals to the robot's current joint state so
    the next control step starts from a consistent no-op target instead of stale
    goals carried over from a previous episode / instance.

    Args:
        robot: OmniGibson robot instance whose pose was overridden.
    """
    robot.keep_still()

    if getattr(robot, "n_joints", 0) > 0:
        current_joint_positions = robot.get_joint_positions()
        robot.set_joint_positions(positions=current_joint_positions, drive=False)
        robot.set_joint_velocities(
            velocities=torch.zeros_like(current_joint_positions),
            drive=False,
        )

    robot.keep_still()


def set_camera_resolution(
    camera_cfg: DictConfig | None,
    robot_cfg: DictConfig,
    r1pro: DictConfig | None = None,
    wrapper_name: str | None = None,
) -> None:
    """Configure the robot cameras before the simulator creates their sensors."""
    use_yaml = wrapper_name in (None, "rgb")
    if use_yaml and camera_cfg is None:
        return
    if r1pro is None:
        import omnigibson as og

        r1pro = OmegaConf.load(os.path.join(og.root_path, "eval", "r1pro.yaml"))
    cameras = r1pro.eval.camera_sensor_names
    if use_yaml:
        head = camera_cfg.get("head_resolution")
        wrist = camera_cfg.get("wrist_resolution")
        resolutions = {role: head if role == "head" else wrist for role in cameras}
    else:
        spec = get_env_wrapper(wrapper_name).camera_spec()
        resolutions = spec["resolution"]
        robot_cfg.obs_modalities = ["proprio", *spec["modalities"]]

    sensors = robot_cfg.get("sensor_config") or {}
    for role, name in cameras.items():
        resolution = resolutions.get(role)
        if resolution is None:
            continue
        height, width = resolution
        link = name.split(":", 1)[1]
        # A per-link entry takes precedence over the shared VisionSensor settings.
        current = sensors.get(link, sensors.get("VisionSensor", {}))
        size = {"image_height": height, "image_width": width}
        if not use_yaml and role == "head":
            size["horizontal_aperture"] = 40.0
        sensor = OmegaConf.merge(current, {"sensor_kwargs": size})
        OmegaConf.update(robot_cfg, f"sensor_config.{link}", sensor)


def apply_runtime_renderer_settings() -> None:
    """
    RLinf-specific renderer overrides after OmniGibson has launched.
    """

    import omnigibson.lazy as lazy

    lazy.carb.settings.get_settings().set_float(
        "/rtx-transient/resourcemanager/texturestreaming/memoryBudget",
        0.1,
    )


def get_env_wrapper(wrapper_name: str):
    if wrapper_name == "rgb":
        from .rgb_wrapper import RGBWrapper

        return RGBWrapper
    if wrapper_name in ("default", "rgb_lowres"):
        from omnigibson.eval.wrappers import DefaultWrapper

        return DefaultWrapper
    if wrapper_name == "rgbd_full_res":
        from omnigibson.eval.wrappers import RGBDFullResWrapper

        return RGBDFullResWrapper
    if wrapper_name == "rich_obs":
        raise ValueError(
            "OmniGibson 3.9.3 removed rich_obs. Use rgbd_full_res for RGB and depth. "
            "It does not provide normals, optical flow or privileged task observations."
        )
    raise ValueError(
        f"Unsupported wrapper name: {wrapper_name}, expected one of {SUPPORTED_ENV_WRAPPERS}"
    )


def convert_uint8_rgb(image: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)

    if image.dtype == torch.uint8:
        return image[..., :3]

    if torch.is_floating_point(image):
        max_val = float(image.detach().max().item()) if image.numel() > 0 else 1.0
        if max_val <= 1.0 + 1e-6:
            image = image * 255.0
        image = image.round().clamp(0, 255).to(torch.uint8)
    else:
        image = image.clamp(0, 255).to(torch.uint8)

    return image[..., :3]


def apply_env_wrapper(vec_env, wrapper_name: str | None):
    if wrapper_name is None:
        return vec_env
    wrapper_cls = get_env_wrapper(wrapper_name)
    return wrapper_cls(vec_env)


def override_sub_cfg(omni_cfg: DictConfig, override_cfg: DictConfig, sub_attr: str):
    omni_sub_cfg = OmegaConf.select(omni_cfg, sub_attr)
    override_sub_cfg = OmegaConf.select(override_cfg, sub_attr)
    if override_sub_cfg is not None:
        setattr(
            omni_cfg,
            sub_attr,
            override_sub_cfg
            if omni_sub_cfg is None
            else OmegaConf.merge(omni_sub_cfg, override_sub_cfg),
        )


def resolve_eval_step_limits(cfg: DictConfig, chunk_size: int) -> None:
    if cfg.max_episode_steps is None:
        # Read the official task statistics without importing the simulator on the driver.
        root = Path(
            os.environ.get("OMNIGIBSON_DATA_PATH", "../../datasets")
        ).expanduser()
        if not root.is_absolute():
            package = find_spec("omnigibson")
            if package is None:
                raise ModuleNotFoundError(
                    "Set OMNIGIBSON_DATA_PATH or install OmniGibson."
                )
            root = Path(package.origin).resolve().parent / root
        stats = root / "2026-challenge-task-instances" / "metadata" / "task.jsonl"
        task = cfg.omni_config.task.activity_name
        with stats.open() as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["task_name"] == task:
                    # The SDK tests its zero-based step counter before incrementing it.
                    cfg.max_episode_steps = int(row["length"] * 1.5) + 1
                    break
            else:
                raise ValueError(
                    f"No human demonstration length for {task!r} in {stats}"
                )
    if cfg.max_episode_steps < 1 or chunk_size < 1:
        raise ValueError(
            "BEHAVIOR episode length and action chunk size must be positive."
        )
    chunks = ceildiv(cfg.max_episode_steps, chunk_size)
    episode_budget = chunks * chunk_size
    if cfg.max_steps_per_rollout_epoch is None:
        cfg.max_steps_per_rollout_epoch = episode_budget
    elif cfg.max_steps_per_rollout_epoch <= 0:
        raise ValueError("BEHAVIOR max_steps_per_rollout_epoch must be positive.")
    elif (
        not cfg.auto_reset
        and not cfg.get("early_stop_on_done", False)
        and cfg.max_steps_per_rollout_epoch > episode_budget
    ):
        raise ValueError(
            f"With auto_reset=False, the episode fits in {chunks} action chunks "
            f"({episode_budget} rollout steps). Reduce max_steps_per_rollout_epoch "
            "or enable early_stop_on_done."
        )


def setup_omni_cfg(cfg: DictConfig) -> DictConfig:
    """
    Setup OmniGibson's config, overrided by user-set config

    Args:
        cfg(DictConfig): rlinf's env config, must have `omni_config` field

    Returns:
        (DictConfig): overrided OmniGibson config
    """
    base_config_name = OmegaConf.select(
        cfg, "base_config_name", default="r1pro_behavior"
    )

    import omnigibson as og
    from omnigibson.macros import gm

    override_cfg = OmegaConf.select(cfg, "omni_config")
    cfg_path = os.path.join(og.example_config_path, f"{base_config_name}.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        omni_cfg = OmegaConf.create(yaml.load(f, Loader=yaml.FullLoader))
    # override env/render/camera/robots/task/scene config
    override_sub_cfg(omni_cfg, override_cfg, "env")
    override_sub_cfg(omni_cfg, override_cfg, "render")
    override_sub_cfg(omni_cfg, override_cfg, "camera")
    override_sub_cfg(omni_cfg, override_cfg, "macro")
    override_sub_cfg(omni_cfg, override_cfg, "task")
    override_sub_cfg(omni_cfg, override_cfg, "scene")
    # here actually we only needs one robot config (and Behavior actually does do that)
    # we must use update rather than merge to keep default robot config fields.
    robot_override = OmegaConf.select(override_cfg, "robots[0]", default=None)
    assert robot_override is not None, (
        "OmniGibson config must contain a non-empty robots list, but robots[0] config is None"
    )
    OmegaConf.update(omni_cfg, "robots[0]", robot_override, merge=True)

    override_proprio_obs = robot_override.get("proprio_obs")
    r1pro = None
    if override_proprio_obs is None:
        if cfg.get("use_2026_eval_proprio_and_rooms", False):
            # Match the official 61-value state in 2026 evaluation and demonstrations.
            r1pro = OmegaConf.load(os.path.join(og.root_path, "eval", "r1pro.yaml"))
            override_proprio_obs = r1pro.proprio_obs
        else:
            override_proprio_obs = R1PRO_PROPRIO_KEYS
    OmegaConf.update(
        omni_cfg, "robots[0].proprio_obs", override_proprio_obs, merge=True
    )

    # Select task-relevant rooms when partial scene loading is enabled.
    partial_scene_load = OmegaConf.select(omni_cfg, "scene.partial_scene_load")
    if partial_scene_load is not None:
        with open_dict(omni_cfg.scene):
            omni_cfg.scene.pop("partial_scene_load", None)
        if partial_scene_load and omni_cfg.scene.get("load_room_instances") is None:
            from gello.utils.og_teleop_utils import (
                augment_rooms,
                get_task_relevant_room_types,
                load_available_tasks,
            )

            activity_name = OmegaConf.select(omni_cfg, "task.activity_name")
            scene_model = OmegaConf.select(omni_cfg, "scene.scene_model")
            if not activity_name or not scene_model:
                raise ValueError(
                    "partial_scene_load requires task.activity_name and scene.scene_model "
                    f"in omni_config; got activity_name={activity_name!r}, "
                    f"scene_model={scene_model!r}."
                )
            relevant_rooms = None
            if cfg.get("use_2026_eval_proprio_and_rooms", False):
                # Challenge room instances are specific to the official task scene.
                tasks = load_available_tasks().get(activity_name, {})
                definition_id = omni_cfg.task.get("activity_definition_id", 0)
                if tasks.get(definition_id, {}).get("scene_model") == scene_model:
                    from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_ROOMS

                    relevant_rooms = TASK_NAMES_TO_ROOMS.get(activity_name)
            if relevant_rooms is None:
                relevant_rooms = get_task_relevant_room_types(
                    activity_name=activity_name
                )
                relevant_rooms = augment_rooms(
                    relevant_rooms, scene_model, activity_name
                )
                relevant_rooms.sort()
                omni_cfg.scene.load_room_types = relevant_rooms
            else:
                omni_cfg.scene.load_room_instances = relevant_rooms
            get_logger().info(
                f"Auto-detected relevant rooms for task {activity_name}: {relevant_rooms}"
            )

    # setup omnigibson macros, according to configuration yaml
    macro_cfg = OmegaConf.select(omni_cfg, "macro")
    gm.HEADLESS = macro_cfg.headless
    gm.ENABLE_OBJECT_STATES = macro_cfg.enable_object_states
    gm.USE_GPU_DYNAMICS = macro_cfg.use_gpu_dynamics
    gm.ENABLE_TRANSITION_RULES = macro_cfg.enable_transition_rules
    gm.RENDER_VIEWER_CAMERA = macro_cfg.render_viewer_camera
    gm.USE_NUMPY_CONTROLLER_BACKEND = macro_cfg.use_numpy_controller_backend

    # setup head/wrist camera resolutions
    camera_cfg = OmegaConf.select(omni_cfg, "camera")
    set_camera_resolution(
        camera_cfg, omni_cfg.robots[0], r1pro, omni_cfg.env.get("env_wrapper")
    )

    # OmniGibson checks its zero-based step counter before incrementing it.
    OmegaConf.update(
        omni_cfg, "task.termination_config.max_steps", cfg.max_episode_steps - 1
    )

    return omni_cfg
