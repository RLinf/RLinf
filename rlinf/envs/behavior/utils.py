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

import inspect
import os

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

SUPPORTED_ENV_WRAPPERS = ("rgb", "default", "rgb_lowres", "rich_obs")

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


def set_camera_resolution(camera_cfg: dict | None) -> None:
    if camera_cfg is None:
        return

    import omnigibson.learning.utils.eval_utils as eval_utils

    head_resolution = camera_cfg.get("head_resolution")
    wrist_resolution = camera_cfg.get("wrist_resolution")
    if head_resolution is not None:
        eval_utils.HEAD_RESOLUTION = tuple(head_resolution)
    if wrist_resolution is not None:
        eval_utils.WRIST_RESOLUTION = tuple(wrist_resolution)


def get_env_wrapper(wrapper_name: str):
    if wrapper_name == "rgb":
        from .rgb_wrapper import RGBWrapper

        return RGBWrapper
    if wrapper_name == "default":
        from omnigibson.learning.wrappers.default_wrapper import DefaultWrapper

        return DefaultWrapper
    if wrapper_name == "rgb_lowres":
        from omnigibson.learning.wrappers.rgb_low_res_wrapper import RGBLowResWrapper

        return RGBLowResWrapper
    if wrapper_name == "rich_obs":
        from omnigibson.learning.wrappers.rich_obs_wrapper import RichObservationWrapper

        return RichObservationWrapper
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


def patch_omnigibson_wrapper_reset_signature() -> None:
    from omnigibson.envs.env_wrapper import EnvironmentWrapper

    reset_fn = EnvironmentWrapper.reset
    if getattr(reset_fn, "__rlinf_patched__", False):
        return

    sig = inspect.signature(reset_fn)
    supports_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if supports_kwargs:
        return

    def _reset_with_kwargs(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    _reset_with_kwargs.__rlinf_patched__ = True
    EnvironmentWrapper.reset = _reset_with_kwargs


def apply_env_wrapper(vec_env, wrapper_name: str | None):
    if wrapper_name is None:
        return vec_env
    patch_omnigibson_wrapper_reset_signature()
    wrapper_cls = get_env_wrapper(wrapper_name)
    for i in range(vec_env.num_envs):
        vec_env.envs[i] = wrapper_cls(vec_env.envs[i])
    return vec_env


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


def setup_omni_cfg(cfg: DictConfig) -> DictConfig:
    """
    Setup OmniGibson's config, overrided by user-set config

    Args:
        cfg(DictConfig): rlinf's env config, must have `omni_config` field

    Returns:
        (DictConfig): overrided OmniGibson config
    """
    import omnigibson as og
    from omnigibson.macros import gm

    override_cfg = OmegaConf.select(cfg, "omni_config")
    cfg_path = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
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

    override_proprio_obs = OmegaConf.select(
        override_cfg, "robots[0].proprio_obs", default=None
    )
    if override_proprio_obs is None:
        override_proprio_obs = R1PRO_PROPRIO_KEYS
    OmegaConf.update(
        omni_cfg, "robots[0].proprio_obs", override_proprio_obs, merge=True
    )

    # setup omnigibson macros, according to configuration yaml
    macro_cfg = OmegaConf.select(omni_cfg, "macro")
    gm.HEADLESS = macro_cfg.headless
    gm.ENABLE_FLATCACHE = macro_cfg.enable_flatcache
    gm.ENABLE_OBJECT_STATES = macro_cfg.enable_object_states
    gm.USE_GPU_DYNAMICS = macro_cfg.use_gpu_dynamics
    gm.ENABLE_TRANSITION_RULES = macro_cfg.enable_transition_rules
    gm.RENDER_VIEWER_CAMERA = macro_cfg.render_viewer_camera
    gm.USE_NUMPY_CONTROLLER_BACKEND = macro_cfg.use_numpy_controller_backend

    # setup head/wrist camera resolutions
    camera_cfg = OmegaConf.select(omni_cfg, "camera")
    set_camera_resolution(camera_cfg)

    # override behavior's termination config `max_steps` field
    max_episode_steps = (
        OmegaConf.select(cfg, "max_episode_steps") - 1
    )  # BEHAVIOR env will off-by-one
    assert max_episode_steps is not None, "must set max_episode_steps in config."
    OmegaConf.update(
        omni_cfg,
        "task.termination_config.max_steps",
        max_episode_steps,
    )

    return omni_cfg
