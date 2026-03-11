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

import numpy as np
import torch

from rlinf.config import SupportedModel
from rlinf.envs import SupportedEnvType


def prepare_actions_for_maniskill(
    raw_chunk_actions,
    num_action_chunks,
    action_dim,
    action_scale,
    policy,
) -> torch.Tensor:
    if "panda" in policy:
        return raw_chunk_actions
    # TODO only suitable for action_dim = 7
    reshaped_actions = raw_chunk_actions.reshape(-1, action_dim)
    batch_size = reshaped_actions.shape[0]
    raw_actions = {
        "world_vector": np.array(reshaped_actions[:, :3]),
        "rotation_delta": np.array(reshaped_actions[:, 3:6]),
        "open_gripper": np.array(
            reshaped_actions[:, 6:7]
        ),  # range [0, 1]; 1 = open; 0 = close
    }

    # process raw_action to obtain the action to be sent to the maniskill2 environment
    actions = {}
    actions["world_vector"] = raw_actions["world_vector"] * action_scale  # [B, 3]
    actions["rot_axangle"] = raw_actions["rotation_delta"] * action_scale  # [B, 3]

    if policy == "google_robot":
        raise NotImplementedError
    elif policy == "widowx_bridge":
        actions["gripper"] = 2.0 * (raw_actions["open_gripper"] > 0.5) - 1.0  # [B, 1]
    elif policy == "panda_wristcam":
        actions["gripper"] = 2.0 * (raw_actions["open_gripper"] > 0.5) - 1.0  # [B, 1]

    actions["terminate_episode"] = np.array([0.0] * batch_size).reshape(-1, 1)  # [B, 1]

    actions = {k: torch.tensor(v, dtype=torch.float32) for k, v in actions.items()}
    actions = torch.cat(
        [actions["world_vector"], actions["rot_axangle"], actions["gripper"]], dim=1
    ).cuda()

    chunk_actions = actions.reshape(-1, num_action_chunks, action_dim)

    return chunk_actions


def prepare_actions_for_libero(
    raw_chunk_actions,
    model_type,
) -> np.ndarray:
    chunk_actions = raw_chunk_actions
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
    ]:
        chunk_actions[..., -1] = 2 * chunk_actions[..., -1] - 1
        chunk_actions[..., -1] = np.sign(chunk_actions[..., -1]) * -1.0
    return chunk_actions


def prepare_actions_for_isaaclab(
    raw_chunk_actions,
    model_type,
) -> torch.Tensor:
    """
    Here reture a general 7 dof action. If the action is modified, please change the output of the model
    For example, in `RLinf/rlinf/models/embodiment/gr00t/simulation_io.py`
    """
    chunk_actions = torch.from_numpy(raw_chunk_actions)
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
    ]:
        chunk_actions[..., -1] = 2 * chunk_actions[..., -1] - 1
        chunk_actions[..., -1] = torch.sign(chunk_actions[..., -1]) * -1.0
    return chunk_actions


def prepare_actions_for_calvin(
    raw_chunk_actions,
    model_type,
) -> np.ndarray:
    chunk_actions = raw_chunk_actions
    if SupportedModel(model_type) == SupportedModel.OPENPI:
        chunk_actions[..., -1] = np.sign(chunk_actions[..., -1])
    else:
        chunk_actions[..., -1] = np.where(chunk_actions[..., -1] > 0, 1, -1)
    return chunk_actions


def prepare_actions_for_metaworld(
    raw_chunk_actions,
    model_type,
) -> np.ndarray:
    chunk_actions = raw_chunk_actions
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
    ]:
        # the action dimesion of metaworld is 4-dim (x, y, z, gripper)
        # we need to extract the first 3-dim and the last dim in a 7-dim action
        if chunk_actions.shape[-1] == 7:
            chunk_actions = np.concatenate(
                [chunk_actions[..., :3], chunk_actions[..., -1:]], axis=-1
            )
    return chunk_actions


def prepare_actions_for_robocasa(
    raw_chunk_actions,
    action_dim,
    model_type,
    env_cfg=None,
) -> np.ndarray:
    """
    Prepare actions for RoboCasa-style mobile-manipulation environments.

    The default mapping preserves the existing RLinf behaviour, while env-specific
    overrides can be supplied via ``env.action_space``.
    """
    action_space_cfg = {}
    if env_cfg is not None:
        action_space_cfg = getattr(env_cfg, "action_space", {})
        if hasattr(action_space_cfg, "items"):
            action_space_cfg = dict(action_space_cfg.items())

    env_action_dim = action_space_cfg.get("env_action_dim", action_dim)
    openpi_valid_action_slice = action_space_cfg.get(
        "openpi_valid_action_slice", [5, 12]
    )
    disable_base_control = action_space_cfg.get("disable_base_control", True)
    base_mode_index = action_space_cfg.get("base_mode_index", env_action_dim - 1)

    if SupportedModel(model_type) == SupportedModel.OPENPI:
        start_idx, end_idx = openpi_valid_action_slice
        actions_7d = raw_chunk_actions[..., start_idx:end_idx]
        output_shape = actions_7d.shape[:-1] + (env_action_dim,)
        actions_env = np.zeros(output_shape, dtype=np.float32)

        copy_dim = min(actions_7d.shape[-1], env_action_dim)
        actions_env[..., :copy_dim] = actions_7d[..., :copy_dim]
        if disable_base_control and 0 <= base_mode_index < env_action_dim:
            actions_env[..., base_mode_index] = 0
        return actions_env

    chunk_actions = raw_chunk_actions[..., :env_action_dim]
    if disable_base_control and 0 <= base_mode_index < env_action_dim:
        chunk_actions[..., base_mode_index] = 0
    return chunk_actions


def prepare_actions_for_mujoco(raw_chunk_actions, model_type):
    if raw_chunk_actions.shape[-1] >= 7:
        chunk_actions = np.concatenate(
            [raw_chunk_actions[..., :3], raw_chunk_actions[..., 6:7]], axis=-1
        )
    else:
        chunk_actions = raw_chunk_actions[..., :4]
    if SupportedModel(model_type) == SupportedModel.OPENPI:
        chunk_actions[..., -1] = np.clip(chunk_actions[..., -1], -1.0, 1.0)
    return chunk_actions


def prepare_actions(
    raw_chunk_actions,
    env_type: str,
    model_type: str,
    num_action_chunks,
    action_dim,
    action_scale: float = 1.0,
    policy: str = "widowx_bridge",
    wm_env_type=None,
    env_cfg=None,
) -> torch.Tensor | np.ndarray:
    env_type = SupportedEnvType(env_type)
    if env_type == SupportedEnvType.LIBERO:
        chunk_actions = prepare_actions_for_libero(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.OPENSORAWM or env_type == SupportedEnvType.WANWM:
        # TODO: Implement prepare_actions_for_opensora_wm
        if wm_env_type == "libero":
            chunk_actions = prepare_actions_for_libero(
                raw_chunk_actions=raw_chunk_actions,
                model_type=model_type,
            )
        else:
            raise NotImplementedError(f"Env type {wm_env_type} not implemented")
    elif env_type == SupportedEnvType.MANISKILL:
        chunk_actions = prepare_actions_for_maniskill(
            raw_chunk_actions=raw_chunk_actions,
            num_action_chunks=num_action_chunks,
            action_dim=action_dim,
            action_scale=action_scale,
            policy=policy,
        )
    elif env_type == SupportedEnvType.ROBOTWIN:
        chunk_actions = raw_chunk_actions
    elif env_type == SupportedEnvType.METAWORLD:
        chunk_actions = prepare_actions_for_metaworld(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.CALVIN:
        chunk_actions = prepare_actions_for_calvin(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.BEHAVIOR:
        chunk_actions = raw_chunk_actions
    elif env_type == SupportedEnvType.ISAACLAB:
        chunk_actions = prepare_actions_for_isaaclab(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type in (SupportedEnvType.ROBOCASA, SupportedEnvType.ROBOCASA365):
        chunk_actions = prepare_actions_for_robocasa(
            raw_chunk_actions=raw_chunk_actions,
            action_dim=action_dim,
            model_type=model_type,
            env_cfg=env_cfg,
        )
    elif env_type == SupportedEnvType.REALWORLD:
        chunk_actions = raw_chunk_actions
    elif env_type == SupportedEnvType.FRANKASIM:
        chunk_actions = prepare_actions_for_mujoco(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    else:
        raise NotImplementedError

    return chunk_actions
