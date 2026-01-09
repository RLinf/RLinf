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
) -> np.ndarray:
    chunk_actions = raw_chunk_actions
    chunk_actions[..., -1] = np.sign(chunk_actions[..., -1])
    return chunk_actions


def prepare_actions_for_robocasa(
    raw_chunk_actions,
    action_dim,
    model_type,
) -> np.ndarray:
    """
    Prepare actions for robocasa environment.

    For Pi0 models:
        - Pi0 outputs 32D, but only [5:12] contains valid data (see norm_stats.json)
        - Extract the valid 7D: [3D arm_pos, 3D arm_ori, 1D gripper]
        - Convert to 12D PandaOmron format: [3D arm_pos, 3D arm_ori, 1D gripper, 4D base, 1D base_mode]

    For other models: Directly extract action_dim dimensions
    """
    if SupportedModel(model_type) == SupportedModel.OPENPI:
        # Pi0: Extract valid 7D from [5:12] and convert to 12D for PandaOmron
        # Note: raw_chunk_actions is already sliced to [:12] by RobocasaOutputs
        actions_7d = raw_chunk_actions[
            ..., 5:12
        ]  # Extract valid 7 dimensions from [5:12]
        output_shape = actions_7d.shape[:-1] + (12,)  # Shape: (..., 12)
        actions_12d = np.zeros(output_shape, dtype=np.float32)

        # PandaOmron action mapping:
        # Pi0's 7D [arm_pos(3), arm_ori(3), gripper(1)] → PandaOmron's 12D
        actions_12d[..., 0:7] = actions_7d  # Map first 7 dimensions directly
        actions_12d[..., -1] = 0  # Always control Panda instead of base

        return actions_12d
    else:
        # Other models: directly extract first action_dim dimensions
        chunk_actions = raw_chunk_actions[..., :action_dim]
        chunk_actions[..., -1] = 0  # Always control Panda instead of base

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
    env_type,
    model_type,
    num_action_chunks,
    action_dim,
    action_scale: float = 1.0,
    policy: str = "widowx_bridge",
) -> torch.Tensor | np.ndarray:
    if env_type == "libero":
        chunk_actions = prepare_actions_for_libero(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == "maniskill":
        chunk_actions = prepare_actions_for_maniskill(
            raw_chunk_actions=raw_chunk_actions,
            num_action_chunks=num_action_chunks,
            action_dim=action_dim,
            action_scale=action_scale,
            policy=policy,
        )
    elif env_type == "robotwin":
        chunk_actions = raw_chunk_actions
    elif env_type == "metaworld":
        chunk_actions = raw_chunk_actions
    elif env_type == "calvin":
        chunk_actions = prepare_actions_for_calvin(
            raw_chunk_actions=raw_chunk_actions,
        )
    elif env_type == "behavior":
        chunk_actions = raw_chunk_actions
    elif env_type == "isaaclab":
        chunk_actions = prepare_actions_for_isaaclab(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == "robocasa":
        chunk_actions = prepare_actions_for_robocasa(
            raw_chunk_actions=raw_chunk_actions,
            action_dim=action_dim,
            model_type=model_type,
        )
    elif env_type == "realworld":
        chunk_actions = raw_chunk_actions
    elif env_type == "mujoco":
        chunk_actions = prepare_actions_for_mujoco(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    else:
        raise NotImplementedError

    return chunk_actions
