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
from rlinf.models.embodiment.prismatic.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NormalizationType,
)


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


def unnormalize_actions(normalized_actions, action_norm_stats):
    """Unnormalize actions using dataset statistics"""

    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["min"], dtype=bool)
        )
        # Ensure mask is a numpy boolean array (may come as list from json)
        mask = np.array(mask, dtype=bool)
        action_high, action_low = (
            np.array(action_norm_stats["max"]),
            np.array(action_norm_stats["min"]),
        )
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = (
            np.array(action_norm_stats["q99"]),
            np.array(action_norm_stats["q01"]),
        )
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    if isinstance(normalized_actions, torch.Tensor):
        normalized_actions = normalized_actions.cpu().numpy()

    action_dim = normalized_actions.shape[-1]

    action_high = action_high[:action_dim]
    action_low = action_low[:action_dim]
    mask = mask[:action_dim]

    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
        normalized_actions,
    )

    return actions


def prepare_actions(
    raw_chunk_actions,
    env_type,
    model_type,
    num_action_chunks,
    action_dim,
    action_scale: float = 1.0,
    policy: str = "widowx_bridge",
    action_norm_stats=None,
    use_openpi_unnormalize: bool = None,
    openpi_use_quantiles: bool = False,
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
    else:
        raise NotImplementedError

    # Auto-detect OpenPI model if use_openpi_unnormalize is not explicitly set
    if use_openpi_unnormalize is None:
        try:
            use_openpi_unnormalize = SupportedModel(model_type) == SupportedModel.OPENPI
        except (ValueError, TypeError):
            use_openpi_unnormalize = False

    # Use OpenPI-specific unnormalization if specified
    if action_norm_stats is not None:
        if use_openpi_unnormalize:
            # Use OpenPI normalization function (defined in this file)
            chunk_actions = unnormalize_openpi_actions(
                chunk_actions,
                action_norm_stats,
            )
        else:
            chunk_actions = unnormalize_actions(chunk_actions, action_norm_stats)

    return chunk_actions


def normalize_openpi_state(state, state_norm_stats, use_quantiles=True):
    """
    Normalize state for OpenPI model using dataset statistics.

    This function normalizes state observations to the range [-1, 1] for OpenPI model input.
    It supports both bounds-based and quantile-based normalization.

    Args:
        state: Raw state observations from environment (can be torch.Tensor or np.ndarray)
        state_norm_stats: Dictionary containing normalization statistics. Should contain:
            - For bounds: "min" and "max" keys
            - For quantiles: "q01" and "q99" keys (or "q10" and "q90")
            - Optional: "mask" key for masking certain state dimensions
        use_quantiles: If True, use quantile-based normalization (q01/q99),
                      otherwise use bounds-based normalization (min/max)

    Returns:
        Normalized state in the range [-1, 1]
    """
    if state_norm_stats is None:
        return state

    if isinstance(state, torch.Tensor):
        state = state.cpu().numpy()

    state_high = np.array(state_norm_stats["q99"])
    state_low = np.array(state_norm_stats["q01"])

    # Check if stats are empty before proceeding
    if state_high.shape[0] == 0 or state_low.shape[0] == 0:
        raise ValueError("State stats are empty, skipping normalization")

    mask = state_norm_stats.get(
        "mask", np.ones_like(state_norm_stats["q01"], dtype=bool)
    )

    state_dim = state.shape[-1]

    state_high = state_high[:state_dim]
    state_low = state_low[:state_dim]
    mask = mask[:state_dim]

    # Normalize: convert from [state_low, state_high] to [-1, 1]
    # Formula: normalized_state = 2 * (state - low) / (high - low) - 1
    normalized_state = np.where(
        mask,
        2.0 * (state - state_low) / (state_high - state_low + 1e-8) - 1.0,
        state,
    )

    return normalized_state


def unnormalize_openpi_actions(
    normalized_actions, action_norm_stats, use_quantiles=True
):
    """
    Unnormalize actions for OpenPI model using dataset statistics.

    This function is designed specifically for OpenPI models.
    It supports both bounds-based and quantile-based normalization.

    Args:
        normalized_actions: Normalized actions from OpenPI model (expected to be in [-1, 1] range)
        action_norm_stats: Dictionary containing normalization statistics. Should contain:
            - For bounds: "min" and "max" keys
            - For quantiles: "q01" and "q99" keys (or "q10" and "q90")
            - Optional: "mask" key for masking certain action dimensions
        use_quantiles: If True, use quantile-based normalization (q01/q99),
                      otherwise use bounds-based normalization (min/max)

    Returns:
        Unnormalized actions in the original action space
    """
    if action_norm_stats is None:
        return normalized_actions

    if isinstance(normalized_actions, torch.Tensor):
        normalized_actions = normalized_actions.cpu().numpy()

    mask = action_norm_stats.get(
        "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
    )
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])

    # If stats are empty, skip unnormalization
    if action_high.shape[0] == 0 or action_low.shape[0] == 0:
        raise ValueError("Action stats are empty, skipping unnormalization")

    # Ensure mask is a numpy boolean array (may come as list from json)
    mask = np.array(mask, dtype=bool)

    action_dim = normalized_actions.shape[-1]
    repeat_factor = action_dim // action_high.shape[0]
    # If dimensions are incompatible, skip unnormalization
    if repeat_factor == 0:
        return normalized_actions
    action_high = action_high.repeat(repeat_factor)
    action_low = action_low.repeat(repeat_factor)
    mask = mask.repeat(repeat_factor) if mask.ndim > 0 else mask

    # Unnormalize: convert from [-1, 1] to [action_low, action_high]
    # Formula: action = 0.5 * (normalized_action + 1) * (high - low) + low
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
        normalized_actions,
    )

    return actions
