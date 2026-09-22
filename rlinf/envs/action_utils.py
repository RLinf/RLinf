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


def _is_openpi_family(model_type) -> bool:
    """True for OpenPI policies loaded as ``openpi``."""
    if model_type is None:
        return False
    return SupportedModel(model_type) == SupportedModel.OPENPI


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
    # Left on CPU: the env moves it to its own device, which is not always an
    # accelerator (ManiSkill's CPU sim backend) nor always CUDA.
    actions = torch.cat(
        [actions["world_vector"], actions["rot_axangle"], actions["gripper"]], dim=1
    )

    chunk_actions = actions.reshape(-1, num_action_chunks, action_dim)

    return chunk_actions


OPENWAM_ROBOTWIN_REPRESENTATIONS = ("absolute_eef20",)


def _openwam_eef20_to_robotwin_ee16(chunk_actions) -> np.ndarray:
    """Convert OpenWAM's 20-D dual-arm EEF actions to RoboTwin's 16-D ``ee`` actions.

    OpenWAM RoboTwin checkpoints (``dataloader.action_mode: eef``) predict
    ``[l_xyz(3), l_rot6d(6), l_grip(1), r_xyz(3), r_rot6d(6), r_grip(1)]`` in
    the same absolute end-effector frame as the ``endpose`` fields of the
    RoboTwin dataset. RoboTwin's ``take_action(..., action_type="ee")`` wants
    ``[l_xyz(3), l_quat_xyzw(4), l_grip(1), r_xyz(3), r_quat_xyzw(4), r_grip(1)]``;
    the gripper stays a raw ``[0, 1]`` opening (1 = open), exactly as trained.
    Mirrors ``benchmarks/utils/action_conversion.eef20d_to_ee16d`` in OpenWAM.
    """
    from rlinf.utils.rot6d import rot6d_to_quat_xyzw

    raw = np.asarray(chunk_actions, dtype=np.float32)
    if raw.shape[-1] != 20:
        raise ValueError(
            f"OpenWAM RoboTwin rollout expects 20-D EEF actions, got {raw.shape}"
        )
    if not np.isfinite(raw).all():
        raise ValueError("OpenWAM RoboTwin rollout produced non-finite actions")
    flat = raw.reshape(-1, 20)
    out = np.empty((flat.shape[0], 16), dtype=np.float32)
    for row, action in enumerate(flat):
        out[row, 0:3] = action[0:3]
        out[row, 3:7] = rot6d_to_quat_xyzw(action[3:9].astype(np.float64))
        out[row, 7] = action[9]
        out[row, 8:11] = action[10:13]
        out[row, 11:15] = rot6d_to_quat_xyzw(action[13:19].astype(np.float64))
        out[row, 15] = action[19]
    return out.reshape(*raw.shape[:-1], 16)


def prepare_actions_for_robotwin(
    raw_chunk_actions,
    model_type,
    env_cfg=None,
) -> np.ndarray:
    """RoboTwin consumes 14-D joint actions as-is; OpenWAM needs the EEF bridge."""
    if SupportedModel(model_type) != SupportedModel.OPENWAM:
        return raw_chunk_actions
    representation = (
        None if env_cfg is None else env_cfg.get("openwam_action_representation", None)
    )
    if representation not in OPENWAM_ROBOTWIN_REPRESENTATIONS:
        raise ValueError(
            "OpenWAM RoboTwin requires env.<split>.openwam_action_representation "
            f"to be one of {OPENWAM_ROBOTWIN_REPRESENTATIONS}, got {representation!r}. "
            "It switches RoboTwinEnv to end-effector control and 20-D native proprio."
        )
    return _openwam_eef20_to_robotwin_ee16(raw_chunk_actions)


def prepare_actions_for_libero(
    raw_chunk_actions,
    model_type,
    env_cfg=None,
) -> np.ndarray:
    chunk_actions = raw_chunk_actions
    if SupportedModel(model_type) == SupportedModel.OPENWAM:
        representation = (
            None
            if env_cfg is None
            else env_cfg.get("openwam_action_representation", None)
        )
        if representation not in ("absolute_eef10", "native_delta_eef10"):
            raise ValueError(
                "OpenWAM LIBERO requires env.eval.openwam_action_representation "
                "to be 'absolute_eef10' or 'native_delta_eef10'."
            )
        if representation == "absolute_eef10":
            raw = np.asarray(chunk_actions, dtype=np.float32)
            if raw.shape[-1] != 10:
                raise ValueError(
                    f"OpenWAM absolute EEF rollout expects 10-D actions, got {raw.shape}"
                )
            return raw
        return _openwam_eef10_to_libero7(chunk_actions)
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
        SupportedModel.GR00T_N1D6,
        SupportedModel.GR00T_N1D7,
        SupportedModel.COSMOS3,
    ]:
        chunk_actions[..., -1] = 2 * chunk_actions[..., -1] - 1
        chunk_actions[..., -1] = np.sign(chunk_actions[..., -1]) * -1.0
    return chunk_actions


def _openwam_absolute_eef10_to_libero7(
    raw_chunk_actions: np.ndarray,
    reference_eef10: np.ndarray,
    *,
    pos_scale: float = 0.05,
    rot_scale: float = 0.5,
) -> np.ndarray:
    """Convert absolute EEF10 goals to LIBERO's per-step OSC action."""
    from scipy.spatial.transform import Rotation

    raw = np.asarray(raw_chunk_actions, dtype=np.float32)
    ref = np.asarray(reference_eef10, dtype=np.float32)
    if raw.shape[-1] != 10 or ref.shape[-1] != 10:
        raise ValueError(
            f"OpenWAM absolute EEF conversion expects (..., 10), got {raw.shape} and {ref.shape}"
        )
    if raw.shape[:-1] != ref.shape[:-1]:
        raise ValueError(
            f"OpenWAM action/reference batch mismatch: {raw.shape} vs {ref.shape}"
        )
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(ref)):
        raise ValueError(
            "OpenWAM absolute EEF action/reference contains non-finite values; "
            "refusing to send NaN or Inf into the LIBERO controller."
        )
    if not (pos_scale > 0 and rot_scale > 0):
        raise ValueError(f"OSC scales must be positive, got {pos_scale=} {rot_scale=}")

    def matrix(r6):
        first = r6[..., :3]
        second = r6[..., 3:6]
        first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
        second = second - np.sum(first * second, axis=-1, keepdims=True) * first
        second = second / np.maximum(
            np.linalg.norm(second, axis=-1, keepdims=True), 1e-8
        )
        return np.stack([first, second, np.cross(first, second)], axis=-1)

    relative = matrix(raw[..., 3:9]) @ np.swapaxes(matrix(ref[..., 3:9]), -1, -2)
    rotvec = (
        Rotation.from_matrix(relative.reshape(-1, 3, 3))
        .as_rotvec()
        .reshape(raw.shape[:-1] + (3,))
    )
    output = np.concatenate(
        [
            (raw[..., :3] - ref[..., :3]) / pos_scale,
            rotvec / rot_scale,
            -raw[..., 9:10],
        ],
        axis=-1,
    )
    return np.clip(output, -1.0, 1.0).astype(np.float32)


def _openwam_eef10_to_libero7(raw_chunk_actions: np.ndarray) -> np.ndarray:
    """Convert native OpenWAM EEF10 deltas to LIBERO's 7-D OSC action."""
    from scipy.spatial.transform import Rotation

    raw = np.asarray(raw_chunk_actions, dtype=np.float32)
    if raw.shape[-1] != 10:
        raise ValueError(
            f"OpenWAM LIBERO rollout expects 10-D EEF actions, got {raw.shape}"
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError(
            "OpenWAM EEF action contains non-finite values; refusing to send "
            "NaN or Inf into the LIBERO controller."
        )
    r6d = raw[..., 3:9].reshape(-1, 6).astype(np.float64)
    first = r6d[:, :3]
    second = r6d[:, 3:]
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second -= np.sum(first * second, axis=-1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    matrices = np.stack([first, second, np.cross(first, second)], axis=-1)
    rotvec = Rotation.from_matrix(matrices).as_rotvec().astype(np.float32)
    output = np.concatenate(
        [raw[..., :3], rotvec.reshape(raw.shape[:-1] + (3,)), -raw[..., 9:10]],
        axis=-1,
    )
    return np.clip(output, -1.0, 1.0).astype(np.float32)


def prepare_actions_for_isaaclab(
    raw_chunk_actions,
    model_type,
) -> torch.Tensor:
    """
    Here reture a general 7 dof action. If the action is modified, please change the output of the model
    For example, in `RLinf/rlinf/models/embodiment/gr00t/simulation_io.py`
    """
    chunk_actions = (
        torch.from_numpy(raw_chunk_actions)
        if isinstance(raw_chunk_actions, np.ndarray)
        else raw_chunk_actions
    )
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
    ]:
        chunk_actions[..., -1] = 2 * chunk_actions[..., -1] - 1
        chunk_actions[..., -1] = torch.sign(chunk_actions[..., -1]) * -1.0
    return chunk_actions


def prepare_actions_for_polaris(
    raw_chunk_actions,
    model_type,
) -> torch.Tensor:
    """
    Here reture a general 7 dof action. If the action is modified, please change the output of the model
    For example, in `RLinf/rlinf/models/embodiment/gr00t/simulation_io.py`
    """
    chunk_actions = (
        torch.from_numpy(raw_chunk_actions)
        if isinstance(raw_chunk_actions, np.ndarray)
        else raw_chunk_actions
    )
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
    ]:
        chunk_actions[..., -1] = 2 * chunk_actions[..., -1] - 1
        chunk_actions[..., -1] = torch.sign(chunk_actions[..., -1]) * -1.0
    elif _is_openpi_family(model_type):
        chunk_actions[..., -1] = torch.where(
            chunk_actions[..., -1] > 0.5,
            torch.ones_like(chunk_actions[..., -1]),
            torch.zeros_like(chunk_actions[..., -1]),
        )
    return chunk_actions


def prepare_actions_for_calvin(
    raw_chunk_actions,
    model_type,
) -> np.ndarray:
    chunk_actions = raw_chunk_actions
    if _is_openpi_family(model_type):
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
    model_type=None,
    env_cfg=None,
    action_space=None,
) -> np.ndarray:
    """
    Prepare actions for RoboCasa-style mobile-manipulation environments.

    RoboCasa365 can override the env-side action schema via ``env.action_space``.
    The legacy RoboCasa path uses the named action-space mapping from
    ``rlinf.envs.sim.robocasa.utils``.
    """
    action_space_cfg = {}
    if env_cfg is not None:
        action_space_cfg = getattr(env_cfg, "action_space", {})
        if hasattr(action_space_cfg, "items"):
            action_space_cfg = dict(action_space_cfg.items())

    if action_space_cfg:
        env_action_dim = action_space_cfg.get("env_action_dim", action_dim)
        openpi_valid_action_slice = action_space_cfg.get(
            "openpi_valid_action_slice", [0, env_action_dim]
        )
        disable_base_control = action_space_cfg.get("disable_base_control", False)
        base_mode_index = action_space_cfg.get("base_mode_index", env_action_dim - 1)
        binarize_gripper_control = action_space_cfg.get(
            "binarize_gripper_control", True
        )

        if _is_openpi_family(model_type):
            start_idx, end_idx = openpi_valid_action_slice
            actions_env = (
                raw_chunk_actions[..., start_idx:end_idx].copy().astype(np.float32)
            )

            if actions_env.shape[-1] != env_action_dim:
                raise ValueError(
                    f"RoboCasa365 OpenPI expects {env_action_dim}D action, "
                    f"but got {actions_env.shape[-1]}D from slice "
                    f"{openpi_valid_action_slice}. raw shape={raw_chunk_actions.shape}"
                )

            if binarize_gripper_control and env_action_dim >= 12:
                actions_env[..., 6] = np.where(actions_env[..., 6] < 0.5, -1.0, 1.0)
                actions_env[..., 11] = np.where(actions_env[..., 11] < 0.5, -1.0, 1.0)

            if disable_base_control and env_action_dim >= 12:
                actions_env[..., 7:10] = 0.0
                actions_env[..., 10] = 0.0
                if 0 <= base_mode_index < env_action_dim:
                    actions_env[..., base_mode_index] = -1.0

            return actions_env

        chunk_actions = raw_chunk_actions[..., :env_action_dim].copy()
        if disable_base_control and env_action_dim >= 12:
            chunk_actions[..., 7:10] = 0.0
            chunk_actions[..., 10] = 0.0
            if 0 <= base_mode_index < env_action_dim:
                chunk_actions[..., base_mode_index] = -1.0
        return chunk_actions

    # raw_chunk_actions shape: [num_chunks, 32]
    # Extract first action_dim (<=12) dimensions as valid action chunks
    # Then pad them to default actions to get (..., 12)-shaped action chunks for RobocasaEnv.step()
    from rlinf.envs.sim.robocasa.utils import (
        ROBOCASA_ALL_ACTION_DIM,
        ROBOCASA_DEFAULT_ACTION,
        get_action_ids,
        get_action_space,
    )

    assert action_dim <= ROBOCASA_ALL_ACTION_DIM, (
        f"Requested action_dim ({action_dim}) exceeds max dimension ({ROBOCASA_ALL_ACTION_DIM})."
    )

    valid_chunk_actions = raw_chunk_actions[..., :action_dim]

    chunk_actions = np.full(
        shape=valid_chunk_actions.shape[:-1] + (ROBOCASA_ALL_ACTION_DIM,),
        fill_value=ROBOCASA_DEFAULT_ACTION,
        dtype=valid_chunk_actions.dtype,
    )

    all_action_ids = get_action_ids(get_action_space(action_space))
    assert len(all_action_ids) == action_dim, (
        f"Mismatch between action_space ids length ({len(all_action_ids)}) and provided action_dim ({action_dim})."
    )
    chunk_actions[..., all_action_ids] = valid_chunk_actions

    return chunk_actions


def prepare_actions_for_genesis(
    raw_chunk_actions,
    model_type,
) -> torch.Tensor:
    """Prepare actions for the Genesis environment.

    For VLA models (OpenVLA / OpenVLA-OFT), transforms the gripper
    dimension from a [0, 1] continuous value to a {-1, +1} binary signal
    (matching the convention used by other embodied envs).

    For all other models the actions are returned as-is, converted to a
    torch tensor on CUDA.
    """
    if isinstance(raw_chunk_actions, np.ndarray):
        chunk_actions = torch.from_numpy(raw_chunk_actions).float()
    else:
        chunk_actions = raw_chunk_actions.clone().float()
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
    ]:
        chunk_actions[..., -1] = 2 * chunk_actions[..., -1] - 1
        chunk_actions[..., -1] = torch.sign(chunk_actions[..., -1]) * -1.0
    return chunk_actions


def prepare_actions_for_mujoco(raw_chunk_actions, model_type):
    if raw_chunk_actions.shape[-1] >= 7:
        chunk_actions = np.concatenate(
            [raw_chunk_actions[..., :3], raw_chunk_actions[..., 6:7]], axis=-1
        )
    else:
        chunk_actions = raw_chunk_actions[..., :4]
    if _is_openpi_family(model_type):
        chunk_actions[..., -1] = np.clip(chunk_actions[..., -1], -1.0, 1.0)
    return chunk_actions


def prepare_actions_for_d4rl(
    raw_chunk_actions,
    action_dim: int,
    model_type,
) -> np.ndarray:
    # D4RL: take first action_dim dims from policy output
    raw = np.asarray(raw_chunk_actions, dtype=np.float32)
    chunk_actions = raw[..., :action_dim].copy()
    # openpi: clip last dim to match continuous action space
    if _is_openpi_family(model_type):
        chunk_actions[..., -1] = np.clip(chunk_actions[..., -1], -1.0, 1.0)
    return chunk_actions


def prepare_actions_for_roboverse(
    raw_chunk_actions,
    model_type,
) -> np.ndarray:
    chunk_actions = raw_chunk_actions
    if _is_openpi_family(model_type):
        chunk_actions[..., -1] = np.where(chunk_actions[..., -1] < 0.0, 1.0, 0.0)
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
    if isinstance(raw_chunk_actions, torch.Tensor):
        raw_chunk_actions = raw_chunk_actions.detach().cpu().contiguous()
        if raw_chunk_actions.dtype == torch.bfloat16:
            raw_chunk_actions = raw_chunk_actions.float()
        raw_chunk_actions = raw_chunk_actions.numpy()

    env_type = SupportedEnvType(env_type)
    if env_type == SupportedEnvType.LIBERO:
        chunk_actions = prepare_actions_for_libero(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
            env_cfg=env_cfg,
        )
    elif env_type == SupportedEnvType.WORLD_MODEL:
        # TODO: Implement prepare_actions_for_opensora_wm
        if wm_env_type == "libero":
            chunk_actions = prepare_actions_for_libero(
                raw_chunk_actions=raw_chunk_actions,
                model_type=model_type,
                env_cfg=env_cfg,
            )
        else:
            raise NotImplementedError(f"Env type {wm_env_type} not implemented")
    elif (
        env_type == SupportedEnvType.MANISKILL
        or env_type == SupportedEnvType.MANISKILL_RLT
    ):
        chunk_actions = prepare_actions_for_maniskill(
            raw_chunk_actions=raw_chunk_actions,
            num_action_chunks=num_action_chunks,
            action_dim=action_dim,
            action_scale=action_scale,
            policy=policy,
        )
    elif env_type == SupportedEnvType.ROBOTWIN:
        chunk_actions = prepare_actions_for_robotwin(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
            env_cfg=env_cfg,
        )
    elif env_type == SupportedEnvType.EMBODICHAIN:
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
    elif env_type == SupportedEnvType.ROBOCASA365:
        chunk_actions = prepare_actions_for_robocasa(
            raw_chunk_actions=raw_chunk_actions,
            action_dim=action_dim,
            model_type=model_type,
            env_cfg=env_cfg,
        )
    elif env_type == SupportedEnvType.ROBOCASA:
        chunk_actions = prepare_actions_for_robocasa(
            raw_chunk_actions=raw_chunk_actions,
            action_dim=action_dim,
            action_space=policy,
        )
    elif env_type == SupportedEnvType.REAL:
        chunk_actions = raw_chunk_actions
    elif env_type == SupportedEnvType.GENESIS:
        chunk_actions = prepare_actions_for_genesis(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.FRANKASIM:
        chunk_actions = prepare_actions_for_mujoco(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.D4RL:
        chunk_actions = prepare_actions_for_d4rl(
            raw_chunk_actions=raw_chunk_actions,
            action_dim=action_dim,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.ROBOVERSE:
        chunk_actions = prepare_actions_for_roboverse(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.DIFFUSION:
        chunk_actions = raw_chunk_actions
    elif env_type == SupportedEnvType.POLARIS:
        chunk_actions = prepare_actions_for_polaris(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    else:
        chunk_actions = raw_chunk_actions

    return chunk_actions
