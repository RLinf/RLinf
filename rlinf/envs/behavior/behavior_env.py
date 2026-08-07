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

import copy
import gc
import inspect
import json
import os
import time
from typing import ClassVar

import gymnasium as gym
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.behavior.instance_loader import (
    RLINF_REPLAY_METADATA_KEY,
    ActivityInstanceLoader,
)
from rlinf.envs.behavior.utils import (
    apply_env_wrapper,
    apply_runtime_renderer_settings,
    convert_uint8_rgb,
    setup_omni_cfg,
)
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.utils.logging import get_logger

__all__ = ["BehaviorEnv"]


def _preload_numba_llvmlite() -> None:
    # Isaac Sim's ``omni.isaac.core_archive`` ships an older numba in its
    # ``pip_prebundle`` and loads a few submodules during Kit startup,
    # which then mix with the venv's newer ``llvmlite`` and fail with
    # ``unknown attr 'nocapture'``. Preload the venv copies of just those
    # submodules so they win the ``sys.modules`` cache.
    import importlib

    for name in (
        "llvmlite",
        "numba",
        "numba.np.arrayobj",
        "numba.core.runtime.context",
    ):
        try:
            importlib.import_module(name)
        except Exception:
            pass


@ray.remote(num_cpus=1)
class BehaviorProcess:
    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        pipeline_stage_num: int,
        replay_seed_offset: int = 0,
    ):
        _preload_numba_llvmlite()
        from omnigibson.envs import VectorEnvironment

        self.logger = get_logger()
        self.pipeline_stage_num = pipeline_stage_num
        self.replay_seed_offset = replay_seed_offset
        self.group_size = int(OmegaConf.select(cfg, "group_size", default=1))
        if self.group_size <= 0:
            raise ValueError(f"env.group_size must be positive, got {self.group_size}.")
        omni_cfg = setup_omni_cfg(cfg)
        self.instance_loader = ActivityInstanceLoader.from_omni_cfg(
            omni_cfg, seed_offset=self.replay_seed_offset
        )

        # create env and apply env wrapper if enabled
        omni_cfg_dict = OmegaConf.to_container(
            omni_cfg,
            resolve=True,
            throw_on_missing=True,
        )
        # When pipeline stages > 1, each stage independently advances the
        # global physics per chunk step.  Divide physics_frequency so the
        # total physics rate stays at the configured value.
        if pipeline_stage_num > 1:
            omni_cfg_dict["env"]["physics_frequency"] = (
                omni_cfg_dict["env"]["physics_frequency"] / pipeline_stage_num
            )
        self.env = VectorEnvironment(num_envs, omni_cfg_dict)
        apply_runtime_renderer_settings()
        wrapper_name = OmegaConf.select(omni_cfg, "env.env_wrapper")
        self.env = apply_env_wrapper(self.env, wrapper_name)

        # Isaac Sim's `omni.kit.app` calls ``gc.disable()`` at startup.
        # OmniGibson has self-referential cycles and leaks memory when
        # cyclic GC is disabled. Since we do not need real-time performance,
        # enable cyclic GC here so that we do not encounter OOMs in long runs.
        gc.enable()

        step_signature = inspect.signature(self.env.step)
        step_params = step_signature.parameters.values()
        step_supports_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in step_params
        )
        self.step_supports_get_obs = (
            step_supports_kwargs or "get_obs" in step_signature.parameters
        )
        self.step_supports_render = (
            step_supports_kwargs or "render" in step_signature.parameters
        )
        self.step_supports_env_indices = "env_indices" in step_signature.parameters
        self.skip_intermediate_obs_in_chunk = bool(
            OmegaConf.select(cfg, "skip_intermediate_obs_in_chunk", default=False)
        )

        if self.skip_intermediate_obs_in_chunk and not self.step_supports_get_obs:
            self.logger.warning(
                "skip_intermediate_obs_in_chunk is True but OmniGibson env step does not "
                "support get_obs; this config will be ignored."
            )

        if self.pipeline_stage_num > 1 and not self.step_supports_env_indices:
            self.logger.warning(
                "pipeline_stage_num > 1 but OmniGibson env step does not support env_indices; "
                "this may cause inefficiency since every pipeline step will still "
                "advance every env with zeroed-out actions for inactive envs."
            )

        self.action_mask = self._parse_action_mask(cfg)
        (
            self.first_chunk_action_override_enabled,
            self.first_chunk_action_ids,
            self.first_chunk_action_value,
        ) = self._parse_first_chunk_action_override(cfg)
        self._first_chunk_action_override_pending = np.zeros(num_envs, dtype=bool)
        self.trace_robot_joints = self._trace_robot_joints_enabled(cfg)
        self._joint_trace_frame_idx = 0

    def get_activity_name(self):
        return self.instance_loader.activity_name

    def _call_step(self, actions, env_indices=None, get_obs=True, render=True):
        """Call ``self.env.step`` forwarding only the kwargs it supports."""
        kwargs = {}
        if self.step_supports_get_obs:
            kwargs["get_obs"] = get_obs
        if self.step_supports_render:
            kwargs["render"] = render
        if env_indices is not None:
            kwargs["env_indices"] = env_indices
        return self.env.step(actions, **kwargs)

    def _call_reset(self, reset_indices=None, get_obs=True):
        """Call ``self.env.reset`` through one normalized code path."""
        kwargs = {"get_obs": get_obs}
        if reset_indices is not None:
            kwargs["env_indices"] = reset_indices
        return self.env.reset(**kwargs)

    def _step_shard(
        self,
        actions: torch.Tensor,
        env_indices: list[int],
        need_obs: bool,
    ):
        """Step one shard for a single chunk timestep.

        ``actions`` is the zero-padded ``[num_shard, action_dim]`` action
        tensor (inactive rows already carry zero actions). ``env_indices``
        is the ascending list of local rows that should advance.

        Returns outputs only for ``env_indices``, in that same order.
        """
        if self.step_supports_env_indices:
            raw_obs, rewards, terminates, truncates, infos = self._call_step(
                [actions[i] for i in env_indices],
                env_indices=env_indices,
                get_obs=need_obs,
                render=need_obs,
            )
        else:
            raw_obs, rewards, terminates, truncates, infos = self._call_step(
                actions,
                get_obs=need_obs,
                render=need_obs,
            )
            if need_obs:
                raw_obs = [raw_obs[i] for i in env_indices]
            rewards = [rewards[i] for i in env_indices]
            terminates = [terminates[i] for i in env_indices]
            truncates = [truncates[i] for i in env_indices]
            infos = [infos[i] for i in env_indices]

        return (
            list(raw_obs) if need_obs else None,
            to_tensor(rewards),
            to_tensor(terminates),
            to_tensor(truncates),
            list(infos),
        )

    @staticmethod
    def _parse_action_mask(cfg: DictConfig) -> list[bool] | None:
        mask_cfg = OmegaConf.select(cfg, "action_mask", default=None)
        if mask_cfg is None:
            return None
        if not bool(OmegaConf.select(cfg, "action_mask.enabled", default=True)):
            return None

        mask = OmegaConf.select(cfg, "action_mask.mask", default=None)
        if mask is None:
            action_dim = int(OmegaConf.select(cfg, "action_mask.action_dim", default=23))
            mask = [True] * action_dim
            if bool(OmegaConf.select(cfg, "action_mask.freeze_base", default=False)):
                mask[:3] = [False] * min(3, action_dim)
            if bool(OmegaConf.select(cfg, "action_mask.freeze_trunk", default=False)):
                start, end = 3, min(7, action_dim)
                mask[start:end] = [False] * (end - start)
        else:
            mask = OmegaConf.to_container(mask, resolve=True)

        if not isinstance(mask, (list, tuple)) or not mask:
            raise ValueError("env.action_mask.mask must be a non-empty bool list.")
        return [bool(value) for value in mask]

    @staticmethod
    def _parse_first_chunk_action_override(
        cfg: DictConfig,
    ) -> tuple[bool, list[int], float]:
        override_cfg = OmegaConf.select(cfg, "first_chunk_action_override", default=None)
        if override_cfg is None:
            return False, [], -1.0
        enabled = bool(
            OmegaConf.select(cfg, "first_chunk_action_override.enabled", default=False)
        )
        action_ids = OmegaConf.select(
            cfg, "first_chunk_action_override.action_ids", default=[]
        )
        if action_ids is None:
            action_ids = []
        if not isinstance(action_ids, (list, tuple)):
            action_ids = OmegaConf.to_container(action_ids, resolve=True)
        if not isinstance(action_ids, (list, tuple)):
            raise ValueError(
                "env.first_chunk_action_override.action_ids must be a list."
            )
        action_value = float(
            OmegaConf.select(cfg, "first_chunk_action_override.value", default=-1.0)
        )
        return enabled, [int(action_id) for action_id in action_ids], action_value

    @staticmethod
    def _action_like_values(action, values: list[float]):
        if torch.is_tensor(action):
            return torch.as_tensor(values, dtype=action.dtype, device=action.device)
        dtype = getattr(action, "dtype", np.float32)
        return np.asarray(values, dtype=dtype)

    @staticmethod
    def _robot_joint_positions_list(robot) -> list[float]:
        joint_positions = robot.get_joint_positions()
        if torch.is_tensor(joint_positions):
            return joint_positions.detach().cpu().reshape(-1).tolist()
        return np.asarray(joint_positions).reshape(-1).tolist()

    def _r1pro_noop_action(self, robot, action):
        action_dim = int(action.shape[-1])
        values = [0.0] * action_dim
        joint_positions = self._robot_joint_positions_list(robot)
        if action_dim >= 23 and len(joint_positions) >= 28:
            values[3:7] = joint_positions[6:10]
            values[7:14] = [joint_positions[i] for i in (10, 12, 14, 16, 18, 20, 22)]
            values[14:21] = [joint_positions[i] for i in (11, 13, 15, 17, 19, 21, 23)]
            values[21] = joint_positions[24] + joint_positions[25]
            values[22] = joint_positions[26] + joint_positions[27]
        return self._action_like_values(action, values)

    def _apply_action_mask(self, actions):
        if self.action_mask is None:
            return actions

        action_dim = int(actions.shape[-1])
        if action_dim != len(self.action_mask):
            raise ValueError(
                f"env.action_mask.mask length {len(self.action_mask)} does not "
                f"match action_dim {action_dim}."
            )

        masked_actions = actions.clone() if torch.is_tensor(actions) else actions.copy()
        child_envs = getattr(self.env, "envs", [])
        for env_idx in range(int(actions.shape[0])):
            robot = self._get_robot_from_child_env(child_envs[env_idx])
            if robot is None:
                continue
            noop_action = self._r1pro_noop_action(robot, actions[env_idx])
            for action_idx, use_policy_action in enumerate(self.action_mask):
                if not use_policy_action:
                    masked_actions[env_idx, action_idx] = noop_action[action_idx]
        return masked_actions

    def _apply_first_chunk_action_override(self, actions, env_mask: np.ndarray):
        if (
            not self.first_chunk_action_override_enabled
            or not self.first_chunk_action_ids
            or not env_mask.any()
        ):
            return actions

        action_dim = int(actions.shape[-1])
        invalid_action_ids = [
            action_id
            for action_id in self.first_chunk_action_ids
            if action_id < 0 or action_id >= action_dim
        ]
        if invalid_action_ids:
            raise ValueError(
                "env.first_chunk_action_override.action_ids contains invalid "
                f"indices {invalid_action_ids} for action_dim {action_dim}."
            )

        overridden_actions = (
            actions.clone() if torch.is_tensor(actions) else actions.copy()
        )
        env_indices = np.flatnonzero(env_mask).tolist()
        for action_id in self.first_chunk_action_ids:
            overridden_actions[env_indices, action_id] = self.first_chunk_action_value
        return overridden_actions

    @staticmethod
    def _get_robot_from_child_env(child_env):
        base_env = BehaviorProcess._unwrap_child_env(child_env)
        task = getattr(base_env, "task", None)
        if task is not None and hasattr(task, "get_agent"):
            robot = task.get_agent(base_env)
            if robot is not None:
                return robot
        robots = getattr(base_env, "robots", None)
        if robots:
            return robots[0]
        return None

    @staticmethod
    def _trace_robot_joints_enabled(cfg: DictConfig) -> bool:
        cfg_value = bool(OmegaConf.select(cfg, "trace_robot_joints", default=False))
        env_value = os.environ.get("RLINF_BEHAVIOR_TRACE_JOINTS", "")
        return cfg_value or env_value.lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _tensor_to_float_list(value, digits: int = 6) -> list[float]:
        if value is None:
            return []
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            value = [value]
        return [round(float(item), digits) for item in value]

    @staticmethod
    def _robot_joint_names(robot) -> list[str]:
        joints = getattr(robot, "joints", None)
        if isinstance(joints, dict):
            return list(joints.keys())
        return []

    def _log_robot_joint_trace(
        self,
        child_env,
        local_env_idx: int,
        event: str,
        chunk_step_idx: int | None = None,
        action=None,
    ) -> None:
        if not self.trace_robot_joints:
            return

        robot = self._get_robot_from_child_env(child_env)
        if robot is None:
            return

        get_joint_velocities = getattr(robot, "get_joint_velocities", None)
        if callable(get_joint_velocities):
            joint_velocities = get_joint_velocities()
        else:
            joint_velocities = None
        robot_pos, robot_quat = robot.get_position_orientation(frame="world")
        record = {
            "event": event,
            "replay_seed_offset": self.replay_seed_offset,
            "local_env_idx": int(local_env_idx),
            "frame_idx": int(self._joint_trace_frame_idx),
            "chunk_step_idx": None if chunk_step_idx is None else int(chunk_step_idx),
            "robot_name": getattr(robot, "name", None),
            "joint_names": self._robot_joint_names(robot),
            "joint_positions": self._tensor_to_float_list(
                robot.get_joint_positions()
            ),
            "joint_velocities": self._tensor_to_float_list(joint_velocities),
            "robot_position_world": self._tensor_to_float_list(robot_pos),
            "robot_quat_world": self._tensor_to_float_list(robot_quat),
        }
        if action is not None:
            record["action"] = self._tensor_to_float_list(action)
        print(
            f"RLINF_BEHAVIOR_JOINT_TRACE {json.dumps(record, sort_keys=True)}",
            flush=True,
        )

    def _log_all_robot_joint_traces(
        self,
        event: str,
        env_indices: list[int] | None = None,
        chunk_step_idx: int | None = None,
        actions=None,
    ) -> None:
        if not self.trace_robot_joints:
            return

        child_envs = getattr(self.env, "envs", [])
        if env_indices is None:
            env_indices = list(range(len(child_envs)))
        for env_idx in env_indices:
            action = None
            if actions is not None:
                action = actions[env_idx]
            self._log_robot_joint_trace(
                child_envs[env_idx],
                local_env_idx=env_idx,
                event=event,
                chunk_step_idx=chunk_step_idx,
                action=action,
            )
        self._joint_trace_frame_idx += 1

    @classmethod
    def _annotate_behavior_hold_metrics(cls, child_env, info: dict | None) -> dict:
        info = {} if info is None else info
        base_env = cls._unwrap_child_env(child_env)
        reward_info = info.get("reward")
        if not isinstance(reward_info, dict):
            reward_info = {}
            info["reward"] = reward_info
        task_info = reward_info.get("task_specific")
        if not isinstance(task_info, dict):
            task_info = {}
            reward_info["task_specific"] = task_info

        task = getattr(base_env, "task", None)
        activity_instance_id = cls._to_int_or_none(
            getattr(task, "activity_instance_id", None)
        )
        if activity_instance_id is not None:
            task_info["activity_instance_id"] = activity_instance_id

        try:
            from omnigibson.reward_functions.support_utils import (
                get_stage_objects_by_name,
                is_target_in_hand,
            )

            robot = cls._get_robot_from_child_env(base_env)
            task_reward = cls._task_reward(base_env)
            target_obj = getattr(task_reward, "_radio_obj", None)
            if target_obj is None:
                target_objects = get_stage_objects_by_name(base_env, ("radio_89",))
                target_obj = target_objects[0] if target_objects else None
            if robot is not None and target_obj is not None:
                task_info["held_in_hand"] = bool(is_target_in_hand(robot, target_obj))
        except Exception:
            task_info["held_in_hand_available"] = False

        return info

    def chunk_step(self, actions, env_indices):
        """Step a full chunk for one shard.

        Args:
            actions: Zero-padded ``[num_shard, chunk, action_dim]`` action
                matrix for this VectorEnvironment.
            env_indices: Ascending local rows that should advance every
                chunk step.
        """
        _, chunk_size, _ = actions.shape

        # Apply first-chunk action override for pending env slots.
        if self._first_chunk_action_override_pending.any():
            full_mask = np.zeros(actions.shape[0], dtype=bool)
            full_mask[env_indices] = True
            pending_mask = full_mask & self._first_chunk_action_override_pending
            if pending_mask.any():
                actions[:, 0] = self._apply_first_chunk_action_override(
                    actions[:, 0], pending_mask
                )
            self._first_chunk_action_override_pending[:] = False

        self._log_all_robot_joint_traces("pre_chunk", env_indices)

        results: list[tuple] = []
        for t in range(chunk_size):
            is_last = t == chunk_size - 1
            need_obs = not self.skip_intermediate_obs_in_chunk or is_last
            step_actions = self._apply_action_mask(actions[:, t])
            self._log_all_robot_joint_traces(
                "pre_step", env_indices, chunk_step_idx=t, actions=step_actions
            )
            results.append(
                self._step_shard(step_actions, env_indices, need_obs=need_obs)
            )
            # Annotate hold/grasp metrics on the last step's infos.
            if is_last:
                _obs, _rewards, _terminates, _truncates, step_infos = results[-1]
                child_envs = getattr(self.env, "envs", [])
                for local_idx, env_idx in enumerate(env_indices):
                    step_infos[local_idx] = self._annotate_behavior_hold_metrics(
                        child_envs[env_idx], step_infos[local_idx]
                    )

        self._log_all_robot_joint_traces("post_chunk", env_indices)
        return tuple(zip(*results))

    @staticmethod
    def _parse_reset_payload(payload):
        """Parse a reset payload into (reset_indices, instance_ids, is_full_reset).

        Supports three formats:
        - ``None``: reset all envs.
        - ``list[bool]``: each element indicates whether to reset that env.
        - ``list[dict]``: each dict may contain ``reset`` (bool), ``full_reset``
          (bool), and ``instance_id`` (int) keys.

        Args:
            payload: Reset specification in one of the supported formats.

        Returns:
            Tuple of (reset_indices, instance_ids, is_full_reset).
            ``instance_ids`` is None when not provided.

        Raises:
            ValueError: If instance_ids are provided for some but not all
                reset envs.
        """
        if payload is None:
            return None, None, False

        if payload and all(isinstance(item, dict) for item in payload):
            reset_indices = [
                idx
                for idx, item in enumerate(payload)
                if bool(item.get("reset", True))
            ]
            is_full_reset = all(
                bool(item.get("full_reset", False)) for item in payload
            )
            instance_ids = [
                int(payload[idx]["instance_id"])
                for idx in reset_indices
                if payload[idx].get("instance_id") is not None
            ]
            if instance_ids and len(instance_ids) != len(reset_indices):
                raise ValueError(
                    "Reset payload must provide instance_id for every reset env "
                    "or for none of them."
                )
            return reset_indices, instance_ids or None, is_full_reset

        # Legacy format: list[bool] where True means reset.
        reset_indices = [idx for idx, flag in enumerate(payload) if bool(flag)]
        return reset_indices, None, False

    def reset(self, reset_indices_or_payload=None, get_obs=True):
        # Detect payload format (dict list → replay reset with per-env
        # instance IDs).
        if reset_indices_or_payload is None:
            # Full reset of all envs — use the fast vectorized path.
            self.instance_loader.prepare_reset(self.env)
            result = self._call_reset(get_obs=get_obs)
            if not get_obs:
                return None, None
            raw_obs, infos = result
            return list(raw_obs), list(infos)

        if reset_indices_or_payload and isinstance(
            reset_indices_or_payload[0], dict
        ):
            # Payload-based reset with optional per-env instance IDs (replay).
            reset_indices, instance_ids, _is_full_reset = self._parse_reset_payload(
                reset_indices_or_payload
            )
            if not reset_indices:
                return [], []
            raw_obs, infos = self._reset_env_indices(
                reset_indices, instance_ids=instance_ids
            )
            return raw_obs, infos

        # Legacy format: list[int] — partial reset of selected env indices
        # (used by env_reset_slice which passes local_rows).
        reset_indices = reset_indices_or_payload
        self.instance_loader.prepare_reset(self.env)
        result = self._call_reset(
            reset_indices=reset_indices, get_obs=get_obs
        )
        if not get_obs:
            return None, None
        raw_obs, infos = result
        return list(raw_obs), list(infos)

    @staticmethod
    def _unwrap_child_env(child_env):
        """Return the underlying OmniGibson env, unwrapping any thin wrappers."""
        return child_env

    @staticmethod
    def _to_int_or_none(value):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _task_reward(child_env):
        try:
            return child_env.task.task_reward
        except Exception:
            return None

    @classmethod
    def _stage_idx_from_info(cls, info: dict | None) -> int | None:
        if info is None:
            return None
        task_reward = info.get("task_reward", {})
        if isinstance(task_reward, dict):
            return cls._to_int_or_none(task_reward.get("current_stage_idx"))
        return cls._to_int_or_none(info.get("current_stage_idx"))

    @classmethod
    def _stage_idx_from_reward(cls, child_env) -> int | None:
        task_reward = cls._task_reward(child_env)
        if task_reward is None:
            return None
        return cls._to_int_or_none(
            getattr(task_reward, "current_stage_idx", None)
        )

    def _apply_replay_tro_metadata(self, child_env, info: dict | None) -> dict:
        """Inject RLinf replay metadata from tro_state into the env info dict.

        Reads ``RLINF_REPLAY_METADATA_KEY`` from the scene, populates
        ``info["replay_init"]`` with replay fields, sets stage information
        in ``info["reward"]["task_specific"]``, and calls
        ``task_reward.set_active_stage_index`` so downstream reward
        computations start from the intended stage.
        """
        base_env = self._unwrap_child_env(child_env)
        scene = getattr(base_env, "scene", None)
        if scene is None or not hasattr(scene, "get_task_metadata"):
            return {} if info is None else info

        metadata = scene.get_task_metadata(key=RLINF_REPLAY_METADATA_KEY)
        if not isinstance(metadata, dict):
            return {} if info is None else info

        info = {} if info is None else info
        stage_idx = self._to_int_or_none(metadata.get("stage_index"))
        stage_prompts = metadata.get("stage_prompts")
        if not isinstance(stage_prompts, (list, tuple)):
            stage_prompts = []
        stage_prompts = [
            str(prompt).strip() for prompt in stage_prompts if str(prompt).strip()
        ]

        task_reward = self._task_reward(base_env)
        total_stages = self._to_int_or_none(getattr(task_reward, "_total_stages", None))
        if stage_idx is not None and hasattr(task_reward, "set_active_stage_index"):
            if total_stages is None or 0 <= stage_idx < total_stages:
                task_reward.set_active_stage_index(stage_idx)

        replay_info = info.get("replay_init")
        if not isinstance(replay_info, dict):
            replay_info = {}
            info["replay_init"] = replay_info
        replay_info.update(metadata)
        replay_info["replay_stage_prompts"] = stage_prompts
        if stage_idx is not None:
            replay_info["replay_stage_idx"] = stage_idx

        reward_info = info.get("reward")
        if not isinstance(reward_info, dict):
            reward_info = {}
            info["reward"] = reward_info
        task_info = reward_info.get("task_specific")
        if not isinstance(task_info, dict):
            task_info = {}
            reward_info["task_specific"] = task_info

        if stage_idx is not None:
            task_info["current_stage_idx"] = stage_idx
            task_info.setdefault("completed_stage_count", stage_idx)
        if total_stages is not None:
            task_info["total_stage_count"] = total_stages
        if stage_idx is not None and 0 <= stage_idx < len(stage_prompts):
            task_info["current_stage_prompt"] = stage_prompts[stage_idx]
        stage_defs = getattr(task_reward, "_stage_defs", None)
        if (
            isinstance(stage_defs, list)
            and stage_idx is not None
            and 0 <= stage_idx < len(stage_defs)
        ):
            stage_name = stage_defs[stage_idx].get("name")
            if stage_name is not None:
                task_info["current_stage_name"] = stage_name
        task_info.setdefault("completion_bonus", 0.0)
        return info

    def _reset_env_indices(
        self, reset_indices: list[int], instance_ids: list[int] | None = None
    ):
        """Reset specific env slots, optionally to specific instance IDs.

        When ``instance_ids`` is provided, each env is reset to its assigned
        instance (disable group-reset to avoid cross-env interference).
        """
        from types import SimpleNamespace

        child_envs = getattr(self.env, "envs", [])
        selected_env = SimpleNamespace(envs=[child_envs[i] for i in reset_indices])
        reset_group_size = 1 if instance_ids is not None else self.group_size
        self.instance_loader.prepare_reset(
            selected_env, instance_ids=instance_ids, group_size=reset_group_size
        )

        raw_obs, infos = [], []
        for env_idx in reset_indices:
            obs, info = child_envs[env_idx].reset(get_obs=True)
            info = self._apply_replay_tro_metadata(child_envs[env_idx], info)
            raw_obs.append(obs)
            infos.append(info)
        return raw_obs, infos

    def dump_replay_tro_states(self, payload: dict) -> list[dict]:
        """Dump replay tro_states inside this BehaviorProcess.

        Delegates to the shared implementation in
        :mod:`rlinf.envs.behavior.replay_tro_state_dumper`.
        """
        from rlinf.envs.behavior.replay_tro_state_dumper import (
            dump_replay_tro_states as dump_replay_tro_states_impl,
        )

        return dump_replay_tro_states_impl(self, payload)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


class BehaviorProcessPool:
    """Singleton OmniGibson subprocess pool manager.

    Use :meth:`acquire_shared` to obtain the singleton pool; use :meth:`release_shared` when done.
    """

    _shared_pool: ClassVar["BehaviorProcessPool | None"] = None
    _shared_refcount: ClassVar[int] = 0
    _pipeline_next_idx: ClassVar[int] = 0

    @classmethod
    def acquire_shared(
        cls,
        cfg: DictConfig,
        worker_info,
        pipeline_stage_num: int,
        num_envs: int,
        seed_offset: int = 0,
    ) -> tuple["BehaviorProcessPool", int]:
        """Attach to the shared pool and return ``(pool, pool_offset)``."""
        if cls._shared_pool is None:  # pool init
            total_envs = int(OmegaConf.select(cfg, "total_num_envs", default=None))
            total_envs_per_worker = total_envs // worker_info.group_world_size
            num_env_subprocess = int(
                OmegaConf.select(cfg, "num_env_subprocess", default=1)
            )
            cls._shared_pool = cls(
                cfg,
                total_envs_per_worker,
                num_env_subprocess,
                pipeline_stage_num,
                seed_offset,
            )

        idx = cls._pipeline_next_idx
        global_offset = idx * num_envs
        cls._pipeline_next_idx += 1
        cls._shared_refcount += 1

        pool = cls._shared_pool

        if global_offset + num_envs > pool.total_num_envs:
            raise ValueError(
                f"BehaviorEnv slice [{global_offset}, {global_offset + num_envs}) "
                f"exceeds pool total_num_envs={pool.total_num_envs}."
            )
        return pool, global_offset

    @classmethod
    def release_shared(cls) -> None:
        """Drop refcount; tear down the shared pool when the last env releases."""
        if cls._shared_pool is None:
            return
        cls._shared_refcount -= 1
        if cls._shared_refcount <= 0:
            cls._shared_pool.close()
            cls._shared_pool = None
            cls._pipeline_next_idx = 0

    def __init__(
        self,
        cfg: DictConfig,
        total_num_envs: int,
        num_env_subprocess: int,
        pipeline_stage_num: int,
        seed_offset: int = 0,
    ):
        if total_num_envs % num_env_subprocess != 0:
            raise ValueError(
                f"total_num_envs({total_num_envs}) must be divisible by num_env_subprocess({num_env_subprocess})"
            )

        self.logger = get_logger()
        self.cfg = cfg
        self.total_num_envs = total_num_envs
        self.num_env_subprocess = num_env_subprocess
        self.num_env_shard = total_num_envs // num_env_subprocess
        self.seed_offset = seed_offset
        self.skip_intermediate_obs_in_chunk = bool(
            OmegaConf.select(cfg, "skip_intermediate_obs_in_chunk", default=False)
        )

        # Create subprocess actors with a retry/backoff loop. Actor startup
        # can fail (e.g. simulator plugin errors); retry a few times to handle
        # transient failures. Configurable via `behavior.init_retry_*` keys.
        max_attempts = int(
            OmegaConf.select(cfg, "behavior.init_retry_count", default=3)
        )
        retry_delay = float(
            OmegaConf.select(cfg, "behavior.init_retry_delay", default=5.0)
        )
        backoff = float(
            OmegaConf.select(cfg, "behavior.init_retry_backoff", default=2.0)
        )

        for attempt in range(1, max_attempts + 1):
            try:
                self.env_processes = [
                    BehaviorProcess.remote(
                        self.cfg,
                        self.num_env_shard,
                        pipeline_stage_num,
                        self.seed_offset + process_idx,
                    )
                    for process_idx in range(self.num_env_subprocess)
                ]

                # Wait for all instances to initialize and fetch their activity name
                activity_names_refs = [
                    proc.get_activity_name.remote() for proc in self.env_processes
                ]
                activity_names = ray.get(activity_names_refs)
                break
            except Exception as e:  # noqa: BLE001 - we want to catch any Ray/OmniGibson init error
                # Best-effort cleanup of any partially-created actors
                for proc in getattr(self, "env_processes", []):
                    try:
                        ray.kill(proc)
                    except Exception:
                        pass
                self.env_processes = []

                if attempt >= max_attempts:
                    self.logger.error(
                        "Failed to start BehaviorProcess actors after %d attempts: %s",
                        attempt,
                        e,
                    )
                    raise

                self.logger.warning(
                    "BehaviorProcess creation failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    max_attempts,
                    e,
                    retry_delay,
                )
                time.sleep(retry_delay)
                retry_delay *= backoff

        if len(set(activity_names)) != 1:
            raise RuntimeError(
                f"Behavior env subprocesses reported different activity_name: "
                f"{activity_names}"
            )
        self.activity_name = activity_names[0]

    def _slice_plan(
        self, global_start: int, num_envs: int
    ) -> list[tuple[int, list[int], list[int]]]:
        """Build the per-subprocess plan for a contiguous global slice.

        Returns entries of ``(subproc_idx, slice_positions, local_rows)``.
        ``slice_positions`` are indices inside the caller's slice and
        ``local_rows`` are the matching rows owned by that subprocess.
        """
        slice_positions_by_proc = [[] for _ in range(self.num_env_subprocess)]
        local_rows_by_proc = [[] for _ in range(self.num_env_subprocess)]
        for pos in range(num_envs):
            global_idx = global_start + pos
            sp = global_idx % self.num_env_subprocess
            slice_positions_by_proc[sp].append(pos)
            local_rows_by_proc[sp].append(global_idx // self.num_env_subprocess)

        return [
            (sp, slice_positions_by_proc[sp], local_rows_by_proc[sp])
            for sp in range(self.num_env_subprocess)
            if slice_positions_by_proc[sp]
        ]

    def env_reset_slice(self, global_start: int, num_envs: int):
        """Reset envs in ``[global_start, global_start + num_envs)``."""
        if num_envs == 0:
            return [], []
        plan = self._slice_plan(global_start, num_envs)
        refs = [
            self.env_processes[sp].reset.remote(local_rows)
            for sp, _positions, local_rows in plan
        ]

        shard_results = ray.get(refs)
        all_raw_obs: list = [None] * num_envs
        all_infos: list = [None] * num_envs
        for (raw_obs, infos), (_sp, positions, _local_rows) in zip(shard_results, plan):
            for pos, obs, info in zip(positions, raw_obs, infos):
                all_raw_obs[pos] = obs
                all_infos[pos] = info
        return all_raw_obs, all_infos

    def env_reset_slice_partial(
        self, global_start: int, num_envs: int, payload_shards: list
    ):
        """Reset a contiguous slice with per-env payload (e.g. instance IDs)."""
        if num_envs == 0:
            return [], []
        plan = self._slice_plan(global_start, num_envs)
        refs = [
            self.env_processes[sp].reset.remote(payload_shards[sp])
            for sp, _positions, _local_rows in plan
        ]
        shard_results = ray.get(refs)
        all_raw_obs: list = [None] * num_envs
        all_infos: list = [None] * num_envs
        for (raw_obs, infos), (_sp, positions, _local_rows) in zip(shard_results, plan):
            for pos, obs, info in zip(positions, raw_obs, infos):
                all_raw_obs[pos] = obs
                all_infos[pos] = info
        return all_raw_obs, all_infos

    def env_chunk_step_slice(
        self,
        global_start: int,
        slice_num_envs: int,
        chunk_actions: torch.Tensor,
    ):
        """Run chunk_step on shards; pool handles all sharding/merging.
        ``chunk_actions`` must be ``[slice_num_envs, chunk, action_dim]``.
        """
        chunk_size = chunk_actions.shape[1]
        action_dim = chunk_actions.shape[-1]
        plan = self._slice_plan(global_start, slice_num_envs)

        refs = []
        for sp, positions, local_rows in plan:
            actions_j = torch.zeros(
                self.num_env_shard,
                chunk_size,
                action_dim,
                dtype=chunk_actions.dtype,
            )
            actions_j[local_rows] = chunk_actions[positions]
            refs.append(self.env_processes[sp].chunk_step.remote(actions_j, local_rows))

        shard_results = ray.get(refs)
        return self._merge_shards(shard_results, plan, slice_num_envs, chunk_size)

    def _merge_shards(
        self,
        shard_results: list,
        plan: list[tuple[int, list[int], list[int]]],
        slice_num_envs: int,
        chunk_size: int,
    ):
        """Gather per-subprocess shard outputs into ``[chunk][slice]`` order."""
        merged_obs: list = []
        merged_rewards: list = []
        merged_terms: list = []
        merged_trunc: list = []
        merged_infos: list = []
        for t in range(chunk_size):
            is_last = t == chunk_size - 1
            need_obs = not self.skip_intermediate_obs_in_chunk or is_last
            obs_t: list | None = [None] * slice_num_envs if need_obs else None
            reward_t = torch.zeros(slice_num_envs, dtype=torch.float32)
            term_t = torch.zeros(slice_num_envs, dtype=torch.bool)
            trunc_t = torch.zeros(slice_num_envs, dtype=torch.bool)
            info_t: list = [{} for _ in range(slice_num_envs)]
            for (obs_per_t, rewards_per_t, terms_per_t, truncs_per_t, infos_per_t), (
                _sp,
                positions,
                _local_rows,
            ) in zip(shard_results, plan):
                obs_at_t = obs_per_t[t]
                rewards_at_t = rewards_per_t[t]
                terms_at_t = terms_per_t[t]
                truncs_at_t = truncs_per_t[t]
                infos_at_t = infos_per_t[t]
                for i, pos in enumerate(positions):
                    if need_obs:
                        obs_t[pos] = obs_at_t[i]
                    reward_t[pos] = float(rewards_at_t[i])
                    term_t[pos] = bool(terms_at_t[i])
                    trunc_t[pos] = bool(truncs_at_t[i])
                    info_t[pos] = infos_at_t[i]
            merged_obs.append(obs_t)
            merged_rewards.append(reward_t)
            merged_terms.append(term_t)
            merged_trunc.append(trunc_t)
            merged_infos.append(info_t)
        return merged_obs, merged_rewards, merged_terms, merged_trunc, merged_infos

    def close(self) -> None:
        refs = [proc.close.remote() for proc in self.env_processes]
        ray.get(refs)

        # Kill the procs to free up resources immediately
        for proc in self.env_processes:
            ray.kill(proc)

        self.env_processes = []


class BehaviorEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        self.cfg = cfg
        self.reward_coef = cfg.get("reward_coef", 1)
        self.is_eval = cfg.get("is_eval", False)

        self.num_envs = num_envs
        self.ignore_terminations = cfg.ignore_terminations
        self.seed_offset = seed_offset
        self.seed = self.cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics
        self._is_start = True
        self.enable_offload = cfg.get("enable_offload", False)
        self.enable_init_offload = cfg.get("enable_init_offload", True)
        self.pool = None
        self.pool_offset = None
        self.task_description = None
        if total_num_processes % worker_info.group_world_size != 0:
            raise ValueError(
                f"total_num_processes ({total_num_processes}) must be divisible by "
                f"worker_info.group_world_size ({worker_info.group_world_size}) to infer pipeline_stage_num."
            )
        self.pipeline_stage_num = total_num_processes // worker_info.group_world_size

        self.auto_reset = cfg.auto_reset
        self.max_episode_steps = torch.tensor(cfg.max_episode_steps)
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.success_stage_idx = cfg.get("success_stage_idx", None)
        if self.success_stage_idx is not None:
            self.success_stage_idx = int(self.success_stage_idx)
        reward_mode = OmegaConf.select(
            cfg, "omni_config.task.reward_config.reward_mode", default=""
        )
        self.behavior_reward_mode = str(reward_mode).lower()
        self.stage_sparse_reward = self.behavior_reward_mode == "stage_sparse"
        self.stage_weighted_reward = self.behavior_reward_mode == "stage_weighted"
        self.stage_cumulative_reward = self.behavior_reward_mode == "stage_cumulative"
        weights = OmegaConf.select(
            cfg, "omni_config.task.reward_config.stage_reward_weights", default=None
        )
        if weights is not None and not isinstance(weights, (list, tuple)):
            weights = OmegaConf.to_container(weights, resolve=True)
        self.stage_reward_weights = (
            None if weights is None else [float(weight) for weight in weights]
        )
        if self.stage_weighted_reward and not self.stage_reward_weights:
            raise ValueError(
                "omni_config.task.reward_config.stage_reward_weights must be set "
                "when reward_mode is stage_weighted."
            )
        if self.stage_sparse_reward and self.success_stage_idx is None:
            raise ValueError(
                "env.success_stage_idx must be set when "
                "omni_config.task.reward_config.reward_mode is stage_sparse."
            )
        self.use_rel_reward = cfg.use_rel_reward
        prompt_override = self.cfg.get("prompt_override", None)
        if prompt_override is not None:
            prompt_override = str(prompt_override).strip() or None

        use_subtask_prompt_cfg = self.cfg.get("use_subtask_prompt", False)
        if isinstance(use_subtask_prompt_cfg, str):
            legacy_prompt = use_subtask_prompt_cfg.strip()
            self.use_subtask_prompt = False
            if prompt_override is None and legacy_prompt:
                prompt_override = legacy_prompt
        else:
            self.use_subtask_prompt = bool(use_subtask_prompt_cfg)
        self.prompt_override = prompt_override
        self._ordered_reset_epoch = 0
        self._ordered_reset_instance_ids = None
        self._stage_prompt_lists: list[list[str] | None] = [None] * self.num_envs
        self.trunk_proprio_randomization = self._parse_trunk_proprio_randomization(cfg)
        self._trunk_proprio_random_values = None
        self._trunk_proprio_rng = torch.Generator()
        self._trunk_proprio_rng.manual_seed(self.seed + 100003)
        if self.record_metrics:
            self._init_metrics()
        if not (self.enable_offload and not self.enable_init_offload):
            self._ensure_pool()
            self._init_env()

    def _ensure_pool(self):
        if self.pool is None:
            self.pool, self.pool_offset = BehaviorProcessPool.acquire_shared(
                self.cfg,
                self.worker_info,
                self.pipeline_stage_num,
                self.num_envs,
                self.seed_offset,
            )

    def _load_tasks_cfg(self, activity_name: str):
        # Read task description

        task_description_path = os.path.join(
            os.path.dirname(__file__), "behavior_task.jsonl"
        )
        with open(task_description_path, "r") as f:
            text = f.read()
            task_description = [json.loads(x) for x in text.strip().split("\n") if x]
        task_description_map = {
            task_description[i]["task_name"]: task_description[i]["task"]
            for i in range(len(task_description))
        }
        self.task_description = task_description_map[activity_name]

    def _init_env(self):
        self._ensure_pool()
        self._load_tasks_cfg(self.pool.activity_name)

    def env_reset(self):
        self._ensure_pool()
        return self.pool.env_reset_slice(self.pool_offset, self.num_envs)

    def env_chunk_step(self, chunk_actions: torch.Tensor):
        self._ensure_pool()
        return self.pool.env_chunk_step_slice(
            self.pool_offset,
            self.num_envs,
            chunk_actions,
        )

    def env_reset_partial(self, reset_mask):
        """Reset env slots indicated by ``reset_mask``, returning
        ``(reset_indices, raw_obs, raw_infos)`` for only the reset slots."""
        reset_mask = [bool(flag) for flag in reset_mask]
        reset_indices = [
            env_idx for env_idx, should_reset in enumerate(reset_mask) if should_reset
        ]
        if not reset_indices:
            return [], [], []
        instance_ids = self._ordered_reset_ids_for_indices(reset_indices)
        payload = self._reset_payload_with_instance_ids(reset_mask, instance_ids)
        self._ensure_pool()
        s = self.pool.num_env_shard
        payload_shards = [
            payload[i * s : (i + 1) * s]
            for i in range(self.pool.num_env_subprocess)
        ]
        all_raw_obs, all_infos = self.pool.env_reset_slice_partial(
            self.pool_offset,
            self.num_envs,
            payload_shards,
        )
        # Extract only the reset slots.
        raw_obs = [all_raw_obs[i] for i in reset_indices]
        raw_infos = [all_infos[i] for i in reset_indices]
        return reset_indices, raw_obs, raw_infos

    def _extract_obs_image(self, raw_obs):
        state = None
        for sensor_data in raw_obs.values():
            assert isinstance(sensor_data, dict)
            for k, v in sensor_data.items():
                if "left_realsense_link:Camera:0" in k:
                    left_image = convert_uint8_rgb(v["rgb"])
                elif "right_realsense_link:Camera:0" in k:
                    right_image = convert_uint8_rgb(v["rgb"])
                elif "zed_link:Camera:0" in k:
                    zed_image = convert_uint8_rgb(v["rgb"])
                elif "proprio" in k:
                    state = v
        assert state is not None, (
            "state is not found in the observation which is required for the behavior training."
        )

        return {
            "main_images": zed_image,  # [H, W, C]
            "wrist_images": torch.stack(
                [left_image, right_image], axis=0
            ),  # [N_IMG, H, W, C]
            "state": state,
        }

    @staticmethod
    def _parse_trunk_proprio_randomization(cfg):
        random_cfg = cfg.get("trunk_proprio_randomization", None)
        if random_cfg is None or not random_cfg.get("enabled", False):
            return None

        indices = list(random_cfg.get("indices", [236, 237, 238, 239]))
        low = list(random_cfg.get("low", [0.0, -0.45, 0.0, 0.0]))
        high = list(random_cfg.get("high", [0.7, 0.35, 0.18, 0.0]))
        fixed_values = list(random_cfg.get("fixed_values", [None, None, None, 0.0]))
        if not (len(indices) == len(low) == len(high) == len(fixed_values)):
            raise ValueError(
                "env.trunk_proprio_randomization indices, low, high, and "
                "fixed_values must have the same length."
            )
        return {
            "indices": [int(index) for index in indices],
            "low": [float(value) for value in low],
            "high": [float(value) for value in high],
            "fixed_values": [
                None if value is None else float(value) for value in fixed_values
            ],
        }

    def _resample_trunk_proprio_randomization(self, env_indices=None) -> None:
        if self.trunk_proprio_randomization is None:
            return

        dim = len(self.trunk_proprio_randomization["indices"])
        if self._trunk_proprio_random_values is None:
            self._trunk_proprio_random_values = torch.zeros(
                self.num_envs, dim, dtype=torch.float32
            )

        if env_indices is None:
            env_indices = list(range(self.num_envs))
        env_indices = [int(index) for index in env_indices]
        if not env_indices:
            return

        low = torch.tensor(
            self.trunk_proprio_randomization["low"], dtype=torch.float32
        )
        high = torch.tensor(
            self.trunk_proprio_randomization["high"], dtype=torch.float32
        )
        sampled = low + torch.rand(
            len(env_indices), dim, generator=self._trunk_proprio_rng
        ) * (high - low)
        for dim_idx, fixed_value in enumerate(
            self.trunk_proprio_randomization["fixed_values"]
        ):
            if fixed_value is not None:
                sampled[:, dim_idx] = float(fixed_value)
        self._trunk_proprio_random_values[env_indices] = sampled

    def _apply_trunk_proprio_randomization(self, states, env_indices=None):
        if self.trunk_proprio_randomization is None:
            return states

        if self._trunk_proprio_random_values is None:
            self._resample_trunk_proprio_randomization()

        if env_indices is None:
            env_indices = list(range(states.shape[0]))
        env_indices = [int(index) for index in env_indices]
        if not env_indices:
            return states

        states = states.clone()
        indices = self.trunk_proprio_randomization["indices"]
        if max(indices) >= states.shape[-1]:
            if states.shape[-1] < 7:
                return states
            indices = [3, 4, 5, 6]

        values = self._trunk_proprio_random_values[env_indices].to(
            device=states.device,
            dtype=states.dtype,
        )
        states[:, indices] = values
        return states

    def _update_stage_prompts_from_info(
        self, env_idx: int, info: dict | None
    ) -> str | None:
        if not isinstance(info, dict):
            return None

        stage_info = info
        reward_info = info.get("reward")
        if isinstance(reward_info, dict):
            task_specific_info = reward_info.get("task_specific")
            if isinstance(task_specific_info, dict):
                stage_info = task_specific_info

        replay_info = info.get("replay_init")
        if isinstance(replay_info, dict):
            replay_prompts = replay_info.get("replay_stage_prompts")
            if isinstance(replay_prompts, (list, tuple)):
                self._stage_prompt_lists[env_idx] = [
                    str(prompt).strip()
                    for prompt in replay_prompts
                    if str(prompt).strip()
                ]

        explicit_prompt = (
            stage_info.get("current_stage_prompt")
            or stage_info.get("stage_prompt")
            or stage_info.get("subtask_prompt")
        )
        if explicit_prompt:
            return str(explicit_prompt).strip()

        stage_idx = stage_info.get("current_stage_idx")
        if stage_idx is None:
            return None
        try:
            stage_idx = int(stage_idx)
        except (TypeError, ValueError):
            return None

        stage_prompts = self._stage_prompt_lists[env_idx]
        if not stage_prompts or stage_idx < 0 or stage_idx >= len(stage_prompts):
            return None
        return stage_prompts[stage_idx]

    def _compose_task_description(self, stage_prompt: str | None) -> str:
        if self.prompt_override is not None:
            return self.prompt_override
        if stage_prompt:
            return self.task_description + "\n" + stage_prompt
        return self.task_description

    def _task_descriptions_from_infos(self, infos=None) -> list[str]:
        if self.prompt_override is not None:
            return [self.prompt_override for _ in range(self.num_envs)]
        if not self.use_subtask_prompt or infos is None:
            return [self.task_description for _ in range(self.num_envs)]
        return [
            self._compose_task_description(
                self._update_stage_prompts_from_info(env_idx, info)
            )
            for env_idx, info in enumerate(infos)
        ]

    def _wrap_obs(self, obs_list, infos=None, env_indices=None):
        extracted_obs_list = []
        for obs in obs_list:
            extracted_obs = self._extract_obs_image(obs)
            extracted_obs_list.append(extracted_obs)

        states = torch.stack(
            [obs["state"] for obs in extracted_obs_list], axis=0
        )
        states = self._apply_trunk_proprio_randomization(states, env_indices)
        obs = {
            "main_images": torch.stack(
                [obs["main_images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, H, W, C]
            "wrist_images": torch.stack(
                [obs["wrist_images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, N_IMG, H, W, C]
            "task_descriptions": self._task_descriptions_from_infos(infos),
            "states": states,
        }
        return obs

    def _completion_bonus_tensor(self, infos, reward):
        bonuses = []
        for info in infos or [{} for _ in range(self.num_envs)]:
            reward_info = info.get("reward", {}) if isinstance(info, dict) else {}
            task_reward = reward_info.get("task_specific", {})
            bonuses.append(float(task_reward.get("completion_bonus", 0.0) or 0.0))
        return self.reward_coef * torch.as_tensor(
            bonuses, dtype=reward.dtype, device=reward.device
        )

    def _is_target_stage_success(self, task_reward: dict) -> bool:
        if self.success_stage_idx is None:
            return False
        current_stage_idx = task_reward.get("current_stage_idx", None)
        if current_stage_idx is None:
            return False
        try:
            current_stage_idx = int(current_stage_idx)
        except (TypeError, ValueError):
            return False
        completion_bonus = float(task_reward.get("completion_bonus", 0.0) or 0.0)
        return current_stage_idx == self.success_stage_idx and completion_bonus != 0.0

    def _stage_sparse_reward_tensor(self, rewards, infos=None):
        reward = torch.as_tensor(rewards)
        bonuses = []
        for info in infos or [{} for _ in range(int(reward.numel()))]:
            reward_info = info.get("reward", {}) if isinstance(info, dict) else {}
            task_reward = reward_info.get("task_specific", {})
            if not isinstance(task_reward, dict):
                task_reward = {}
            bonus = float(task_reward.get("completion_bonus", 0.0) or 0.0)
            bonuses.append(bonus if self._is_target_stage_success(task_reward) else 0.0)
        return self.reward_coef * torch.as_tensor(
            bonuses, dtype=reward.dtype, device=reward.device
        )

    def _stage_weighted_reward_tensor(self, rewards, infos=None):
        reward = torch.as_tensor(rewards)
        bonuses = []
        for info in infos or [{} for _ in range(int(reward.numel()))]:
            reward_info = info.get("reward", {}) if isinstance(info, dict) else {}
            task_reward = reward_info.get("task_specific", {})
            if not isinstance(task_reward, dict):
                task_reward = {}
            stage_idx = int(task_reward.get("current_stage_idx", 0) or 0) - 1
            bonus = float(task_reward.get("completion_bonus", 0.0) or 0.0)
            if 0 <= stage_idx < len(self.stage_reward_weights):
                bonus *= self.stage_reward_weights[stage_idx]
            else:
                bonus = 0.0
            bonuses.append(bonus)
        return self.reward_coef * torch.as_tensor(
            bonuses, dtype=reward.dtype, device=reward.device
        )

    def _stage_cumulative_reward_tensor(self, rewards, infos=None):
        reward = torch.as_tensor(rewards)
        counts = []
        for info in infos or [{} for _ in range(int(reward.numel()))]:
            reward_info = info.get("reward", {}) if isinstance(info, dict) else {}
            task_reward = reward_info.get("task_specific", {})
            if not isinstance(task_reward, dict):
                task_reward = {}
            counts.append(float(task_reward.get("completed_stage_count", 0.0) or 0.0))
        return self.reward_coef * torch.as_tensor(
            counts, dtype=reward.dtype, device=reward.device
        )

    def _calc_step_reward(self, rewards, infos=None):
        reward = self.reward_coef * rewards
        if self.stage_sparse_reward:
            return self._stage_sparse_reward_tensor(rewards, infos)
        if self.stage_weighted_reward:
            return self._stage_weighted_reward_tensor(rewards, infos)
        if self.stage_cumulative_reward:
            return self._stage_cumulative_reward_tensor(rewards, infos)

        if not self.use_rel_reward:
            return reward

        completion_bonus = self._completion_bonus_tensor(infos, reward)
        dense_reward = reward - completion_bonus
        reward_diff = dense_reward - self.prev_step_reward.to(dense_reward.device)
        self.prev_step_reward = dense_reward.to(self.prev_step_reward.device)
        return reward_diff + completion_bonus

    def _extract_episode_success(self, info: dict | None) -> bool:
        if not isinstance(info, dict):
            return False
        reward_info = info.get("reward", {})
        if isinstance(reward_info, dict):
            task_reward = reward_info.get("task_specific", {})
        else:
            task_reward = {}
        if self.success_stage_idx is not None:
            return self._is_target_stage_success(task_reward)
        done_dict = info.get("done", {})
        if isinstance(done_dict, dict):
            return bool(done_dict.get("success", False))
        return bool(info.get("success", False))

    def _extract_episode_done(self, info: dict | None) -> bool:
        if not isinstance(info, dict):
            return False
        if self.success_stage_idx is not None:
            return self._extract_episode_success(info)
        return self._extract_info_done(info)

    def _info_done_tensor(self, infos, device=None) -> torch.Tensor:
        done_flags = [self._extract_episode_done(info) for info in infos]
        return torch.as_tensor(done_flags, dtype=torch.bool, device=device)

    def _apply_info_dones(
        self,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        infos: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        info_dones = self._info_done_tensor(infos, device=terminations.device)
        if self.ignore_terminations:
            terminations = torch.zeros_like(terminations, dtype=torch.bool)
            truncations = torch.logical_or(truncations, info_dones)
        else:
            terminations = torch.logical_or(terminations, info_dones)
        dones = torch.logical_or(terminations, truncations)
        return terminations, truncations, dones

    def reset(self):
        if self.enable_offload and self.pool is None:
            self._init_env()
        raw_obs, infos = self.env_reset()
        self._resample_trunk_proprio_randomization()
        obs = self._wrap_obs(raw_obs, infos)
        rewards = torch.zeros(self.num_envs, dtype=bool)
        infos = self._record_metrics(rewards, infos)
        self._reset_metrics()
        return obs, infos

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim].
        chunk_actions = torch.as_tensor(chunk_actions).detach().cpu()
        (
            raw_obs_list,
            raw_rewards_list,
            raw_terminations_list,
            raw_truncations_list,
            raw_infos_list,
        ) = self.env_chunk_step(chunk_actions)

        obs_list = []
        infos_list = []
        scaled_rewards_list = []
        processed_terminations_list = []
        processed_truncations_list = []
        for raw_obs, raw_rewards, raw_terminations, raw_truncations, step_infos in zip(
            raw_obs_list,
            raw_rewards_list,
            raw_terminations_list,
            raw_truncations_list,
            raw_infos_list,
        ):
            if raw_obs is None or (
                isinstance(raw_obs, (list, tuple))
                and all(obs is None for obs in raw_obs)
            ):
                obs_list.append(None)
            else:
                obs_list.append(self._wrap_obs(raw_obs, step_infos))

            step_rewards = self._calc_step_reward(raw_rewards, step_infos)
            raw_terminations = raw_terminations.bool()
            raw_truncations = raw_truncations.bool()
            raw_terminations, raw_truncations, step_dones = self._apply_info_dones(
                raw_terminations, raw_truncations, step_infos
            )
            infos_list.append(
                self._record_metrics(step_rewards, step_infos, dones=step_dones)
            )
            scaled_rewards_list.append(step_rewards)
            processed_terminations_list.append(raw_terminations)
            processed_truncations_list.append(raw_truncations)

        chunk_rewards = torch.stack(
            scaled_rewards_list, dim=1
        )  # [num_envs, chunk_steps]
        raw_terminations = torch.stack(
            processed_terminations_list, dim=1
        )  # [num_envs, chunk_steps]
        raw_truncations = torch.stack(
            processed_truncations_list, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_terminations.any(dim=1)
        past_truncations = raw_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros_like(raw_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_truncations)
        chunk_truncations[:, -1] = past_truncations
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    @property
    def device(self):
        return "cuda"

    @property
    def elapsed_steps(self):
        return self.max_episode_steps

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
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.prev_step_reward = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if not self.record_metrics:
            return
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
        else:
            mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        self.prev_step_reward[mask] = 0.0
        self.success_once[mask] = False
        self.returns[mask] = 0

    def _record_metrics(self, rewards, infos, dones=None, env_indices=None):
        info_lists = []
        replay_info_lists = []
        if env_indices is None:
            env_indices = list(range(len(infos)))
        for local_idx, (env_idx, reward, info) in enumerate(
            zip(env_indices, rewards, infos)
        ):
            reward_info = info.get("reward", {}) if isinstance(info, dict) else {}
            if isinstance(reward_info, dict):
                task_reward = reward_info.get("task_specific", {})
            else:
                task_reward = {}
            completion_bonus = float(task_reward.get("completion_bonus", 0.0) or 0.0)
            step_success = self._extract_episode_success(info)
            end_success = (
                step_success
                if self.success_stage_idx is not None
                else info.get("success", step_success)
            )
            episode_length = info.get("episode_length", 0)
            current_stage_idx = task_reward.get("current_stage_idx", -1)
            total_stage_count = task_reward.get("total_stage_count", 0)
            current_stage_success = completion_bonus != 0.0
            activity_instance_id = task_reward.get("activity_instance_id", -1)
            held_in_hand = task_reward.get("held_in_hand", False)
            episode_done = (
                bool(dones[local_idx].item())
                if isinstance(dones, torch.Tensor)
                else self._extract_episode_done(info)
            )
            episode_info = {
                "episode_length": episode_length,
                "completion_bonus": completion_bonus,
                "current_stage_idx": int(
                    current_stage_idx if current_stage_idx is not None else -1
                ),
                "total_stage_count": int(
                    total_stage_count if total_stage_count is not None else 0
                ),
                "current_stage_success": current_stage_success,
                "success_stage_idx": (
                    -1 if self.success_stage_idx is None else self.success_stage_idx
                ),
                "target_stage_success": step_success,
                "done": episode_done,
                "activity_instance_id": int(
                    activity_instance_id
                    if activity_instance_id is not None
                    else -1
                ),
                "held_in_hand_at_end": bool(held_in_hand),
            }
            for metric_key in (
                "held_in_hand_available",
                "all_stages_completed",
                "completed_stage_count",
            ):
                metric_value = task_reward.get(metric_key)
                if isinstance(metric_value, (bool, int, float)):
                    episode_info[metric_key] = metric_value
            self.returns[env_idx] += reward
            self.success_once[env_idx] = self.success_once[env_idx] | step_success
            episode_info["success_once"] = self.success_once[env_idx].clone()
            episode_info["success_at_end"] = end_success
            episode_info["return"] = self.returns[env_idx].clone()
            episode_info["episode_len"] = episode_length
            episode_info["reward"] = episode_info["return"] / torch.clamp(
                to_tensor(episode_length), min=1
            ).to(self.device)

            info_lists.append(episode_info)
            if "replay_init" in info:
                replay_info_lists.append(
                    {
                        key: value
                        for key, value in info["replay_init"].items()
                        if isinstance(value, (bool, int, float))
                    }
                )

        infos = {"episode": to_tensor(list_of_dict_to_dict_of_list(info_lists))}
        if replay_info_lists:
            infos["replay_init"] = to_tensor(
                list_of_dict_to_dict_of_list(replay_info_lists)
            )
        return infos

    @staticmethod
    def _extract_info_done(info: dict) -> bool:
        tc = info["done"]["termination_conditions"]
        return any(v["done"] for v in tc.values())

    @staticmethod
    def _clone_obs(obs):
        cloned = {}
        for key, value in obs.items():
            if torch.is_tensor(value):
                cloned[key] = value.clone()
            elif isinstance(value, list):
                cloned[key] = list(value)
            else:
                cloned[key] = copy.deepcopy(value)
        return cloned

    @staticmethod
    def _merge_obs_rows(base_obs, update_obs, env_indices):
        merged_obs = BehaviorEnv._clone_obs(base_obs)
        for key, update_value in update_obs.items():
            if key not in merged_obs:
                merged_obs[key] = update_value
                continue
            base_value = merged_obs[key]
            if torch.is_tensor(base_value):
                index = torch.as_tensor(
                    env_indices, device=base_value.device, dtype=torch.long
                )
                merged_obs[key][index] = update_value.to(base_value.device)
            elif isinstance(base_value, list):
                for local_idx, env_idx in enumerate(env_indices):
                    base_value[env_idx] = update_value[local_idx]
            else:
                merged_obs[key] = update_value
        return merged_obs

    def _merge_info_rows(self, base_infos, update_infos, env_indices):
        merged_infos = copy.deepcopy(base_infos)

        def merge_value(base_value, update_value):
            if isinstance(update_value, dict):
                if not isinstance(base_value, dict):
                    base_value = {}
                for key, child_update in update_value.items():
                    base_value[key] = merge_value(base_value.get(key), child_update)
                return base_value

            if torch.is_tensor(update_value):
                if not torch.is_tensor(base_value):
                    base_shape = (self.num_envs, *update_value.shape[1:])
                    base_value = torch.zeros(
                        base_shape, dtype=update_value.dtype, device=update_value.device
                    )
                index = torch.as_tensor(
                    env_indices, device=base_value.device, dtype=torch.long
                )
                base_value[index] = update_value.to(base_value.device)
                return base_value

            return update_value

        for key, value in update_infos.items():
            merged_infos[key] = merge_value(merged_infos.get(key), value)
        return merged_infos

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        reset_indices = dones.nonzero(as_tuple=False).flatten().tolist()
        final_obs = self._clone_obs(extracted_obs)
        final_info = copy.deepcopy(infos)

        reset_indices, reset_raw_obs, reset_raw_infos = self.env_reset_partial(
            dones.tolist()
        )
        if reset_indices:
            self._resample_trunk_proprio_randomization(reset_indices)
            reset_obs = self._wrap_obs(
                reset_raw_obs,
                reset_raw_infos,
                env_indices=reset_indices,
            )
            extracted_obs = self._merge_obs_rows(
                extracted_obs, reset_obs, reset_indices
            )

            self._reset_metrics(reset_indices)
            reset_rewards = torch.zeros(
                len(reset_indices), device=self.device, dtype=torch.float32
            )
            reset_infos = self._record_metrics(
                reset_rewards, reset_raw_infos, env_indices=reset_indices
            )
            infos = self._merge_info_rows(infos, reset_infos, reset_indices)

        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def update_reset_state_ids(self):
        if self._ordered_reset_instance_ids is not None:
            self._ordered_reset_epoch += 1

    def _init_ordered_reset_instance_ids(self) -> list[int] | None:
        if not (self.is_eval and self.use_fixed_reset_state_ids):
            return None

        from rlinf.envs.behavior.instance_loader import parse_activity_instance_ids

        instance_ids = parse_activity_instance_ids(
            OmegaConf.select(self.cfg, "omni_config.task.activity_instance_id")
        )
        if not instance_ids:
            return None
        return [int(instance_id) for instance_id in instance_ids]

    def _ordered_reset_ids_for_indices(
        self, reset_indices: list[int]
    ) -> list[int] | None:
        if self._ordered_reset_instance_ids is None:
            return None

        global_env_count = self.num_envs * self.total_num_processes
        base_index = (
            self._ordered_reset_epoch * global_env_count
            + self.seed_offset * self.num_envs
        )
        ordered_ids = self._ordered_reset_instance_ids
        return [
            ordered_ids[(base_index + int(env_idx)) % len(ordered_ids)]
            for env_idx in reset_indices
        ]

    @staticmethod
    def _reset_payload_with_instance_ids(
        reset_mask: list[bool],
        instance_ids: list[int] | None,
        full_reset: bool = False,
    ):
        if instance_ids is None:
            return reset_mask

        instance_iter = iter(instance_ids)
        payload = []
        for should_reset in reset_mask:
            item = {"reset": bool(should_reset), "full_reset": full_reset}
            if should_reset:
                item["instance_id"] = next(instance_iter)
            payload.append(item)
        return payload

    def dump_replay_tro_states(self, payload: dict) -> list[dict]:
        """Dispatch ``dump_replay_tro_states`` to every Ray actor in the pool."""
        self._ensure_pool()
        results = ray.get([
            proc.dump_replay_tro_states.remote(payload)
            for proc in self.pool.env_processes
        ])
        merged = []
        for sub_results in results:
            merged.extend(sub_results)
        return merged

    def offload(self):
        self.close()

    def close(self):
        if self.pool:
            BehaviorProcessPool.release_shared()
            self.pool = None
            self.pool_offset = None
