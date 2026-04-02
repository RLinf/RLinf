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

"""LingBot-VA adapter for RLinf embodied rollout."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.lingbotva.history_buffer import (
    LingbotVAEpisodeState,
    select_key_frames,
)
from rlinf.models.embodiment.lingbotva.native_backend import LingbotVANativeBackend
from rlinf.models.embodiment.lingbotva.observation_adapter import (
    LingbotVAObservationAdapter,
)


class LingbotVAActionModel(nn.Module, BasePolicy):
    """RLinf action model wrapper for official single-session LingBot-VA eval."""

    def __init__(self, cfg: Any, torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = cfg
        self.torch_dtype = torch_dtype
        self.num_action_chunks = int(getattr(cfg, "num_action_chunks", 32))
        self.action_dim = int(getattr(cfg, "action_dim", 16))
        self.action_env_dim = int(getattr(cfg, "action_env_dim", self.action_dim))
        self._backend: LingbotVANativeBackend | None = None
        # Runtime guards keep LingBot-VA single-session today; the dict is retained
        # to keep the existing reset/refill helpers simple around env_idx.
        self._episode_states: dict[int, LingbotVAEpisodeState] = {}
        self.action_per_frame = int(getattr(cfg.lingbotva, "action_per_frame", 16))

    @staticmethod
    def _get_extra_obs(env_obs: dict[str, Any]) -> dict[str, Any]:
        extra_obs = env_obs.get("extra_obs")
        if isinstance(extra_obs, dict):
            return extra_obs
        if isinstance(extra_obs, (list, tuple)):
            merged: dict[str, Any] = {}
            for item in extra_obs:
                if not isinstance(item, dict):
                    continue
                for key, value in item.items():
                    if key not in merged:
                        merged[key] = value
                        continue
                    existing = merged[key]
                    if isinstance(existing, list):
                        if isinstance(value, list):
                            existing.extend(value)
                        else:
                            existing.append(value)
                        continue
                    if isinstance(existing, tuple):
                        if isinstance(value, tuple):
                            merged[key] = existing + value
                        else:
                            merged[key] = existing + (value,)
                        continue
                    if torch.is_tensor(existing):
                        if torch.is_tensor(value):
                            try:
                                merged[key] = torch.cat([existing, value], dim=0)
                            except Exception:
                                merged[key] = value
                        else:
                            merged[key] = value
                        continue
                    merged[key] = value
            return merged
        return {}

    def _get_env_meta(self, env_obs: dict[str, Any], key: str) -> Any:
        extra_obs = self._get_extra_obs(env_obs)
        if key in extra_obs:
            return extra_obs[key]
        return env_obs.get(key)

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "LingBot-VA currently supports rollout/evaluation via predict_action_batch only."
        )

    def _ensure_backend(self) -> LingbotVANativeBackend:
        if self._backend is None:
            self._backend = LingbotVANativeBackend(self.config, self.torch_dtype)
        return self._backend

    def _get_prompt(self, env_obs: dict[str, Any], env_idx: int) -> str:
        prompts = env_obs.get("task_descriptions")
        if prompts is None:
            raise ValueError(
                "LingBot-VA requires task_descriptions in env observations."
            )
        return str(prompts[env_idx])

    def _get_state(self, env_idx: int) -> LingbotVAEpisodeState:
        if env_idx not in self._episode_states:
            self._episode_states[env_idx] = LingbotVAEpisodeState()
        return self._episode_states[env_idx]

    def _format_initial_eef_pose(
        self, env_obs: dict[str, Any], env_idx: int
    ) -> np.ndarray | None:
        eef_poses = self._get_env_meta(env_obs, "eef_poses")
        if eef_poses is None:
            return None
        if isinstance(eef_poses, torch.Tensor):
            pose = eef_poses[env_idx].detach().cpu().numpy().astype(np.float32)
        else:
            pose = np.asarray(eef_poses[env_idx], dtype=np.float32)
        if pose.shape[-1] != 16:
            raise ValueError(f"Expected 16D reset ee pose, got shape {pose.shape}.")
        return pose.copy()

    @staticmethod
    def _add_eef_pose(delta_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
        delta_rot = R.from_quat(delta_pose[3:7][None])
        init_rot = R.from_quat(init_pose[3:7][None])
        out_rot = (init_rot * delta_rot).as_quat().reshape(-1)
        out_trans = delta_pose[:3] + init_pose[:3]
        return np.concatenate([out_trans, out_rot, delta_pose[7:8]], axis=0)

    def _add_init_pose(
        self, delta_action: np.ndarray, initial_pose: np.ndarray
    ) -> np.ndarray:
        left = self._add_eef_pose(delta_action[:8], initial_pose[:8])
        right = self._add_eef_pose(delta_action[8:], initial_pose[8:])
        out = np.concatenate([left, right], axis=0).astype(np.float32)
        out[3:7] = out[3:7] / np.linalg.norm(out[3:7])
        out[11:15] = out[11:15] / np.linalg.norm(out[11:15])
        return out

    def _flatten_raw_action(self, raw_action: np.ndarray) -> np.ndarray:
        return np.transpose(raw_action, (1, 2, 0)).reshape(-1, raw_action.shape[0])

    def _select_executable_actions(
        self, raw_action: np.ndarray, first_chunk: bool
    ) -> np.ndarray:
        start_idx = 1 if first_chunk else 0
        selected = raw_action[:, start_idx:, :]
        return np.transpose(selected, (1, 2, 0)).reshape(-1, raw_action.shape[0])

    def _convert_raw_to_env_actions(
        self,
        raw_action: np.ndarray,
        initial_eef_pose: np.ndarray,
        first_chunk: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        flattened_model = self._flatten_raw_action(raw_action)
        executable_model = self._select_executable_actions(raw_action, first_chunk)
        env_actions = np.stack(
            [self._add_init_pose(step, initial_eef_pose) for step in executable_model],
            axis=0,
        )
        return flattened_model, env_actions

    def _reset_episode(
        self, env_idx: int, prompt: str, env_obs: dict[str, Any]
    ) -> LingbotVAEpisodeState:
        state = self._get_state(env_idx)
        initial_eef_pose = self._format_initial_eef_pose(env_obs, env_idx)
        if initial_eef_pose is None:
            raise ValueError(
                "LingBot-VA requires reset-time eef_poses from RoboTwinEnv."
            )
        state.reset(prompt=prompt, initial_eef_pose=initial_eef_pose)
        self._ensure_backend().reset(prompt)
        return state

    def _refill_action_queue(
        self, env_idx: int, env_obs: dict[str, Any], state: LingbotVAEpisodeState
    ) -> None:
        prompt = state.prompt or self._get_prompt(env_obs, env_idx)
        backend = self._ensure_backend()
        if state.first_chunk:
            first_obs = LingbotVAObservationAdapter.format_observation(
                env_obs, env_idx, prompt
            )
            state.first_obs = first_obs
            raw_action = backend.infer(first_obs, prompt)
        else:
            action_per_frame = state.last_action_per_frame or self.action_per_frame
            key_frames = select_key_frames(
                chunk_observations=self._get_env_meta(env_obs, "chunk_observations"),
                env_idx=env_idx,
                prompt=prompt,
                action_per_frame=action_per_frame,
            )
            if not key_frames:
                raise ValueError(
                    "LingBot-VA follow-up chunk requires non-empty key_frame_list."
                )
            if state.first_obs is None:
                raise ValueError(
                    "LingBot-VA follow-up chunk requires cached first_obs from the first infer."
                )
            backend.compute_kv_cache(key_frames, state.prev_model_action)
            raw_action = backend.infer(state.first_obs, prompt)

        state.prev_model_action = raw_action.astype(np.float32)
        state.last_action_per_frame = max(1, raw_action.shape[2] // 4)
        _, env_actions = self._convert_raw_to_env_actions(
            raw_action=raw_action,
            initial_eef_pose=state.initial_eef_pose,
            first_chunk=state.first_chunk,
        )
        state.action_queue.clear()
        for action in env_actions:
            state.action_queue.append(action.astype(np.float32))
        state.first_chunk = False
        state.chunk_id += 1

    def predict_action_batch(
        self, env_obs: dict[str, Any], mode: str = "eval", **_: Any
    ):
        del mode
        states_tensor = env_obs.get("states")
        if states_tensor is None:
            raise ValueError("LingBot-VA requires batched states in env observations.")
        batch_size = states_tensor.shape[0]
        if batch_size != 1:
            raise ValueError(
                "Official LingBot-VA websocket backend is stateful and currently "
                "only supports single-session evaluation with batch_size == 1."
            )
        actions = []
        for env_idx in range(batch_size):
            prompt = self._get_prompt(env_obs, env_idx)
            episode_done = False
            episode_dones = self._get_env_meta(env_obs, "episode_dones")
            if isinstance(episode_dones, torch.Tensor):
                if episode_dones.dim() == 1:
                    episode_done = bool(episode_dones[env_idx].item())
                else:
                    episode_done = bool(episode_dones[env_idx, -1].item())
            state = self._get_state(env_idx)
            if state.prompt != prompt or state.prompt is None or episode_done:
                state = self._reset_episode(env_idx, prompt, env_obs)
            if not state.action_queue:
                self._refill_action_queue(env_idx, env_obs, state)
            if not state.action_queue:
                raise RuntimeError("LingBot-VA refill produced an empty action queue.")
            chunk_actions = np.stack(list(state.action_queue), axis=0)
            state.action_queue.clear()
            actions.append(torch.from_numpy(chunk_actions))

        chunk_lengths = [action.shape[0] for action in actions]
        if len(set(chunk_lengths)) != 1:
            raise RuntimeError(
                "LingBot-VA chunk lengths must match across batch: "
                + str(chunk_lengths)
            )

        action_tensor = torch.stack(actions, dim=0).to(dtype=torch.float32)
        result = {
            "prev_logprobs": torch.zeros(action_tensor.shape[:2], dtype=torch.float32),
            "prev_values": torch.zeros(action_tensor.shape[:2], dtype=torch.float32),
            "forward_inputs": {"action": action_tensor},
        }
        return action_tensor, result
