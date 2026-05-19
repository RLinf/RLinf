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

"""LingBot-VA adapter for RLinf embodied rollout and SFT."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.lingbotva.history_buffer import (
    LingbotVAEpisodeState,
    select_key_frames,
)
from rlinf.models.embodiment.lingbotva.native_backend import LingbotVANativeBackend
from rlinf.models.embodiment.lingbotva.observation_adapter import (
    LingbotVAObservationAdapter,
)
from rlinf.models.embodiment.lingbotva.sft_core import LingbotVASFTCore


class LingbotVAActionModel(nn.Module, BasePolicy):
    def __init__(self, cfg: Any, torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = cfg
        self.torch_dtype = torch_dtype
        self.num_action_chunks = int(getattr(cfg, "num_action_chunks", 32))
        self.action_dim = int(getattr(cfg, "action_dim", 16))
        self.action_per_frame = int(getattr(cfg.lingbotva, "action_per_frame", 16))

        self._backend: LingbotVANativeBackend | None = None
        self._episode_states: dict[int, LingbotVAEpisodeState] = {}
        self._sft_core: LingbotVASFTCore | None = None
        if bool(getattr(cfg.lingbotva, "enable_sft", False)):
            self._sft_core = LingbotVASFTCore(cfg=cfg, torch_dtype=torch_dtype)

    @staticmethod
    def _get_extra_obs(env_obs: dict[str, Any]) -> dict[str, Any]:
        extra_obs = env_obs.get("extra_obs")
        if isinstance(extra_obs, dict):
            return extra_obs
        if isinstance(extra_obs, (list, tuple)):
            merged: dict[str, Any] = {}
            for item in extra_obs:
                if item is None:
                    continue
                if not isinstance(item, dict):
                    raise TypeError(
                        "LingBot-VA expects extra_obs shards to be dict or None, "
                        f"got {type(item)!r}."
                    )
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
                    elif isinstance(existing, tuple):
                        if isinstance(value, tuple):
                            merged[key] = existing + value
                        else:
                            merged[key] = existing + (value,)
                    elif torch.is_tensor(existing) and torch.is_tensor(value):
                        try:
                            merged[key] = torch.cat([existing, value], dim=0)
                        except RuntimeError as exc:
                            raise ValueError(
                                "LingBot-VA extra_obs tensor shards must be concatenable "
                                f"for key {key!r}: existing shape {tuple(existing.shape)} "
                                f"vs new shape {tuple(value.shape)}."
                            ) from exc
                    else:
                        merged[key] = value
            return merged
        if extra_obs is not None:
            raise TypeError(
                "LingBot-VA expects extra_obs to be a dict, list/tuple of dict shards, "
                f"or None, got {type(extra_obs)!r}."
            )
        return {}

    def _get_env_meta(self, env_obs: dict[str, Any], key: str) -> Any:
        extra_obs = self._get_extra_obs(env_obs)
        if key in extra_obs:
            return extra_obs[key]
        return env_obs.get(key)

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        if self._sft_core is not None:
            self._sft_core.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        if self._sft_core is not None:
            self._sft_core.gradient_checkpointing_disable()

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def sft_forward(self, data, **kwargs):
        del kwargs
        if self._sft_core is None:
            raise NotImplementedError(
                "LingBot-VA SFT support is disabled for the current config."
            )
        return self._sft_core(data)

    def default_forward(
        self,
        **kwargs,
    ):
        del kwargs
        raise NotImplementedError(
            "LingBot-VA default_forward is not supported in the current eval/SFT integration. "
            "Use predict_action_batch for evaluation and sft_forward for supervised fine-tuning."
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
    ) -> np.ndarray:
        executable_model = self._select_executable_actions(raw_action, first_chunk)
        env_actions = np.stack(
            [self._add_init_pose(step, initial_eef_pose) for step in executable_model],
            axis=0,
        )
        return env_actions

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
        return state

    @staticmethod
    def _group_env_indices_for_batch_refill(
        env_indices: list[int],
        episode_states: dict[int, LingbotVAEpisodeState],
    ) -> list[list[int]]:
        grouped: dict[tuple[bool, int], list[int]] = {}
        for env_idx in env_indices:
            state = episode_states[env_idx]
            group_key = (
                state.first_chunk,
                len(state.kv_cache_history),
            )
            grouped.setdefault(group_key, []).append(env_idx)
        return list(grouped.values())

    def _fill_action_queue_from_raw_action(
        self,
        state: LingbotVAEpisodeState,
        raw_action: np.ndarray,
    ) -> None:
        if raw_action.ndim != 3:
            raise ValueError(
                f"LingBot-VA raw action must be 3D, got shape {tuple(raw_action.shape)}."
            )
        if raw_action.shape[2] % 4 != 0:
            raise ValueError(
                "LingBot-VA follow-up key-frame cadence requires raw_action.shape[2] "
                f"to be divisible by 4, got {raw_action.shape[2]}."
            )
        state.prev_model_action = raw_action.astype(np.float32)
        state.last_action_per_frame = raw_action.shape[2] // 4
        env_actions = self._convert_raw_to_env_actions(
            raw_action=raw_action,
            initial_eef_pose=state.initial_eef_pose,
            first_chunk=state.first_chunk,
        )
        state.action_queue.clear()
        for action in env_actions:
            state.action_queue.append(action.astype(np.float32))
        state.first_chunk = False

    def _pop_action_chunk(
        self,
        state: LingbotVAEpisodeState,
        chunk_len: int,
    ) -> np.ndarray:
        if chunk_len <= 0:
            raise ValueError(f"chunk_len must be positive, got {chunk_len}.")
        if len(state.action_queue) < chunk_len:
            raise RuntimeError(
                "LingBot-VA action queue does not have enough buffered steps: "
                f"env_queue={len(state.action_queue)}, requested={chunk_len}."
            )
        env_actions = np.stack(
            [state.action_queue.popleft() for _ in range(chunk_len)], axis=0
        ).astype(np.float32)
        return env_actions

    @staticmethod
    def _history_signature(
        history: list[tuple[list[dict[str, Any]], np.ndarray]],
    ) -> tuple[int, ...]:
        return tuple(len(obs_seq) for obs_seq, _state in history)

    def _refill_action_queue_batch(
        self,
        env_indices: list[int],
        env_obs: dict[str, Any],
    ) -> None:
        backend = self._ensure_backend()
        prompts = [self._get_prompt(env_obs, env_idx) for env_idx in env_indices]
        states = [self._get_state(env_idx) for env_idx in env_indices]

        if all(state.first_chunk for state in states):
            obs_batch = []
            for env_idx, prompt, state in zip(env_indices, prompts, states):
                first_obs = LingbotVAObservationAdapter.format_observation(
                    env_obs, env_idx, prompt
                )
                state.first_obs = first_obs
                obs_batch.append(first_obs)
            raw_actions = backend.infer_batch(obs_batch, prompts)
        else:
            followup_groups: dict[
                tuple[int, ...],
                dict[str, list[Any]],
            ] = {}
            for env_idx, prompt, state in zip(env_indices, prompts, states):
                raw_chunk_observations = self._get_env_meta(
                    env_obs, "chunk_observations"
                )
                if (
                    raw_chunk_observations is None
                    or env_idx >= len(raw_chunk_observations)
                    or not raw_chunk_observations[env_idx]
                ):
                    raise ValueError(
                        "LingBot-VA follow-up chunk requires non-empty raw chunk observations."
                    )
                if state.first_obs is None:
                    raise ValueError(
                        "LingBot-VA follow-up chunk requires cached first_obs from the first infer."
                    )
                if state.prev_model_action is None:
                    raise ValueError(
                        "LingBot-VA follow-up chunk requires cached previous model action."
                    )
                action_per_frame = state.last_action_per_frame or self.action_per_frame
                key_frames = select_key_frames(
                    chunk_observations=raw_chunk_observations,
                    env_idx=env_idx,
                    prompt=prompt,
                    action_per_frame=action_per_frame,
                )
                if not key_frames:
                    raise ValueError(
                        "LingBot-VA follow-up chunk requires non-empty key_frame_list."
                    )
                history_entry = (key_frames, state.prev_model_action.copy())
                full_history = state.kv_cache_history + [history_entry]
                signature = self._history_signature(full_history)
                group = followup_groups.setdefault(
                    signature,
                    {
                        "obs_batch": [],
                        "prompts": [],
                        "history_batch": [],
                        "states": [],
                        "pending_entries": [],
                    },
                )
                group["obs_batch"].append(state.first_obs)
                group["prompts"].append(prompt)
                group["history_batch"].append(full_history)
                group["states"].append(state)
                group["pending_entries"].append(history_entry)

            for group in followup_groups.values():
                raw_actions = backend.infer_batch(
                    group["obs_batch"],
                    group["prompts"],
                    kv_cache_histories=group["history_batch"],
                )
                for state, history_entry, raw_action in zip(
                    group["states"],
                    group["pending_entries"],
                    raw_actions,
                ):
                    state.kv_cache_history.append(history_entry)
                    self._fill_action_queue_from_raw_action(state, raw_action)
            return

        for state, raw_action in zip(states, raw_actions):
            self._fill_action_queue_from_raw_action(state, raw_action)

    def predict_action_batch(
        self, env_obs: dict[str, Any], mode: str = "eval", **_: Any
    ):
        if mode != "eval":
            raise NotImplementedError(
                "LingBot-VA predict_action_batch only supports eval mode in the current integration."
            )
        states_tensor = env_obs.get("states")
        if states_tensor is None:
            raise ValueError("LingBot-VA requires batched states in env observations.")
        batch_size = states_tensor.shape[0]

        actions = []
        episode_dones = self._get_env_meta(env_obs, "episode_dones")
        pending_refill_envs: list[int] = []
        for env_idx in range(batch_size):
            prompt = self._get_prompt(env_obs, env_idx)
            episode_done = False
            if isinstance(episode_dones, torch.Tensor):
                if episode_dones.dim() == 1:
                    episode_done = bool(episode_dones[env_idx].item())
                else:
                    episode_done = bool(episode_dones[env_idx, -1].item())

            state = self._get_state(env_idx)
            if state.prompt != prompt or state.prompt is None or episode_done:
                state = self._reset_episode(env_idx, prompt, env_obs)
            if not state.action_queue:
                pending_refill_envs.append(env_idx)

        for refill_group in self._group_env_indices_for_batch_refill(
            pending_refill_envs, self._episode_states
        ):
            self._refill_action_queue_batch(refill_group, env_obs)

        available_chunk_lengths = []
        for env_idx in range(batch_size):
            state = self._get_state(env_idx)
            if not state.action_queue:
                raise RuntimeError("LingBot-VA refill produced an empty action queue.")
            available_chunk_lengths.append(len(state.action_queue))

        common_chunk_len = min(available_chunk_lengths)
        if common_chunk_len <= 0:
            raise RuntimeError(
                "LingBot-VA could not determine a positive shared chunk length: "
                + str(available_chunk_lengths)
            )

        for env_idx in range(batch_size):
            state = self._get_state(env_idx)
            chunk_actions = self._pop_action_chunk(state, common_chunk_len)
            actions.append(torch.from_numpy(chunk_actions))

        action_tensor = torch.stack(actions, dim=0).to(dtype=torch.float32)
        zeros = torch.zeros(action_tensor.shape[:2], dtype=torch.float32)
        result = {
            "prev_logprobs": zeros,
            "prev_values": zeros,
            "forward_inputs": {"action": action_tensor},
        }
        return action_tensor, result
