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

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

LOCAL_CRITIC_PREFIX = "critic."
RLINF_VALUE_HEAD_PREFIX = "value_head.mlp."

LOCAL_EXPECTED_SHAPES = {
    "0.weight": lambda obs_dim: (256, obs_dim),
    "0.bias": lambda obs_dim: (256,),
    "2.weight": lambda obs_dim: (256, 256),
    "2.bias": lambda obs_dim: (256,),
    "4.weight": lambda obs_dim: (256, 256),
    "4.bias": lambda obs_dim: (256,),
    "6.weight": lambda obs_dim: (1, 256),
    "6.bias": lambda obs_dim: (1,),
}

RLINF_EXPECTED_SHAPES = {
    "0.weight": lambda obs_dim: (256, obs_dim),
    "0.bias": lambda obs_dim: (256,),
    "2.weight": lambda obs_dim: (256, 256),
    "2.bias": lambda obs_dim: (256,),
    "4.weight": lambda obs_dim: (256, 256),
    "4.bias": lambda obs_dim: (256,),
    "6.weight": lambda obs_dim: (1, 256),
}


@dataclass(frozen=True)
class LoadedValueModel:
    model: nn.Module
    obs_dim: int
    checkpoint_format: str
    final_bias: bool


@dataclass(frozen=True)
class OracleDeltaGAEConfig:
    value_checkpoint_path: str
    gamma: float
    gae_lambda: float
    label_mode: str
    delta_threshold: float
    positive_reward: float
    unchanged_reward: float
    negative_reward: float
    device: str = "cpu"


class MLPValueModel(nn.Module):
    def __init__(self, obs_dim: int, *, final_bias: bool):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1, bias=final_bias),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class LocalPPOCritic(MLPValueModel):
    def __init__(self, obs_dim: int):
        super().__init__(obs_dim=obs_dim, final_bias=True)


class RLinfMLPValueHead(MLPValueModel):
    def __init__(self, obs_dim: int):
        super().__init__(obs_dim=obs_dim, final_bias=False)


def resolve_device(device: str) -> str:
    if not device.startswith("cuda"):
        return device
    if not torch.cuda.is_available():
        return "cpu"
    if ":" not in device:
        return device
    try:
        index = int(device.split(":", maxsplit=1)[1])
    except ValueError:
        return device
    if index < torch.cuda.device_count():
        return device
    return "cuda:0"


def _unwrap_state_dict(state: object, path: str) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict):
        raise ValueError(
            f"Expected checkpoint {path} to be a dict-like object, got {type(state)}"
        )
    if state and all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        return state
    for candidate in ("state_dict", "model_state_dict"):
        nested = state.get(candidate)
        if (
            isinstance(nested, dict)
            and nested
            and all(
                isinstance(key, str) and torch.is_tensor(value)
                for key, value in nested.items()
            )
        ):
            return nested
    raise ValueError(
        f"Checkpoint {path} does not contain a directly usable state_dict of tensors"
    )


def load_raw_checkpoint(path: str) -> dict[str, torch.Tensor]:
    return _unwrap_state_dict(torch.load(path, map_location="cpu"), path)


def _extract_prefixed_state_dict(
    state_dict: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {
        key.split(prefix, maxsplit=1)[1]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _validate_shapes(
    state_dict: dict[str, torch.Tensor],
    *,
    obs_dim: int,
    expected_shapes: dict[str, callable],
    checkpoint_label: str,
) -> None:
    expected_keys = set(expected_shapes)
    actual_keys = set(state_dict)
    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    if missing_keys or unexpected_keys:
        raise ValueError(
            f"Checkpoint does not match the expected {checkpoint_label} architecture. "
            f"Missing keys: {missing_keys}; unexpected keys: {unexpected_keys}"
        )

    shape_errors = []
    for key, shape_fn in expected_shapes.items():
        expected_shape = shape_fn(obs_dim)
        actual_shape = tuple(state_dict[key].shape)
        if actual_shape != expected_shape:
            shape_errors.append(f"{key}: expected {expected_shape}, got {actual_shape}")
    if shape_errors:
        raise ValueError(
            f"Checkpoint {checkpoint_label} shapes do not match the supported "
            f"architecture: {shape_errors}"
        )


def _detect_checkpoint_format(state_dict: dict[str, torch.Tensor]) -> str:
    has_local = any(key.startswith(LOCAL_CRITIC_PREFIX) for key in state_dict)
    has_rlinf = any(key.startswith(RLINF_VALUE_HEAD_PREFIX) for key in state_dict)
    if has_local and has_rlinf:
        return "local_ppo"
    if has_local:
        return "local_ppo"
    if has_rlinf:
        return "rlinf_mlp_value_head"
    raise ValueError(
        "Unsupported checkpoint format. Expected either local ManiSkill PPO critic.* "
        "parameters or RLinf value_head.mlp.* parameters."
    )


def infer_local_ppo_obs_dim(state_dict: dict[str, torch.Tensor]) -> int:
    weight = state_dict.get(f"{LOCAL_CRITIC_PREFIX}0.weight")
    if weight is None:
        raise ValueError(
            "Checkpoint is missing critic.0.weight. Only local ManiSkill PPO state "
            "checkpoints with critic.* parameters are supported."
        )
    if weight.ndim != 2:
        raise ValueError(
            f"Expected critic.0.weight to be rank-2, got shape {tuple(weight.shape)}"
        )
    return int(weight.shape[1])


def infer_rlinf_mlp_obs_dim(state_dict: dict[str, torch.Tensor]) -> int:
    weight = state_dict.get(f"{RLINF_VALUE_HEAD_PREFIX}0.weight")
    if weight is None:
        raise ValueError(
            "Checkpoint is missing value_head.mlp.0.weight. Only RLinf MLP policies "
            "with value_head.mlp.* parameters are supported."
        )
    if weight.ndim != 2:
        raise ValueError(
            "Expected value_head.mlp.0.weight to be rank-2, got shape "
            f"{tuple(weight.shape)}"
        )
    return int(weight.shape[1])


def load_value_model(path: str, device: str) -> LoadedValueModel:
    state_dict = load_raw_checkpoint(path)
    checkpoint_format = _detect_checkpoint_format(state_dict)
    resolved_device = resolve_device(device)

    if checkpoint_format == "local_ppo":
        obs_dim = infer_local_ppo_obs_dim(state_dict)
        critic_state_dict = _extract_prefixed_state_dict(
            state_dict, LOCAL_CRITIC_PREFIX
        )
        _validate_shapes(
            critic_state_dict,
            obs_dim=obs_dim,
            expected_shapes=LOCAL_EXPECTED_SHAPES,
            checkpoint_label="local ManiSkill PPO critic",
        )
        model = LocalPPOCritic(obs_dim)
        model.network.load_state_dict(critic_state_dict, strict=True)
        return LoadedValueModel(
            model=model.to(resolved_device).eval(),
            obs_dim=obs_dim,
            checkpoint_format=checkpoint_format,
            final_bias=True,
        )

    obs_dim = infer_rlinf_mlp_obs_dim(state_dict)
    value_head_state_dict = _extract_prefixed_state_dict(
        state_dict, RLINF_VALUE_HEAD_PREFIX
    )
    _validate_shapes(
        value_head_state_dict,
        obs_dim=obs_dim,
        expected_shapes=RLINF_EXPECTED_SHAPES,
        checkpoint_label="RLinf value_head.mlp",
    )
    model = RLinfMLPValueHead(obs_dim)
    model.network.load_state_dict(value_head_state_dict, strict=True)
    return LoadedValueModel(
        model=model.to(resolved_device).eval(),
        obs_dim=obs_dim,
        checkpoint_format=checkpoint_format,
        final_bias=False,
    )


def compute_gae_targets(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    rewards = rewards.float()
    values = values.float()
    next_values = next_values.float()
    dones = dones.bool()
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape [T, B], got {tuple(rewards.shape)}")
    if values.shape != rewards.shape:
        raise ValueError(
            f"values shape {tuple(values.shape)} must match rewards shape {tuple(rewards.shape)}"
        )
    if next_values.shape != rewards.shape:
        raise ValueError(
            "next_values shape "
            f"{tuple(next_values.shape)} must match rewards shape {tuple(rewards.shape)}"
        )
    if dones.shape != rewards.shape:
        raise ValueError(
            f"dones shape {tuple(dones.shape)} must match rewards shape {tuple(rewards.shape)}"
        )

    targets = torch.zeros_like(rewards, dtype=torch.float32)
    gae = torch.zeros((rewards.shape[1],), dtype=torch.float32)
    for step in range(rewards.shape[0] - 1, -1, -1):
        not_done = (~dones[step]).float()
        delta = rewards[step] + gamma * next_values[step] * not_done - values[step]
        gae = delta + gamma * gae_lambda * not_done * gae
        targets[step] = gae + values[step]
    return targets


def compute_reward_assign_lengths(
    history_lengths: Optional[dict[str, list[int]]],
    *,
    num_envs: int,
    current_rollout_length: int,
) -> torch.Tensor:
    if history_lengths is None:
        return torch.ones((num_envs,), dtype=torch.long)

    assign_lengths = []
    for env_id in range(num_envs):
        assign_length = min(
            history_buffer_length[env_id]
            for history_buffer_length in history_lengths.values()
        )
        assign_lengths.append(min(assign_length, current_rollout_length))
    return torch.as_tensor(assign_lengths, dtype=torch.long)


def compute_delta_gae_step_rewards(
    gae_targets: torch.Tensor,
    reward_assign_lengths: torch.Tensor,
    *,
    label_mode: str,
    delta_threshold: float,
    positive_reward: float,
    unchanged_reward: float,
    negative_reward: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gae_targets.ndim != 2:
        raise ValueError(
            f"gae_targets must have shape [T, B], got {tuple(gae_targets.shape)}"
        )
    if reward_assign_lengths.shape != gae_targets.shape:
        raise ValueError(
            "reward_assign_lengths shape "
            f"{tuple(reward_assign_lengths.shape)} must match gae_targets shape {tuple(gae_targets.shape)}"
        )

    reward_assign_lengths = reward_assign_lengths.to(dtype=torch.long)
    delta = torch.zeros_like(gae_targets, dtype=torch.float32)
    step_rewards = torch.full_like(
        gae_targets, fill_value=float(unchanged_reward), dtype=torch.float32
    )

    traj_len, batch_size = gae_targets.shape
    for step in range(traj_len):
        for env_id in range(batch_size):
            assign_length = max(int(reward_assign_lengths[step, env_id].item()), 1)
            start_idx = max(0, step - assign_length + 1)
            current_delta = float(
                gae_targets[step, env_id] - gae_targets[start_idx, env_id]
            )
            delta[step, env_id] = current_delta

    valid_mask = reward_assign_lengths > 1
    if label_mode == "threshold":
        threshold_eps = 1.0e-6
        positive_mask = delta > delta_threshold + threshold_eps
        negative_mask = delta < -(delta_threshold + threshold_eps)
    elif label_mode == "batch_tercile":
        valid_deltas = delta[valid_mask]
        if valid_deltas.numel() == 0:
            return step_rewards, delta
        lower_cutoff = torch.quantile(valid_deltas, 1.0 / 3.0)
        upper_cutoff = torch.quantile(valid_deltas, 2.0 / 3.0)
        positive_mask = valid_mask & (delta >= upper_cutoff)
        negative_mask = valid_mask & (delta <= lower_cutoff)
    else:
        raise ValueError(f"Unsupported oracle delta-GAE label mode: {label_mode}")

    step_rewards[positive_mask] = float(positive_reward)
    step_rewards[negative_mask] = float(negative_reward)
    return step_rewards, delta


def apply_history_reward_assignment(
    step_rewards: torch.Tensor,
    reward_assign_lengths: torch.Tensor,
) -> torch.Tensor:
    if step_rewards.ndim != 2:
        raise ValueError(
            f"step_rewards must have shape [T, B], got {tuple(step_rewards.shape)}"
        )
    if reward_assign_lengths.shape != step_rewards.shape:
        raise ValueError(
            "reward_assign_lengths shape "
            f"{tuple(reward_assign_lengths.shape)} must match step_rewards shape {tuple(step_rewards.shape)}"
        )

    assigned_rewards = step_rewards.clone()
    traj_len, batch_size = step_rewards.shape
    reward_assign_lengths = reward_assign_lengths.to(dtype=torch.long)
    for step in range(traj_len):
        for env_id in range(batch_size):
            assign_length = max(int(reward_assign_lengths[step, env_id].item()), 1)
            if assign_length <= 1:
                continue
            start_idx = max(0, step - assign_length + 1)
            assigned_rewards[start_idx:step, env_id] += step_rewards[step, env_id]
    return assigned_rewards


def apply_success_bonus(
    rewards: torch.Tensor,
    success_mask: Optional[torch.Tensor],
    success_bonus: float,
) -> torch.Tensor:
    if success_mask is None or success_bonus == 0.0:
        return rewards
    success_mask = success_mask.to(dtype=rewards.dtype)
    if success_mask.shape != rewards.shape:
        raise ValueError(
            f"success_mask shape {tuple(success_mask.shape)} must match rewards shape {tuple(rewards.shape)}"
        )
    return rewards + success_bonus * success_mask


def normalize_total_reward(
    rewards: torch.Tensor,
    *,
    mode: str,
    eps: float,
) -> torch.Tensor:
    if mode != "batch_zscore":
        raise ValueError(f"Unsupported reward normalization mode: {mode}")

    rewards = rewards.float()
    mean = rewards.mean()
    centered = rewards - mean
    variance = centered.pow(2).mean()
    std = variance.sqrt()
    if float(std.item()) < eps:
        return centered
    return centered / std.clamp_min(eps)


class OracleDeltaGAERewardProcessor:
    def __init__(self, cfg: OracleDeltaGAEConfig):
        self.cfg = cfg
        self.loaded_value_model = load_value_model(
            cfg.value_checkpoint_path,
            device=cfg.device,
        )
        self._model = self.loaded_value_model.model
        self._device = resolve_device(cfg.device)

    @torch.inference_mode()
    def predict_values(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError(
                f"states must have shape [T, B, D], got {tuple(states.shape)}"
            )
        if states.shape[-1] != self.loaded_value_model.obs_dim:
            raise ValueError(
                "State feature dimension "
                f"{states.shape[-1]} does not match oracle value model obs_dim "
                f"{self.loaded_value_model.obs_dim}"
            )
        flat_states = states.reshape(-1, states.shape[-1]).to(
            device=self._device, dtype=torch.float32
        )
        values = (
            self._model(flat_states).detach().float().cpu().reshape(*states.shape[:2])
        )
        return values

    @torch.inference_mode()
    def compute_oracle_rewards(
        self,
        *,
        curr_states: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        reward_assign_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = self.predict_values(curr_states)
        next_values = self.predict_values(next_states)
        gae_targets = compute_gae_targets(
            rewards,
            values,
            next_values,
            dones,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )
        step_rewards, delta = compute_delta_gae_step_rewards(
            gae_targets,
            reward_assign_lengths,
            label_mode=self.cfg.label_mode,
            delta_threshold=self.cfg.delta_threshold,
            positive_reward=self.cfg.positive_reward,
            unchanged_reward=self.cfg.unchanged_reward,
            negative_reward=self.cfg.negative_reward,
        )
        assigned_rewards = apply_history_reward_assignment(
            step_rewards,
            reward_assign_lengths,
        )
        return assigned_rewards, delta, gae_targets
