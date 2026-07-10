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

from dataclasses import dataclass, field
from typing import Any, Optional, TypeAlias

import torch

from rlinf.data.embodied_io_struct import Trajectory, get_model_weights_id
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.utils import normalize_device


def _split_protocol_value(value: Any, sizes: list[int]) -> list[Any]:
    if value is None:
        return [None for _ in sizes]
    if isinstance(value, (torch.Tensor, dict, list)):
        from rlinf.scheduler import split_batch

        return split_batch(value, sizes)
    return [value for _ in sizes]


@dataclass
class Observations:
    global_step: int
    rank: int
    current_step: int
    current_epoch: int
    mode: str = "train"
    stage_id: int = 0
    obs: dict[str, torch.Tensor] = field(default_factory=dict)
    next_obs: dict[str, torch.Tensor] = field(default_factory=dict)
    reward_obs: dict[str, Any] = field(default_factory=dict)
    rlt_switch_flags: Optional[torch.Tensor] = None
    intervene_actions: Optional[torch.Tensor] = None
    intervene_flags: Optional[torch.Tensor] = None
    env_infos: Optional[dict[str, Any]] = None
    has_final_obs: bool = False
    rewards: Optional[torch.Tensor] = None
    dones: Optional[torch.Tensor] = None
    terminations: Optional[torch.Tensor] = None
    truncations: Optional[torch.Tensor] = None
    # The regular fields above are delivered to the rollout worker.  A single
    # post-step message also carries the transition-aligned view required by
    # storage: the observation before the action and the previous transition's
    # reward.  Keeping this metadata on the protocol object lets
    # TrajectoryChannel persist it without a second application-level send.
    storage_obs: Optional[dict[str, torch.Tensor]] = None
    storage_next_obs: Optional[dict[str, torch.Tensor]] = None
    storage_reward_obs: Optional[dict[str, Any]] = None
    storage_env_infos: Optional[dict[str, Any]] = None
    storage_has_final_obs: bool = False
    storage_rewards: Optional[torch.Tensor] = None
    storage_dones: Optional[torch.Tensor] = None
    storage_terminations: Optional[torch.Tensor] = None
    storage_truncations: Optional[torch.Tensor] = None

    def to_rollout_dict(self) -> dict[str, Any]:
        return {
            "obs": self.obs,
            "final_obs": self.next_obs if self.has_final_obs else None,
            "rlt_switch_flags": self.rlt_switch_flags,
            "intervene_flags": self.intervene_flags,
        }

    def split(self, sizes: list[int]) -> list["Observations"]:
        values = {
            field_name: _split_protocol_value(getattr(self, field_name), sizes)
            for field_name in self.__dataclass_fields__
        }
        return [
            Observations(
                **{
                    field_name: field_values[index]
                    for field_name, field_values in values.items()
                }
            )
            for index in range(len(sizes))
        ]


@dataclass
class Intervention:
    global_step: int
    rank: int
    current_step: int
    current_epoch: int
    intervene_actions: torch.Tensor
    intervene_flags: torch.Tensor


@dataclass
class Actions:
    global_step: int
    rank: int
    current_step: int
    current_epoch: int
    actions: torch.Tensor
    mode: str = "train"
    stage_id: int = 0
    prev_logprobs: Optional[torch.Tensor] = None
    prev_values: Optional[torch.Tensor] = None
    bootstrap_values: Optional[torch.Tensor] = None
    intervene_flags: Optional[torch.Tensor] = None
    versions: Optional[torch.Tensor] = None
    save_flags: Optional[torch.Tensor] = None
    forward_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    is_bootstrap: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "actions",
            "prev_logprobs",
            "prev_values",
            "bootstrap_values",
            "intervene_flags",
            "versions",
            "save_flags",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, value.detach().cpu().contiguous())
        if self.forward_inputs:
            self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")

    @staticmethod
    def merge_actions(actions_list: list["Actions"]) -> "Actions":
        """Merge routed action shards while preserving their protocol metadata."""
        if not actions_list:
            raise ValueError("Cannot merge an empty action batch.")

        def merge_optional(field_name: str) -> torch.Tensor | None:
            values = [getattr(actions, field_name) for actions in actions_list]
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                raise ValueError(f"Inconsistent action field: {field_name}.")
            return torch.cat(values, dim=0)

        first = actions_list[0]
        if any(
            (
                item.global_step,
                item.current_epoch,
                item.current_step,
                item.mode,
                item.stage_id,
                item.is_bootstrap,
            )
            != (
                first.global_step,
                first.current_epoch,
                first.current_step,
                first.mode,
                first.stage_id,
                first.is_bootstrap,
            )
            for item in actions_list[1:]
        ):
            raise ValueError("Cannot merge Actions from different protocol steps.")

        forward_input_keys = set().union(
            *(item.forward_inputs.keys() for item in actions_list)
        )
        forward_inputs = {
            key: torch.cat([item.forward_inputs[key] for item in actions_list], dim=0)
            for key in forward_input_keys
        }
        return Actions(
            global_step=first.global_step,
            rank=first.rank,
            current_step=first.current_step,
            current_epoch=first.current_epoch,
            actions=merge_optional("actions"),
            mode=first.mode,
            stage_id=first.stage_id,
            prev_logprobs=merge_optional("prev_logprobs"),
            prev_values=merge_optional("prev_values"),
            bootstrap_values=merge_optional("bootstrap_values"),
            intervene_flags=merge_optional("intervene_flags"),
            versions=merge_optional("versions"),
            save_flags=merge_optional("save_flags"),
            forward_inputs=forward_inputs,
            is_bootstrap=first.is_bootstrap,
        )


@dataclass
class EnvBootstrap:
    global_step: int
    rank: int
    current_epoch: int
    observations: dict[str, torch.Tensor] = field(default_factory=dict)
    env_infos: Optional[dict[str, Any]] = None
    has_final_obs: bool = False
    rewards: Optional[torch.Tensor] = None
    dones: Optional[torch.Tensor] = None
    terminations: Optional[torch.Tensor] = None
    truncations: Optional[torch.Tensor] = None

    def to_rollout_dict(self) -> dict[str, Any]:
        return {
            "obs": self.observations,
            "final_obs": self.observations if self.has_final_obs else None,
            "rlt_switch_flags": None,
            "intervene_flags": None,
        }

    def split(self, sizes: list[int]) -> list["EnvBootstrap"]:
        values = {
            field_name: _split_protocol_value(getattr(self, field_name), sizes)
            for field_name in self.__dataclass_fields__
        }
        return [
            EnvBootstrap(
                **{
                    field_name: field_values[index]
                    for field_name, field_values in values.items()
                }
            )
            for index in range(len(sizes))
        ]


@dataclass
class RewardRequest:
    global_step: int
    rank: int
    source_rank: int
    current_step: int
    current_epoch: int
    reward_mode: str
    observations: dict[str, Any] = field(default_factory=dict)
    history_lengths: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class RolloutBootstrap:
    global_step: int
    rank: int
    current_epoch: int
    prev_values: Optional[torch.Tensor] = None


@dataclass
class Rewards:
    global_step: int
    rank: int
    current_step: int
    current_epoch: int
    rewards: torch.Tensor
    reward_mode: str = "per_step"
    history_lengths: dict[str, list[int]] = field(default_factory=dict)


TrajectoryData: TypeAlias = (
    Observations
    | Intervention
    | Actions
    | EnvBootstrap
    | RewardRequest
    | RolloutBootstrap
    | Rewards
)


class TrajectoryStorage:
    """Fixed-size transition storage for one training global step."""

    def __init__(
        self,
        *,
        num_envs: int,
        rollout_epoch: int,
        max_steps_per_rollout_epoch: int,
        max_episode_length: int,
        data_device: torch.device | str = "cpu",
        requires_values: bool = True,
    ):
        self.num_envs = int(num_envs)
        self.rollout_epoch = int(rollout_epoch)
        self.max_steps_per_rollout_epoch = int(max_steps_per_rollout_epoch)
        self.max_episode_length = int(max_episode_length)
        self.num_steps = self.rollout_epoch * self.max_steps_per_rollout_epoch
        self.num_value_steps = self.rollout_epoch * (
            self.max_steps_per_rollout_epoch + 1
        )
        self.data_device = normalize_device(data_device)
        self.requires_values = bool(requires_values)

        self.global_step: Optional[int] = None
        self.curr_obs: dict[str, torch.Tensor] = {}
        self.next_obs: dict[str, torch.Tensor] = {}
        self.forward_inputs: dict[str, torch.Tensor] = {}
        self.actions: Optional[torch.Tensor] = None
        self.intervene_flags: Optional[torch.Tensor] = None
        self.rewards: Optional[torch.Tensor] = None
        self.terminations: Optional[torch.Tensor] = None
        self.truncations: Optional[torch.Tensor] = None
        self.dones: Optional[torch.Tensor] = None
        self.prev_logprobs: Optional[torch.Tensor] = None
        self.prev_values: Optional[torch.Tensor] = None
        self.versions: Optional[torch.Tensor] = None

        self.observation_ready = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.bool
        )
        self.action_ready = torch.zeros_like(self.observation_ready)
        self.reward_ready = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.bool
        )
        self.value_ready = torch.zeros(
            (self.num_value_steps, self.num_envs), dtype=torch.bool
        )
        self.failed_slot_mask = torch.zeros(self.num_envs, dtype=torch.bool)
        self.bootstrap_reward_applied = torch.zeros(
            self.rollout_epoch, dtype=torch.bool
        )
        self._pending_interventions: list[
            tuple[int, int, list[int], torch.Tensor, torch.Tensor]
        ] = []

    def reset(self, global_step: Optional[int] = None) -> None:
        self.global_step = global_step
        for tensor_dict in (self.curr_obs, self.next_obs, self.forward_inputs):
            for tensor in tensor_dict.values():
                tensor.zero_()
        for tensor in (
            self.actions,
            self.intervene_flags,
            self.rewards,
            self.terminations,
            self.truncations,
            self.dones,
            self.prev_logprobs,
            self.prev_values,
            self.versions,
        ):
            if tensor is not None:
                tensor.zero_()
        self.observation_ready.zero_()
        self.action_ready.zero_()
        self.reward_ready.zero_()
        self.value_ready.zero_()
        self.failed_slot_mask.zero_()
        self.bootstrap_reward_applied.zero_()
        self._pending_interventions.clear()

    def write_observations(
        self,
        data: Observations,
        slot_indices: list[int],
        *,
        mark_rewards: bool = True,
    ) -> None:
        self._ensure_global_step(data.global_step)
        time_idx = self._action_time_index(data.current_epoch, data.current_step)
        value_time_idx = self._value_time_index(data.current_epoch, data.current_step)
        reward_time_idx = self._previous_reward_time_index(
            data.current_epoch, data.current_step
        )
        slot_indices = self._validate_slots(slot_indices)
        self._write_tensor_dict(self.curr_obs, data.obs, time_idx, slot_indices)
        self._write_tensor_dict(self.next_obs, data.next_obs, time_idx, slot_indices)
        if reward_time_idx is not None:
            self.rewards = self._write_step_tensor(
                self.rewards,
                data.rewards,
                reward_time_idx,
                slot_indices,
                total_steps=self.num_steps,
            )
        self.terminations = self._write_step_tensor(
            self.terminations,
            data.terminations,
            value_time_idx,
            slot_indices,
            total_steps=self.num_value_steps,
        )
        self.truncations = self._write_step_tensor(
            self.truncations,
            data.truncations,
            value_time_idx,
            slot_indices,
            total_steps=self.num_value_steps,
        )
        self.dones = self._write_step_tensor(
            self.dones,
            data.dones,
            value_time_idx,
            slot_indices,
            total_steps=self.num_value_steps,
        )
        self.observation_ready[time_idx, slot_indices] = True
        if mark_rewards and reward_time_idx is not None:
            self.reward_ready[reward_time_idx, slot_indices] = True

    def write_actions(self, data: Actions, slot_indices: list[int]) -> None:
        self._ensure_global_step(data.global_step)
        time_idx = self._action_time_index(data.current_epoch, data.current_step)
        value_time_idx = self._value_time_index(data.current_epoch, data.current_step)
        slot_indices = self._validate_slots(slot_indices)
        stored_actions = data.forward_inputs.get("action", data.actions)
        self.actions = self._write_step_tensor(
            self.actions, stored_actions, time_idx, slot_indices
        )
        self.prev_logprobs = self._write_step_tensor(
            self.prev_logprobs, data.prev_logprobs, time_idx, slot_indices
        )
        self.prev_values = self._write_step_tensor(
            self.prev_values,
            data.prev_values,
            value_time_idx,
            slot_indices,
            total_steps=self.num_value_steps,
        )
        if data.prev_values is not None:
            self.value_ready[value_time_idx, slot_indices] = True
        self.versions = self._write_step_tensor(
            self.versions, data.versions, time_idx, slot_indices
        )
        for key, value in data.forward_inputs.items():
            self.forward_inputs[key] = self._write_step_tensor(
                self.forward_inputs.get(key), value, time_idx, slot_indices
            )
        self.action_ready[time_idx, slot_indices] = True
        if data.save_flags is not None:
            self.mark_save_flags(
                current_epoch=data.current_epoch,
                current_step=data.current_step,
                slot_indices=slot_indices,
                save_flags=data.save_flags,
            )
        self._apply_pending_interventions(
            current_epoch=data.current_epoch,
            current_step=data.current_step,
            slot_indices=slot_indices,
        )

    def write_rewards(
        self,
        data: Rewards,
        slot_indices: list[int],
        *,
        reward_weight: float = 1.0,
        env_reward_weight: float = 0.0,
    ) -> None:
        self._ensure_global_step(data.global_step)
        time_idx = self._reward_time_index_for_step(
            data.current_epoch, data.current_step
        )
        slot_indices = self._validate_slots(slot_indices)
        rewards = data.rewards
        if env_reward_weight and self.rewards is not None:
            env_rewards = self.rewards[time_idx, slot_indices].detach().cpu()
            rewards = env_rewards.to(rewards.dtype) * env_reward_weight + (
                rewards * reward_weight
            )
        elif reward_weight != 1.0:
            rewards = rewards * reward_weight
        self.rewards = self._write_step_tensor(
            self.rewards,
            rewards,
            time_idx,
            slot_indices,
            total_steps=self.num_steps,
        )
        self.reward_ready[time_idx, slot_indices] = True

    def write_terminal_rewards(
        self,
        data: Rewards,
        slot_indices: list[int],
        *,
        reward_weight: float = 1.0,
        env_reward_weight: float = 0.0,
    ) -> None:
        self._ensure_global_step(data.global_step)
        reward_time_idx = self._reward_time_index_for_step(
            data.current_epoch, data.current_step
        )
        done_time_idx = self._done_time_index_for_step(
            data.current_epoch, data.current_step
        )
        slots = self._validate_slots(slot_indices)
        if self.rewards is None or self.dones is None:
            return

        env_rewards = self.rewards[reward_time_idx, slots]
        dones = self.dones[done_time_idx, slots].to(torch.bool)
        if dones.ndim == 1:
            dones = dones[:, None]
        sparse_rewards = torch.zeros_like(env_rewards, dtype=data.rewards.dtype)
        done_envs = dones.any(dim=1)
        if done_envs.any():
            done_steps = dones.to(torch.int64).argmax(dim=1)
            reward_values = (
                data.rewards.detach().to(self.data_device).reshape(len(slots), -1)
            )
            sparse_rewards[done_envs, done_steps[done_envs]] = (
                reward_values[done_envs].reshape(-1).to(sparse_rewards.dtype)
            )

        rewards = sparse_rewards * reward_weight
        if env_reward_weight:
            rewards = env_rewards.to(rewards.dtype) * env_reward_weight + rewards
        self.rewards = self._write_step_tensor(
            self.rewards,
            rewards.detach().cpu(),
            reward_time_idx,
            slots,
            total_steps=self.num_steps,
        )
        self.reward_ready[reward_time_idx, slots] = True

    def assign_history_rewards(
        self,
        *,
        current_epoch: int,
        current_step: int,
        slot_indices: list[int],
        rewards: torch.Tensor,
        history_lengths: dict[str, list[int]],
        reward_weight: float,
    ) -> None:
        if self.rewards is None or not history_lengths:
            return
        time_idx = self._reward_time_index_for_step(current_epoch, current_step)
        slots = self._validate_slots(slot_indices)
        reward_values = (rewards.detach().to(self.data_device) * reward_weight).reshape(
            len(slots), -1
        )
        assign_lengths = []
        for env_id in range(len(slots)):
            assign_lengths.append(
                min(
                    history_buffer_length[env_id]
                    for history_buffer_length in history_lengths.values()
                )
            )
        for local_idx, assign_length in enumerate(assign_lengths):
            for reward_assign_step in range(2, min(assign_length, time_idx + 1) + 1):
                self.rewards[time_idx - reward_assign_step + 1, slots[local_idx]] += (
                    reward_values[local_idx].reshape_as(
                        self.rewards[
                            time_idx - reward_assign_step + 1, slots[local_idx]
                        ]
                    )
                )

    def write_env_bootstrap(
        self,
        data: EnvBootstrap,
        slot_indices: list[int],
        *,
        mark_rewards: bool = True,
    ) -> None:
        self._ensure_global_step(data.global_step)
        value_time_idx = self._bootstrap_value_time_index(data.current_epoch)
        reward_time_idx = self._reward_time_index_for_step(
            data.current_epoch, self.max_steps_per_rollout_epoch
        )
        slot_indices = self._validate_slots(slot_indices)
        self.rewards = self._write_step_tensor(
            self.rewards,
            data.rewards,
            reward_time_idx,
            slot_indices,
            total_steps=self.num_steps,
        )
        self.terminations = self._write_step_tensor(
            self.terminations,
            data.terminations,
            value_time_idx,
            slot_indices,
            total_steps=self.num_value_steps,
        )
        self.truncations = self._write_step_tensor(
            self.truncations,
            data.truncations,
            value_time_idx,
            slot_indices,
            total_steps=self.num_value_steps,
        )
        self.dones = self._write_step_tensor(
            self.dones,
            data.dones,
            value_time_idx,
            slot_indices,
            total_steps=self.num_value_steps,
        )
        if mark_rewards:
            self.reward_ready[reward_time_idx, slot_indices] = True

    def write_rollout_bootstrap(
        self, data: RolloutBootstrap, slot_indices: list[int]
    ) -> None:
        self._ensure_global_step(data.global_step)
        time_idx = self._bootstrap_value_time_index(data.current_epoch)
        slot_indices = self._validate_slots(slot_indices)
        self.prev_values = self._write_step_tensor(
            self.prev_values,
            data.prev_values,
            time_idx,
            slot_indices,
            total_steps=self.num_value_steps,
        )
        if data.prev_values is not None:
            self.value_ready[time_idx, slot_indices] = True

    def apply_terminal_bootstrap_reward(
        self,
        *,
        current_epoch: int,
        gamma: float,
        auto_reset: bool,
        bootstrap_type: str,
    ) -> None:
        """Apply the final-value correction once both bootstrap records arrive."""
        self._check_epoch(current_epoch)
        if (
            self.bootstrap_reward_applied[current_epoch]
            or not auto_reset
            or self.rewards is None
            or self.dones is None
            or self.prev_values is None
        ):
            return
        done_time_idx = self._bootstrap_value_time_index(current_epoch)
        reward_time_idx = self._reward_time_index_for_step(
            current_epoch, self.max_steps_per_rollout_epoch
        )
        terminal_mask = (
            self.truncations[done_time_idx]
            if bootstrap_type == "standard" and self.truncations is not None
            else self.dones[done_time_idx]
        ).to(torch.bool)
        if terminal_mask.ndim > 1:
            terminal_mask = terminal_mask[:, -1]
        if terminal_mask.any():
            rewards = self.rewards[reward_time_idx]
            bootstrap_values = self.prev_values[done_time_idx]
            rewards[terminal_mask] += gamma * bootstrap_values[terminal_mask].reshape(
                -1, *([1] * (rewards.ndim - 1))
            ).to(rewards.dtype)
        self.bootstrap_reward_applied[current_epoch] = True

    def complete(self) -> bool:
        transition_ready = self.observation_ready & self.action_ready
        transition_ready[:, self.failed_slot_mask] = True
        reward_ready = self.reward_ready.clone()
        reward_ready[:, self.failed_slot_mask] = True
        if not bool(transition_ready.all().item() and reward_ready.all().item()):
            return False
        if not self.requires_values:
            return True
        value_ready = self.value_ready.clone()
        value_ready[:, self.failed_slot_mask] = True
        return bool(value_ready.all().item())

    def mark_failed_slots(self, slot_indices: list[int]) -> None:
        slots = self._validate_slots(slot_indices)
        self.failed_slot_mask[slots] = True

    def apply_intervention(
        self,
        *,
        current_epoch: int,
        current_step: int,
        slot_indices: list[int],
        intervene_actions: torch.Tensor,
        intervene_flags: torch.Tensor,
    ) -> None:
        time_idx = self._action_time_index(current_epoch, current_step)
        slots = self._validate_slots(slot_indices)
        if self.actions is None or not bool(self.action_ready[time_idx, slots].all()):
            self._pending_interventions.append(
                (
                    int(current_epoch),
                    int(current_step),
                    slots,
                    intervene_actions.detach().cpu(),
                    intervene_flags.detach().cpu(),
                )
            )
            return
        self._apply_intervention_to_actions(
            time_idx=time_idx,
            slots=slots,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
        )

    def _apply_intervention_to_actions(
        self,
        *,
        time_idx: int,
        slots: list[int],
        intervene_actions: torch.Tensor,
        intervene_flags: torch.Tensor,
    ) -> None:
        if self.actions is None:
            raise RuntimeError("Cannot apply intervention before actions are written.")
        if intervene_flags.dim() == 1:
            intervene_flags = intervene_flags[:, None]
        if intervene_actions.shape[0] != len(slots):
            raise ValueError(
                f"Expected intervention batch {len(slots)}, got {intervene_actions.shape}."
            )

        current_actions = self.actions[time_idx, slots]
        flags = intervene_flags.to(torch.bool).to(self.data_device)
        actions = intervene_actions.detach().to(self.data_device)
        full_actions = actions.reshape(
            actions.shape[0], flags.shape[1], -1
        ) * flags.reshape(flags.shape[0], flags.shape[1], 1) + current_actions.reshape(
            current_actions.shape[0], flags.shape[1], -1
        ) * (~flags.reshape(flags.shape[0], flags.shape[1], 1))
        self.actions[time_idx, slots] = full_actions.reshape_as(current_actions)
        expanded_flags = flags.reshape(flags.shape[0], flags.shape[1], 1).expand_as(
            full_actions
        )
        self.intervene_flags = self._write_step_tensor(
            self.intervene_flags,
            expanded_flags.reshape_as(current_actions).detach().cpu(),
            time_idx,
            slots,
        )
        if "action" in self.forward_inputs:
            self.forward_inputs["action"][time_idx, slots] = self.actions[
                time_idx, slots
            ]
        self.forward_inputs.pop("model_action", None)

    def _apply_pending_interventions(
        self,
        *,
        current_epoch: int,
        current_step: int,
        slot_indices: list[int],
    ) -> None:
        if not self._pending_interventions:
            return
        slots = self._validate_slots(slot_indices)
        remaining = []
        time_idx = self._action_time_index(current_epoch, current_step)
        for (
            pending_epoch,
            pending_step,
            pending_slots,
            intervene_actions,
            intervene_flags,
        ) in self._pending_interventions:
            if (
                pending_epoch == int(current_epoch)
                and pending_step == int(current_step)
                and pending_slots == slots
            ):
                self._apply_intervention_to_actions(
                    time_idx=time_idx,
                    slots=slots,
                    intervene_actions=intervene_actions,
                    intervene_flags=intervene_flags,
                )
            else:
                remaining.append(
                    (
                        pending_epoch,
                        pending_step,
                        pending_slots,
                        intervene_actions,
                        intervene_flags,
                    )
                )
        self._pending_interventions = remaining

    def mark_save_flags(
        self,
        *,
        current_epoch: int,
        current_step: int,
        slot_indices: list[int],
        save_flags: torch.Tensor,
    ) -> None:
        if self.actions is None:
            return
        time_idx = self._action_time_index(current_epoch, current_step)
        slots = self._validate_slots(slot_indices)
        if save_flags.dim() == 1:
            save_flags = save_flags[:, None]
        current_actions = self.actions[time_idx, slots]
        flags = save_flags.to(torch.bool).to(self.data_device)
        expanded_flags = flags.reshape(flags.shape[0], flags.shape[1], 1).expand_as(
            current_actions.reshape(current_actions.shape[0], flags.shape[1], -1)
        )
        self.intervene_flags = self._write_step_tensor(
            self.intervene_flags,
            expanded_flags.reshape_as(current_actions).detach().cpu(),
            time_idx,
            slots,
        )

    def to_trajectory(self, slot_indices: Optional[list[int]] = None) -> Trajectory:
        slots = self._validate_slots(
            list(range(self.num_envs)) if slot_indices is None else slot_indices
        )
        versions = (
            self.versions[:, slots].contiguous()
            if self.versions is not None
            else torch.zeros(1, dtype=torch.float32)
        )
        trajectory = Trajectory(
            max_episode_length=self.max_episode_length,
            model_weights_id=get_model_weights_id(versions),
        )
        if self.actions is not None:
            trajectory.actions = self.actions[:, slots].cpu().contiguous()
        if self.intervene_flags is not None:
            trajectory.intervene_flags = (
                self.intervene_flags[:, slots].cpu().contiguous()
            )
        if self.rewards is not None:
            trajectory.rewards = self.rewards[:, slots].cpu().contiguous()
        if self.terminations is not None:
            trajectory.terminations = self.terminations[:, slots].cpu().contiguous()
        if self.truncations is not None:
            trajectory.truncations = self.truncations[:, slots].cpu().contiguous()
        if self.dones is not None:
            trajectory.dones = self.dones[:, slots].cpu().contiguous()
        if self.prev_logprobs is not None:
            trajectory.prev_logprobs = self.prev_logprobs[:, slots].cpu().contiguous()
        if self.prev_values is not None:
            trajectory.prev_values = self.prev_values[:, slots].cpu().contiguous()
        if self.versions is not None:
            trajectory.versions = self.versions[:, slots].cpu().contiguous()
        trajectory.forward_inputs = {
            key: value[:, slots].cpu().contiguous()
            for key, value in self.forward_inputs.items()
        }
        trajectory.curr_obs = {
            key: value[:, slots].cpu().contiguous()
            for key, value in self.curr_obs.items()
        }
        trajectory.next_obs = {
            key: value[:, slots].cpu().contiguous()
            for key, value in self.next_obs.items()
        }
        return trajectory

    def _ensure_global_step(self, global_step: int) -> None:
        if self.global_step is None:
            self.global_step = int(global_step)
            return
        if self.global_step != int(global_step):
            raise ValueError(
                f"Storage is for global_step={self.global_step}, got {global_step}."
            )

    def _check_epoch(self, current_epoch: int) -> None:
        if current_epoch < 0 or current_epoch >= self.rollout_epoch:
            raise IndexError(
                f"current_epoch={current_epoch} out of range [0, {self.rollout_epoch - 1}]."
            )

    def _action_time_index(self, current_epoch: int, current_step: int) -> int:
        self._check_epoch(current_epoch)
        if current_step < 0 or current_step >= self.max_steps_per_rollout_epoch:
            raise IndexError(
                f"current_step={current_step} out of range [0, {self.max_steps_per_rollout_epoch - 1}]."
            )
        return current_epoch * self.max_steps_per_rollout_epoch + current_step

    def _value_time_index(self, current_epoch: int, current_step: int) -> int:
        self._check_epoch(current_epoch)
        if current_step < 0 or current_step >= self.max_steps_per_rollout_epoch:
            raise IndexError(
                f"current_step={current_step} out of range [0, {self.max_steps_per_rollout_epoch - 1}]."
            )
        return current_epoch * (self.max_steps_per_rollout_epoch + 1) + current_step

    def _bootstrap_value_time_index(self, current_epoch: int) -> int:
        self._check_epoch(current_epoch)
        return (
            current_epoch * (self.max_steps_per_rollout_epoch + 1)
            + self.max_steps_per_rollout_epoch
        )

    def _previous_reward_time_index(
        self, current_epoch: int, current_step: int
    ) -> Optional[int]:
        if current_step == 0:
            return None
        return self._action_time_index(current_epoch, current_step - 1)

    def _reward_time_index_for_step(self, current_epoch: int, current_step: int) -> int:
        if current_step <= 0:
            raise IndexError(
                "Rewards are transition-aligned and require current_step > 0."
            )
        if current_step > self.max_steps_per_rollout_epoch:
            raise IndexError(
                f"current_step={current_step} out of range [1, {self.max_steps_per_rollout_epoch}]."
            )
        return self._action_time_index(current_epoch, current_step - 1)

    def _done_time_index_for_step(self, current_epoch: int, current_step: int) -> int:
        if current_step == self.max_steps_per_rollout_epoch:
            return self._bootstrap_value_time_index(current_epoch)
        return self._value_time_index(current_epoch, current_step)

    def _validate_slots(self, slot_indices: list[int]) -> list[int]:
        slots = [int(slot) for slot in slot_indices]
        for slot in slots:
            if slot < 0 or slot >= self.num_envs:
                raise IndexError(f"slot={slot} out of range [0, {self.num_envs - 1}].")
        return slots

    def _write_step_tensor(
        self,
        current: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        time_idx: int,
        slot_indices: list[int],
        *,
        total_steps: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return current
        value = value.detach().to(self.data_device)
        if value.shape[0] != len(slot_indices):
            raise ValueError(
                f"Expected batch {len(slot_indices)}, got tensor shape {value.shape}."
            )
        if current is None:
            current = torch.zeros(
                (
                    self.num_steps if total_steps is None else total_steps,
                    self.num_envs,
                    *value.shape[1:],
                ),
                dtype=value.dtype,
                device=self.data_device,
            )
        current[time_idx, slot_indices] = value
        return current

    def _write_tensor_dict(
        self,
        storage: dict[str, torch.Tensor],
        values: dict[str, torch.Tensor],
        time_idx: int,
        slot_indices: list[int],
    ) -> None:
        for key, value in values.items():
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"TrajectoryStorage only supports tensor values for '{key}', got {type(value)}."
                )
            storage[key] = self._write_step_tensor(
                storage.get(key), value, time_idx, slot_indices
            )
