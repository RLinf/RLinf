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
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    pass

from rlinf.utils.nested_dict_process import (
    cat_list_of_dict_tensor,
    put_tensor_device,
    split_dict,
    split_dict_to_chunk,
    stack_list_of_dict_tensor,
)


def get_model_weights_id(versions: torch.Tensor) -> str:
    """
    Get the model weights id from the tensor.

    Args:
        versions (torch.Tensor): The tensor to get the model weights id from.

    Returns:
        str: The model weights id.
    """

    name_bytes = versions.cpu().numpy().tobytes()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name_bytes.hex()))


@dataclass(kw_only=True)
class EnvOutput:
    """Environment output for a single chunk step."""

    obs: dict[str, Any]
    final_obs: Optional[dict[str, Any]] = None
    dones: Optional[torch.Tensor] = None  # [B]
    terminations: Optional[torch.Tensor] = None  # [B]
    truncations: Optional[torch.Tensor] = None  # [B]
    rewards: Optional[torch.Tensor] = None  # [B]
    env_infos: Optional[dict[str, Any]] = None

    intervene_actions: Optional[torch.Tensor] = None  # [B]
    intervene_flags: Optional[torch.Tensor] = None  # [B]
    rlt_switch_flags: Optional[torch.Tensor] = None  # [B] or [B, action_chunk]

    def __post_init__(self):
        self.obs = put_tensor_device(self.obs, "cpu")
        self.final_obs = (
            put_tensor_device(self.final_obs, "cpu")
            if self.final_obs is not None
            else None
        )
        self.dones = self.dones.cpu().contiguous() if self.dones is not None else None
        self.terminations = (
            self.terminations.cpu().contiguous()
            if self.terminations is not None
            else None
        )
        self.truncations = (
            self.truncations.cpu().contiguous()
            if self.truncations is not None
            else None
        )
        self.rewards = (
            self.rewards.cpu().contiguous() if self.rewards is not None else None
        )
        self.env_infos = (
            put_tensor_device(self.env_infos, "cpu")
            if self.env_infos is not None
            else None
        )
        self.intervene_actions = (
            self.intervene_actions.cpu().contiguous()
            if self.intervene_actions is not None
            else None
        )
        self.intervene_flags = (
            self.intervene_flags.cpu().contiguous()
            if self.intervene_flags is not None
            else None
        )
        self.rlt_switch_flags = (
            self.rlt_switch_flags.cpu().contiguous()
            if self.rlt_switch_flags is not None
            else None
        )

    def prepare_observations(self, obs: dict[str, Any]) -> dict[str, Any]:
        image_tensor = obs["main_images"] if "main_images" in obs else None
        wrist_image_tensor = obs["wrist_images"] if "wrist_images" in obs else None
        extra_view_image_tensor = (
            obs["extra_view_images"] if "extra_view_images" in obs else None
        )
        states = obs["states"] if "states" in obs else None
        task_descriptions = (
            list(obs["task_descriptions"])
            if "task_descriptions" in obs and obs["task_descriptions"] is not None
            else None
        )

        return {
            "main_images": image_tensor,  # [N_ENV, H, W, C]
            "wrist_images": wrist_image_tensor,  # [N_ENV, H, W, C] or [N_ENV, N_IMG, H, W, C]
            "extra_view_images": extra_view_image_tensor,  # [N_ENV, N_IMG, H, W, C]
            "states": states,
            "task_descriptions": task_descriptions,
        }

    @staticmethod
    def merge_env_outputs(env_outputs: list[dict]) -> dict[str, Any]:
        """Merge multiple env output dicts into one batch-aligned env output.

        Merge strategy:

        - Tensor fields: concatenate on batch dimension.
        - List fields: flatten in source order.
        - ``None`` fields: keep ``None``.
        - ``final_obs`` supports partial ``None`` across shards. For shards
            without ``final_obs``, use the corresponding ``obs`` as fallback to
            keep batch alignment.

        Args:
            env_outputs: Per-source env output dicts that share the same schema.

        Returns:
            A merged env output dict produced via ``EnvOutput(...).to_dict()``.
        """

        def _get_batch_size(env_output: dict[str, Any]) -> int:
            dones = env_output.get("dones")
            if isinstance(dones, torch.Tensor):
                return dones.shape[0]

            obs = env_output["obs"]
            for key in ("states", "main_images", "task_descriptions"):
                value = obs.get(key)
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
                if isinstance(value, list):
                    return len(value)
            raise ValueError("Cannot infer batch size from env output.")

        def _merge_obs_dicts(obs_dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged_obs = {}
            for key in obs_dicts[0].keys():
                obs_elements = [obs_dict[key] for obs_dict in obs_dicts]
                first_non_none = next(
                    (element for element in obs_elements if element is not None), None
                )
                if first_non_none is None:
                    merged_obs[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged_obs[key] = torch.cat(obs_elements, dim=0)
                elif isinstance(first_non_none, list):
                    merged_obs[key] = [
                        item for sublist in obs_elements for item in sublist
                    ]
                else:
                    merged_obs[key] = obs_elements
            return merged_obs

        def _merge_optional_tensor_field(
            field_name: str,
            *,
            allow_partial_none: bool = False,
            fill_value: float | bool = 0,
        ) -> torch.Tensor | None:
            values = [env_output[field_name] for env_output in env_outputs]
            if all(value is None for value in values):
                return None

            if any(value is None for value in values):
                if not allow_partial_none:
                    raise ValueError(
                        f"Inconsistent field '{field_name}': some shards are None while others are tensors."
                    )

                ref_tensor = next(value for value in values if value is not None)
                filled_values = []
                for env_output, value in zip(env_outputs, values):
                    if value is None:
                        batch_size = _get_batch_size(env_output)
                        fill_shape = (batch_size, *ref_tensor.shape[1:])
                        filled_values.append(
                            torch.full(
                                fill_shape,
                                fill_value=fill_value,
                                dtype=ref_tensor.dtype,
                            )
                        )
                    else:
                        filled_values.append(value)
                values = filled_values

            return torch.cat(values, dim=0)

        merged_obs = _merge_obs_dicts([env_output["obs"] for env_output in env_outputs])

        merged_final_obs = None
        final_obs_list = [env_output["final_obs"] for env_output in env_outputs]
        if any(final_obs is not None for final_obs in final_obs_list):
            # Some shards may not have done episodes in this step, so their final_obs
            # is None. Use obs as fallback to keep merged batch shape aligned.
            final_obs_or_obs = [
                final_obs if final_obs is not None else env_output["obs"]
                for env_output, final_obs in zip(env_outputs, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        merged_dones = _merge_optional_tensor_field("dones")
        merged_terminations = _merge_optional_tensor_field("terminations")
        merged_truncations = _merge_optional_tensor_field("truncations")
        merged_rewards = _merge_optional_tensor_field("rewards")
        merged_intervene_actions = _merge_optional_tensor_field(
            "intervene_actions",
            allow_partial_none=True,
            fill_value=0.0,
        )
        merged_intervene_flags = _merge_optional_tensor_field(
            "intervene_flags",
            allow_partial_none=True,
            fill_value=False,
        )
        merged_rlt_switch_flags = _merge_optional_tensor_field(
            "rlt_switch_flags",
            allow_partial_none=True,
            fill_value=False,
        )
        # turn to EnvOutput and turn to dict to call post init for tensor processing
        return EnvOutput(
            obs=merged_obs,
            final_obs=merged_final_obs,
            dones=merged_dones,
            terminations=merged_terminations,
            truncations=merged_truncations,
            rewards=merged_rewards,
            intervene_actions=merged_intervene_actions,
            intervene_flags=merged_intervene_flags,
            rlt_switch_flags=merged_rlt_switch_flags,
        ).to_dict()

    def to_dict(self) -> dict[str, Any]:
        env_output_dict = {}

        env_output_dict["obs"] = self.prepare_observations(self.obs)
        env_output_dict["final_obs"] = (
            self.prepare_observations(self.final_obs)
            if self.final_obs is not None
            else None
        )
        env_output_dict["dones"] = self.dones
        env_output_dict["terminations"] = self.terminations
        env_output_dict["truncations"] = self.truncations
        env_output_dict["rewards"] = self.rewards
        env_output_dict["env_infos"] = self.env_infos
        env_output_dict["intervene_actions"] = self.intervene_actions
        env_output_dict["intervene_flags"] = self.intervene_flags
        env_output_dict["rlt_switch_flags"] = self.rlt_switch_flags

        return env_output_dict


@dataclass(kw_only=True)
class RTCRequest:
    """Real-time correction request sent from the env worker to rollout."""

    obs: dict[str, Any]
    request_type: str = "bootstrap"
    executed_horizon: int = 0
    predicted_delay_steps: int = 0
    chunk_id: int = 0

    def __post_init__(self):
        # Keep Ray channel payloads on CPU so the control node never receives
        # CUDA tensors from the rollout node.
        self.obs = put_tensor_device(self.obs, "cpu")


@dataclass(kw_only=True)
class RTCActionResponse:
    """RTC response carrying a fresh action chunk."""

    actions: torch.Tensor = None
    model_actions: torch.Tensor | None = None
    chunk_id: int = 0
    guidance_applied: bool = False

    def __post_init__(self):
        # Actions are executed by the env worker, while model_actions are kept
        # for the next RTC overlap constraint.
        if self.actions is not None:
            self.actions = self.actions.cpu().contiguous()
        if self.model_actions is not None:
            self.model_actions = self.model_actions.cpu().contiguous()


@dataclass(kw_only=True)
class RolloutResult:
    """Rollout result for a single chunk step."""

    actions: torch.Tensor = None  # [B, action_dim]
    prev_logprobs: torch.Tensor = None  # [B, action_dim]
    prev_values: torch.Tensor = None  # [B, 1]

    bootstrap_values: torch.Tensor = None  # [B, 1]
    intervene_flags: torch.Tensor = None  # [B, num_action_chunks]
    forward_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    versions: torch.Tensor = None  # [B, 1]

    def __post_init__(self):
        if self.actions is not None:
            self.actions = self.actions.cpu().contiguous()
        if self.prev_logprobs is not None:
            self.prev_logprobs = self.prev_logprobs.cpu().contiguous()
        if self.prev_values is not None:
            self.prev_values = self.prev_values.cpu().contiguous()
        if self.bootstrap_values is not None:
            self.bootstrap_values = self.bootstrap_values.cpu().contiguous()
        if self.intervene_flags is not None:
            self.intervene_flags = self.intervene_flags.cpu().contiguous()
        if self.forward_inputs:
            self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")
        if self.versions is not None:
            self.versions = self.versions.cpu().contiguous()

    @staticmethod
    def merge_rollout_results(
        rollout_results: list["RolloutResult"],
    ) -> "RolloutResult":
        def _merge_optional_tensor(field_name: str) -> torch.Tensor | None:
            values = [
                getattr(rollout_result, field_name)
                for rollout_result in rollout_results
            ]
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                raise ValueError(
                    f"Inconsistent field '{field_name}': some shards are None while others are tensors."
                )
            return torch.cat(values, dim=0)

        merged_actions = _merge_optional_tensor("actions")
        merged_prev_logprobs = _merge_optional_tensor("prev_logprobs")
        merged_prev_values = _merge_optional_tensor("prev_values")
        merged_bootstrap_values = _merge_optional_tensor("bootstrap_values")
        merged_intervene_flags = _merge_optional_tensor("intervene_flags")
        merged_versions = _merge_optional_tensor("versions")

        forward_inputs_list = [
            rollout_result.forward_inputs for rollout_result in rollout_results
        ]
        if all(not forward_inputs for forward_inputs in forward_inputs_list):
            merged_forward_inputs = {}
        else:
            merged_forward_inputs = cat_list_of_dict_tensor(forward_inputs_list)

        return RolloutResult(
            actions=merged_actions,
            prev_logprobs=merged_prev_logprobs,
            prev_values=merged_prev_values,
            bootstrap_values=merged_bootstrap_values,
            intervene_flags=merged_intervene_flags,
            forward_inputs=merged_forward_inputs,
            versions=merged_versions,
        )


@dataclass(kw_only=True)
class ChunkStepResult:
    """Model outputs, env outputs (without observations), and training forward inputs for a chunk step."""

    actions: torch.Tensor = None  # [B, action_dim]
    prev_logprobs: torch.Tensor = None  # [B, action_dim]
    prev_values: torch.Tensor = None  # [B, 1]
    dones: torch.Tensor = None  # [B, 1]
    truncations: torch.Tensor = None  # [B, 1]
    terminations: torch.Tensor = None  # [B, 1]
    rewards: torch.Tensor = None  # [B, 1]
    forward_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    versions: torch.Tensor = None  # [B, 1]

    def __post_init__(self):
        if self.actions is not None:
            self.actions = self.actions.cpu().contiguous()
        if self.prev_logprobs is not None:
            self.prev_logprobs = self.prev_logprobs.cpu().contiguous()
        if self.prev_values is not None:
            self.prev_values = self.prev_values.cpu().contiguous()
        if self.dones is not None:
            self.dones = self.dones.cpu().contiguous()
        if self.terminations is not None:
            self.terminations = self.terminations.cpu().contiguous()
        if self.truncations is not None:
            self.truncations = self.truncations.cpu().contiguous()
        if self.rewards is not None:
            self.rewards = self.rewards.cpu().contiguous()
        if self.forward_inputs:
            self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")
        if self.versions is not None:
            self.versions = self.versions.cpu().contiguous()


@dataclass
class Trajectory:
    """
    trajectory contains multiple episodes.
    """

    max_episode_length: int = 0  # max episode length
    model_weights_id: str = ""  # str(uuid(versions))
    actions: torch.Tensor = None
    intervene_flags: torch.Tensor = None
    rewards: torch.Tensor = None
    terminations: torch.Tensor = None
    truncations: torch.Tensor = None
    dones: torch.Tensor = None
    prev_logprobs: torch.Tensor = None
    prev_values: torch.Tensor = None
    versions: torch.Tensor = None
    forward_inputs: dict[str, Any] = field(default_factory=dict)

    curr_obs: dict[str, Any] = field(default_factory=dict)
    next_obs: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _generate_field_mask(
        ref_tensor: torch.Tensor, mask: torch.Tensor, traj_len: int
    ) -> torch.Tensor:
        """
        Generate a mask for terminations/truncations/dones based on their original shape.
        """
        assert mask.dim() == 1, f"Expected 1D mask, got {mask.shape=}"
        if ref_tensor.shape[0] == traj_len:
            return mask
        elif ref_tensor.shape[0] > traj_len:
            extra = int(ref_tensor.shape[0] - traj_len)
            assert traj_len % extra == 0, (
                f"Trajectory length {traj_len} is not divisible by extra {extra} for terminations/truncations/dones"
            )
            epoch_len = traj_len // extra

            field_mask = torch.zeros(
                ref_tensor.shape[0], dtype=torch.bool, device=mask.device
            )
            original_indices = torch.arange(ref_tensor.shape[0], device=mask.device)
            epoch_idx = original_indices // (epoch_len + 1)
            step_idx = original_indices % (epoch_len + 1)

            # Keep the first position of each epoch (step_idx == 0)
            field_mask[step_idx == 0] = True

            # Map positions with step_idx >= 1 to mask
            valid_mask = step_idx >= 1
            mask_idx = epoch_idx[valid_mask] * epoch_len + (step_idx[valid_mask] - 1)
            valid_original_indices = original_indices[valid_mask]
            valid_mask_idx = mask_idx < len(mask)
            field_mask[valid_original_indices[valid_mask_idx]] = mask[
                mask_idx[valid_mask_idx]
            ].to(dtype=torch.bool)

            return field_mask
        else:
            raise ValueError(
                f"Reference tensor length {ref_tensor.shape[0]} < traj_len {traj_len}"
            )

    def extract_intervene_traj(self, mode="any"):
        if self.intervene_flags is None or (~self.intervene_flags).all():
            return None

        if mode == "any":
            mask = self.intervene_flags.any(dim=-1)
        elif mode == "all":
            mask = self.intervene_flags.all(dim=-1)
        else:
            raise NotImplementedError(
                f"Unsupported extract_intervene_traj mode: {mode}"
            )
        assert mask.dim() == 2, (
            f"Expected 2D mask after processing (traj len, bsz), got {mask.shape=}"
        )
        traj_len = int(mask.shape[0])

        def apply_mask(tensor, i):
            return tensor[:, i][mask[:, i]].unsqueeze(1) if tensor is not None else None

        def apply_mask_to_dict(d, i):
            return (
                {k: v[:, i][mask[:, i]].unsqueeze(1) for k, v in d.items()} if d else {}
            )

        filtered_trajectories = []
        for i in range(mask.shape[1]):
            if not mask[:, i].any():
                continue

            actions = apply_mask(self.actions, i)
            rewards = apply_mask(self.rewards, i)
            prev_logprobs = apply_mask(self.prev_logprobs, i)
            prev_values = apply_mask(self.prev_values, i)
            intervene_flags = apply_mask(self.intervene_flags, i)

            forward_inputs = apply_mask_to_dict(self.forward_inputs, i)
            curr_obs = apply_mask_to_dict(self.curr_obs, i)
            next_obs = apply_mask_to_dict(self.next_obs, i)

            terminations = truncations = dones = None
            if self.terminations is not None:
                field_mask = self._generate_field_mask(
                    self.terminations[:, i : i + 1], mask[:, i], traj_len
                )
                terminations = self.terminations[:, i : i + 1][field_mask]
                truncations = self.truncations[:, i : i + 1][field_mask]
                dones = self.dones[:, i : i + 1][field_mask]

            filtered_trajectories.append(
                Trajectory(
                    max_episode_length=self.max_episode_length,
                    model_weights_id=self.model_weights_id,
                    actions=actions,
                    intervene_flags=intervene_flags,
                    rewards=rewards,
                    terminations=terminations,
                    truncations=truncations,
                    dones=dones,
                    prev_logprobs=prev_logprobs,
                    prev_values=prev_values,
                    forward_inputs=forward_inputs,
                    curr_obs=curr_obs,
                    next_obs=next_obs,
                )
            )

        return filtered_trajectories if filtered_trajectories else None


@dataclass(kw_only=True)
class EmbodiedRolloutResult:
    """
    Collect chunk-step results and transitions during rollout,
    and convert them into trajectory tensors.
    """

    max_episode_length: int = 0

    actions: list[torch.Tensor] = field(default_factory=list)  # trajectory_length
    intervene_flags: list[torch.Tensor] = field(
        default_factory=list
    )  # trajectory_length
    rewards: list[torch.Tensor] = field(default_factory=list)  # trajectory_length
    terminations: list[torch.Tensor] = field(
        default_factory=list
    )  # trajectory_length + rollout_epoch
    truncations: list[torch.Tensor] = field(
        default_factory=list
    )  # trajectory_length + rollout_epoch
    dones: list[torch.Tensor] = field(
        default_factory=list
    )  # trajectory_length + rollout_epoch
    prev_logprobs: list[torch.Tensor] = field(default_factory=list)  # trajectory_length
    prev_values: list[torch.Tensor] = field(
        default_factory=list
    )  # trajectory_length + rollout_epoch
    versions: list[torch.Tensor] = field(default_factory=list)  # trajectory_length
    forward_inputs: list[dict[str, Any]] = field(
        default_factory=list
    )  # trajectory_length

    curr_obs: list[dict[str, Any]] = field(default_factory=list)  # trajectory_length
    next_obs: list[dict[str, Any]] = field(default_factory=list)  # trajectory_length

    def append_step_result(self, result: ChunkStepResult):
        if result.actions is not None:
            self.actions.append(result.actions)
            self.intervene_flags.append(
                torch.zeros_like(result.actions, dtype=torch.bool)
            )
        if result.rewards is not None:
            self.rewards.append(result.rewards)
        if result.terminations is not None:
            self.terminations.append(result.terminations)
        if result.truncations is not None:
            self.truncations.append(result.truncations)
        if result.dones is not None:
            self.dones.append(result.dones)
        if result.prev_logprobs is not None:
            self.prev_logprobs.append(result.prev_logprobs)
        if result.prev_values is not None:
            self.prev_values.append(result.prev_values)
        if result.versions is not None:
            self.versions.append(result.versions)
        if result.forward_inputs:
            self.forward_inputs.append(result.forward_inputs)

    def mark_last_step_with_intervene_flags(self, intervene_flags: torch.Tensor):
        if not self.intervene_flags:
            return

        if intervene_flags.dim() == 1:
            intervene_flags = intervene_flags[:, None]
        assert intervene_flags.dim() == 2, (
            f"Expected 2D tensor, got {intervene_flags.shape=}"
        )

        last_action = self.actions[-1]
        bsz, num_action_chunks = intervene_flags.shape
        expanded_flags = intervene_flags.reshape(bsz, num_action_chunks, 1).expand_as(
            last_action.reshape(bsz, num_action_chunks, -1)
        )
        self.intervene_flags[-1] = expanded_flags.reshape(bsz, -1).to(torch.bool)

    def update_last_actions(
        self, intervene_actions: torch.Tensor, intervene_flags: torch.Tensor
    ):
        # action: [bsz, num-chunk-size x action-dim]
        # intervene_actions: [bsz, num-chunk-size x action-dim]
        # intervene_flags: [bsz, num-chunk-size]

        if self.actions and len(self.actions) > 0:
            last_action = self.actions[-1]
            assert last_action.dim() == 2, (
                f"Expected 2D tensor, got {last_action.shape=}"
            )
            assert intervene_actions.dim() == 2, (
                f"Expected 2D tensor, got {intervene_actions.shape=}"
            )

            # Normalize intervene_flags dimensions
            if intervene_flags.dim() == 1:
                intervene_flags = intervene_flags[:, None]
            assert intervene_flags.dim() == 2, (
                f"Expected 2D tensor, got {intervene_flags.shape=}"
            )

            bsz, num_action_chunks = intervene_flags.shape[:2]
            flags = intervene_flags.reshape(-1, num_action_chunks, 1)

            # Combine intervene_actions and last_action based on flags
            last_full_action = intervene_actions.reshape(
                bsz, num_action_chunks, -1
            ) * flags + last_action.reshape(bsz, num_action_chunks, -1) * (~flags)
            self.actions[-1] = last_full_action.reshape(bsz, -1)

            full_flags = flags.expand_as(last_full_action).reshape(bsz, -1)
            self.intervene_flags[-1] = full_flags

            if self.forward_inputs:
                last_fi = self.forward_inputs[-1]
                if "action" in last_fi:
                    last_fi["action"] = (
                        last_full_action.reshape(bsz, -1).cpu().contiguous()
                    )
                last_fi.pop("model_action", None)

    def append_transitions(self, curr_obs=None, next_obs=None):
        assert curr_obs is not None and next_obs is not None
        if "task_descriptions" in curr_obs:
            curr_obs.pop("task_descriptions")
        if "task_descriptions" in next_obs:
            next_obs.pop("task_descriptions")
        self.curr_obs.append(curr_obs)
        self.next_obs.append(next_obs)

    def clear(self):
        self.actions.clear()
        self.intervene_flags.clear()
        self.rewards.clear()
        self.terminations.clear()
        self.truncations.clear()
        self.dones.clear()
        self.prev_logprobs.clear()
        self.prev_values.clear()
        self.versions.clear()
        self.forward_inputs.clear()
        self.curr_obs.clear()
        self.next_obs.clear()

    def to_trajectory(self) -> Trajectory:
        # return [trajectory_length, B, ...]
        trajectory = Trajectory(
            max_episode_length=self.max_episode_length,
        )
        if len(self.actions) > 0:
            trajectory.actions = torch.stack(self.actions, dim=0).cpu().contiguous()
        if len(self.intervene_flags) > 0:
            trajectory.intervene_flags = (
                torch.stack(self.intervene_flags, dim=0).cpu().contiguous()
            )
        if len(self.rewards) > 0:
            trajectory.rewards = torch.stack(self.rewards, dim=0).cpu().contiguous()
        if len(self.terminations) > 0:
            trajectory.terminations = (
                torch.stack(self.terminations, dim=0).cpu().contiguous()
            )
        if len(self.truncations) > 0:
            trajectory.truncations = (
                torch.stack(self.truncations, dim=0).cpu().contiguous()
            )
        if len(self.dones) > 0:
            trajectory.dones = torch.stack(self.dones, dim=0).cpu().contiguous()
        if len(self.prev_logprobs) > 0:
            trajectory.prev_logprobs = (
                torch.stack(self.prev_logprobs, dim=0).cpu().contiguous()
            )
        if len(self.prev_values) > 0:
            trajectory.prev_values = (
                torch.stack(self.prev_values, dim=0).cpu().contiguous()
            )
        if len(self.versions) > 0:
            trajectory.versions = torch.stack(self.versions, dim=0).cpu().contiguous()
        if len(self.forward_inputs) > 0:
            trajectory.forward_inputs = stack_list_of_dict_tensor(self.forward_inputs)
            for key in trajectory.forward_inputs.keys():
                trajectory.forward_inputs[key] = (
                    trajectory.forward_inputs[key].cpu().contiguous()
                )

        if len(self.curr_obs) > 0:
            trajectory.curr_obs = stack_list_of_dict_tensor(self.curr_obs)
            for key in trajectory.curr_obs.keys():
                trajectory.curr_obs[key] = trajectory.curr_obs[key].cpu().contiguous()
        if len(self.next_obs) > 0:
            trajectory.next_obs = stack_list_of_dict_tensor(self.next_obs)
            for key in trajectory.next_obs.keys():
                trajectory.next_obs[key] = trajectory.next_obs[key].cpu().contiguous()

        trajectory.model_weights_id = get_model_weights_id(
            trajectory.versions
            if trajectory.versions is not None
            else torch.zeros(1, dtype=torch.float32)
        )

        return trajectory

    def to_splited_trajectories(self, split_size: int) -> list[Trajectory]:
        all_trajectory: Trajectory = self.to_trajectory()
        splited_trajectories: list[Trajectory] = [
            Trajectory() for _ in range(split_size)
        ]

        if len(all_trajectory.curr_obs) > 0:
            splited_obs = split_dict_to_chunk(
                all_trajectory.curr_obs, split_size, dim=1
            )
            for i in range(split_size):
                splited_trajectories[i].curr_obs = splited_obs[i]
        if len(all_trajectory.next_obs) > 0:
            splited_obs = split_dict_to_chunk(
                all_trajectory.next_obs, split_size, dim=1
            )
            for i in range(split_size):
                splited_trajectories[i].next_obs = splited_obs[i]

        if (
            all_trajectory.forward_inputs is not None
            and len(all_trajectory.forward_inputs) > 0
        ):
            splited_forward_inputs = split_dict_to_chunk(
                all_trajectory.forward_inputs, split_size, dim=1
            )
            for i in range(split_size):
                splited_trajectories[i].forward_inputs = splited_forward_inputs[i]

        for field_name in all_trajectory.__dataclass_fields__.keys():
            value = getattr(all_trajectory, field_name)

            if value is None or isinstance(value, dict):
                continue

            if isinstance(value, int) or isinstance(value, str):
                for i in range(split_size):
                    setattr(splited_trajectories[i], field_name, value)
                continue
            elif isinstance(value, torch.Tensor):
                chunks = torch.chunk(value, split_size, dim=1)
                for i in range(split_size):
                    setattr(splited_trajectories[i], field_name, chunks[i].contiguous())
            else:
                raise ValueError(
                    f"Unsupported value type: {type(value)} for field_name: {field_name}"
                )

        del all_trajectory
        return splited_trajectories

    def to_splited_trajectories_by_sizes(
        self, split_sizes: list[int]
    ) -> list[Trajectory]:
        trajectory = self.to_trajectory()
        trajectories = [Trajectory() for _ in split_sizes]

        for field_name in trajectory.__dataclass_fields__:
            value = getattr(trajectory, field_name)
            if value is None:
                continue
            if isinstance(value, (int, str)):
                for split_trajectory in trajectories:
                    setattr(split_trajectory, field_name, value)
            elif isinstance(value, torch.Tensor):
                for split_trajectory, split_value in zip(
                    trajectories, torch.split(value, split_sizes, dim=1)
                ):
                    setattr(split_trajectory, field_name, split_value.contiguous())
            elif isinstance(value, dict):
                for split_trajectory, split_value in zip(
                    trajectories, split_dict(value, split_sizes, dim=1)
                ):
                    setattr(split_trajectory, field_name, split_value)
            else:
                raise ValueError(
                    f"Unsupported value type: {type(value)} for field_name: {field_name}"
                )

        return trajectories


@dataclass(kw_only=True)
class EmbodiedLerobotRolloutResult(EmbodiedRolloutResult):
    """Online LeRobot episode collector for embodied env workers.

    EnvWorker creates one collector per pipeline stage when
    ``algorithm.dagger.online_lerobot.enabled`` is set. Completed episodes are
    sent in memory to the actor for DAgger training; no manual instantiation is
    required.

    **Configuration**: Set ``algorithm.dagger.online_lerobot`` in the training
    yaml. EnvWorker and Actor both read that block directly. A minimal example
    is shown below; see
    ``examples/embodiment/config/libero_spatial_dagger_openpi_lerobot.yaml`` for
    a full reference config.

    Example yaml::

        algorithm:
          dagger:
            online_lerobot:
              enabled: true
              only_success: true
              robot_type: "panda"
              fps: 10
              finalize_interval: 8
              data_path: /path/to/shards
              rolling_lerobot_window_size: 50000
              enable_decoded_cache: true
              decoded_cache_capacity: 25000
              cache_ingest_mode: new_shards
              lerobot_num_workers: 0

    **Integration**: When online LeRobot collection is enabled, EnvWorker builds
    this class and the actor loads
    :class:`~rlinf.data.datasets.dagger.RollingLeRobotDataset`. Each training
    chunk calls :meth:`append_chunk_episode_data`; at the end of
    :meth:`interact`, completed episodes flow through
    :meth:`drain_episodes` → ``EnvWorker.send_lerobot_episodes`` →
    ``EmbodiedDAGGERFSDPPolicy.recv_lerobot_rollout_trajectories``. Do not
    enable ``env.train.data_collection`` on the same train env; that wrapper is
    for offline disk export only.

    **Responsibilities**:

    1. Accumulate per-env, in-progress episode frames from each
       ``env_interact_step`` call.
    2. Detect episode boundaries (termination / truncation), apply
       ``only_success`` filtering, and enqueue completed episodes.
    3. Preserve unfinished episodes across multiple ``interact`` rounds until
       each parallel env finishes its current episode.
    4. Optionally retain per-chunk ``rewards`` for history-buffer reward-model
       back-propagation (see :meth:`append_step_result`).

    **Episode semantics**: Frame construction follows the same rules as offline
    :class:`CollectEpisode`.

    1. Auto-reset envs: ``final_observation`` is attributed to the finished
       episode; post-reset observations are carried via ``_pending_obs``.
    2. DAgger intervention: ``RolloutResult.intervene_flags`` and expert actions
       override recorded actions and set ``intervene_flag``.
    3. Real-world hooks: ``record_reset``, ``pre_record``, and
       ``segment_advance`` info flags are honored.
    4. ``only_success=True`` (from
       ``algorithm.dagger.online_lerobot.only_success``): only successful
       terminations are exported; failed episodes are discarded.

    Each exported frame is a ``dict`` compatible with
    ``LeRobotDatasetWriter.add_episode`` and
    :class:`~rlinf.data.datasets.dagger.RollingLeRobotDataset`, with fields such
    as ``state``, ``actions``, ``task``, ``image``, ``intervene_flag``,
    ``segment_id``, ``is_success``, and ``done``.

    **Runtime lifecycle inside EnvWorker**:

    1. First ``interact`` round: create one collector per pipeline stage.
    2. Later ``interact`` rounds: ``rewards.clear()`` only; episode buffers
       persist until each env finishes its episode.
    3. Every chunk: ``append_chunk_episode_data(rollout_result=..., **payload)``.
    4. End of ``interact``: ``drain_episodes()`` and send to the actor.
    5. Non auto-reset bootstrap ``env.reset()``: :meth:`reset_episode_buffers`.

    **Offline vs online collection**: Offline collection uses
    ``env.train.data_collection.enabled`` and writes shards to disk via
    :class:`CollectEpisode` (for example ``collect_real_data.py``). Online
    collection uses ``algorithm.dagger.online_lerobot.enabled`` and sends
    episodes to the actor in memory (for example
    ``libero_spatial_dagger_openpi_lerobot.yaml``).

    **Differences from** :class:`EmbodiedRolloutResult`: This collector does not
    build PPO trajectories. :meth:`update_last_actions` and
    :meth:`append_transitions` are intentionally no-ops. Use
    :class:`EmbodiedRolloutResult` when online LeRobot collection is disabled.

    Args:
        max_episode_length: Inherited upper bound used by the parent class.
        num_envs: Number of parallel environments in this pipeline stage.
        only_success: If ``True``, export only episodes that terminate
            successfully; otherwise export every finished episode.

    """

    num_envs: int = 1
    only_success: bool = False
    num_action_chunks: int = 1
    action_dim: int = 7
    _env_buffers: list[list[dict[str, Any]]] = field(
        default_factory=list, init=False, repr=False
    )
    episodes: list[list[dict[str, Any]]] = field(
        default_factory=list, init=False, repr=False
    )
    _pending_obs: list[Any] = field(default_factory=list, init=False, repr=False)
    _pending_info: list[Any] = field(default_factory=list, init=False, repr=False)
    _segment_ids: list[int] = field(default_factory=list, init=False, repr=False)
    _episode_success: list[bool] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self._reset_episode_state()

    def append_step_result(self, result: ChunkStepResult):
        if result.rewards is not None:
            self.rewards.append(result.rewards)

    def update_last_actions(
        self, intervene_actions: torch.Tensor, intervene_flags: torch.Tensor
    ):
        return

    def append_transitions(self, curr_obs=None, next_obs=None):
        return

    def reset_episode_buffers(self) -> None:
        """Reset in-progress episode state, e.g. after an env ``reset()``."""
        self._env_buffers = [[] for _ in range(self.num_envs)]
        self._pending_obs = [None] * self.num_envs
        self._pending_info = [None] * self.num_envs
        self._segment_ids = [0] * self.num_envs
        self._episode_success = [False] * self.num_envs

    def _reset_episode_state(self) -> None:
        self.reset_episode_buffers()
        self.episodes = []

    @staticmethod
    def _to_numpy(data):
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr
        return (
            (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        )

    @staticmethod
    def _expand_multi_view_images(
        base_key: str, arr: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        if arr is None:
            return {}
        if arr.ndim == 3:
            return {base_key: arr}
        if arr.ndim == 4:
            if arr.shape[0] == 1:
                return {base_key: arr[0]}
            return {f"{base_key}-{i}": arr[i] for i in range(arr.shape[0])}
        return {base_key: arr}

    @staticmethod
    def _slice_data(data, env_idx: int, num_envs: int):
        if isinstance(data, torch.Tensor):
            return (
                data[env_idx] if data.dim() > 0 and data.shape[0] == num_envs else data
            )
        if isinstance(data, np.ndarray):
            return (
                data[env_idx] if data.ndim > 0 and data.shape[0] == num_envs else data
            )
        if isinstance(data, dict):
            return {
                k: EmbodiedLerobotRolloutResult._slice_data(v, env_idx, num_envs)
                for k, v in data.items()
            }
        if isinstance(data, list):
            return data[env_idx] if len(data) == num_envs else data
        return data

    @staticmethod
    def _scalar_flag(flags, env_idx: int, num_envs: int) -> bool:
        if isinstance(flags, torch.Tensor):
            if flags.dim() > 1:
                return bool(flags[env_idx, -1].item())
            if flags.dim() == 1 and flags.shape[0] == num_envs:
                return bool(flags[env_idx].item())
            return bool(flags.item())
        if isinstance(flags, np.ndarray):
            if flags.ndim > 0 and flags.shape[0] == num_envs:
                return bool(flags[env_idx])
            return bool(flags)
        return bool(flags)

    @staticmethod
    def _extract_obs_image_state(obs):
        if not isinstance(obs, dict):
            return None, None, None, None, "unknown task"
        image = obs.get("main_images", obs.get("image", obs.get("full_image")))
        wrist_image = obs.get("wrist_images", obs.get("wrist_image"))
        extra_view_image = obs.get("extra_view_images", obs.get("extra_view_image"))
        state = obs.get("states", obs.get("state"))
        task = obs.get("task_descriptions", "unknown task")
        if isinstance(task, (list, tuple)):
            task = task[0] if task else "unknown task"
        return (
            EmbodiedLerobotRolloutResult._to_numpy(image),
            EmbodiedLerobotRolloutResult._to_numpy(wrist_image),
            EmbodiedLerobotRolloutResult._to_numpy(extra_view_image),
            EmbodiedLerobotRolloutResult._to_numpy(state),
            str(task),
        )

    @staticmethod
    def _bool_from_env_info(env_info: Any, key: str) -> bool:
        if not isinstance(env_info, dict) or env_info.get(key) is None:
            return False
        return bool(np.asarray(env_info[key]).any())

    @staticmethod
    def _extract_success_from_info(info: Any) -> bool | None:
        if not isinstance(info, dict):
            return None
        for src in (info.get("episode"), info):
            if not isinstance(src, dict):
                continue
            for key in ("success_once", "success_at_end", "success"):
                val = src.get(key)
                if val is None:
                    continue
                arr = EmbodiedLerobotRolloutResult._to_numpy(val)
                if arr is not None:
                    return bool(np.asarray(arr).reshape(-1).any())
        return None

    @staticmethod
    def _extract_success(info: Any) -> bool:
        success = EmbodiedLerobotRolloutResult._extract_success_from_info(info)
        return bool(success) if success is not None else False

    @staticmethod
    def _intervene_flag_from_info(info: Any) -> bool:
        if not isinstance(info, dict):
            return False
        val = info.get("intervene_flag")
        if val is None:
            return False
        arr = EmbodiedLerobotRolloutResult._to_numpy(val)
        if arr is None:
            return False
        return bool(np.asarray(arr, dtype=bool).reshape(-1).any())

    @staticmethod
    def _reshape_chunk_actions(
        actions,
        *,
        num_envs: int,
        num_chunks: int,
        action_dim: int,
    ) -> np.ndarray | None:
        """Match ``CollectEpisode.chunk_step``: [B,C,D] per-step actions."""
        if actions is None:
            return None
        arr = EmbodiedLerobotRolloutResult._to_numpy(actions)
        batch_size = arr.shape[0]
        flat_dim = num_chunks * action_dim
        if arr.ndim == 3:
            return arr
        if arr.ndim == 2:
            if arr.shape[-1] == flat_dim:
                return arr.reshape(batch_size, num_chunks, action_dim)
            if arr.shape[-1] == action_dim:
                return arr[:, None, :]
        raise ValueError(
            f"Unexpected chunk action shape {arr.shape}; expected "
            f"[{num_envs}, {num_chunks}, {action_dim}] or flat dim {flat_dim}."
        )

    @staticmethod
    def _normalize_intervene_in_info(env_info: Any, action_dim: int) -> None:
        """Port of ``CollectEpisode._record_step`` intervene reshape (per env)."""
        if not isinstance(env_info, dict) or "intervene_action" not in env_info:
            return
        intervene_action = EmbodiedLerobotRolloutResult._to_numpy(
            env_info["intervene_action"]
        )
        if intervene_action.size <= action_dim:
            env_info["intervene_action"] = intervene_action.reshape(-1)[:action_dim]
            return
        chunk_size = intervene_action.reshape(-1, action_dim).shape[0]
        env_info["intervene_action"] = intervene_action.reshape(-1, action_dim)[-1]
        if "intervene_flag" in env_info:
            intervene_flag = EmbodiedLerobotRolloutResult._to_numpy(
                env_info["intervene_flag"]
            )
            env_info["intervene_flag"] = intervene_flag.reshape(chunk_size, -1)[-1, 0]

    @staticmethod
    def _inject_expert_into_step_info(
        step_info: dict,
        expert_actions: np.ndarray,
        intervene_flags,
        *,
        step_idx: int,
        step_term,
        step_trunc,
    ) -> None:
        """Port of pre-refactor ``CollectEpisode.chunk_step`` expert injection."""
        if (
            expert_actions is None
            or intervene_flags is None
            or "intervene_action" in step_info
        ):
            return
        step_intervene_flags = intervene_flags[:, step_idx]
        step_expert = expert_actions[:, step_idx]
        if "final_info" in step_info:
            step_info["final_info"]["intervene_action"] = expert_actions
            step_info["final_info"]["intervene_flag"] = intervene_flags
            step_info["intervene_action"] = step_expert
            term = EmbodiedLerobotRolloutResult._to_numpy(step_term)
            trunc = EmbodiedLerobotRolloutResult._to_numpy(step_trunc)
            step_info["intervene_flag"] = step_intervene_flags & ~(term | trunc)
        else:
            step_info["intervene_action"] = step_expert
            step_info["intervene_flag"] = step_intervene_flags

    def _update_episode_success(self, env_idx: int, env_info: Any) -> None:
        success = self._extract_success_from_info(env_info)
        if success is not None:
            self._episode_success[env_idx] = self._episode_success[env_idx] or success

    def _get_episode_success(self, env_idx: int) -> bool:
        if self._episode_success[env_idx]:
            return True
        found_any = False
        is_success = False
        for frame in self._env_buffers[env_idx]:
            frame_success = frame.get("_frame_success")
            if frame_success is not None:
                found_any = True
                is_success = is_success or bool(frame_success)
        if found_any:
            return is_success
        return self._episode_success[env_idx]

    def _resolve_step_obs_info(
        self,
        *,
        step_obs: Any,
        step_info: Any,
        env_idx: int,
        env_done: bool,
    ) -> tuple[Any, Any]:
        has_final_obs = isinstance(step_info, dict) and "final_observation" in step_info
        if has_final_obs and env_done:
            final_observation = step_info["final_observation"]
            final_info_batch = step_info["final_info"]
            info_no_reset = copy.deepcopy(step_info)
            info_no_reset.pop("final_observation")
            info_no_reset.pop("final_info")
            env_obs = self._slice_data(final_observation, env_idx, self.num_envs)
            env_info = self._slice_data(final_info_batch, env_idx, self.num_envs)
            self._pending_obs[env_idx] = self._slice_data(
                step_obs, env_idx, self.num_envs
            )
            self._pending_info[env_idx] = self._slice_data(
                info_no_reset, env_idx, self.num_envs
            )
        else:
            env_obs = self._slice_data(step_obs, env_idx, self.num_envs)
            env_info = self._slice_data(step_info, env_idx, self.num_envs)
            if isinstance(env_info, dict):
                env_info = copy.deepcopy(env_info)
                env_info.pop("final_observation", None)
                env_info.pop("final_info", None)
        return env_obs, env_info

    def _consume_pending_obs(self, env_idx: int) -> tuple[Any | None, Any | None]:
        pending_obs = self._pending_obs[env_idx]
        pending_info = self._pending_info[env_idx]
        self._pending_obs[env_idx] = None
        self._pending_info[env_idx] = None
        return pending_obs, pending_info

    def _reset_env_buffer(self, env_idx: int) -> None:
        self._env_buffers[env_idx] = []
        self._episode_success[env_idx] = False
        self._segment_ids[env_idx] = 0
        pending_obs, pending_info = self._consume_pending_obs(env_idx)
        if pending_obs is not None:
            self._env_buffers[env_idx].append(
                {
                    "_pending_seed": True,
                    "obs": pending_obs,
                    "info": pending_info if pending_info is not None else {},
                }
            )

    def _flush_env_episode(self, env_idx: int, is_success: bool) -> None:
        buf = self._env_buffers[env_idx]
        frames = [entry for entry in buf if not entry.get("_pending_seed")]
        if not frames:
            return
        for frame_entry in frames:
            frame_entry.pop("_frame_success", None)
            frame_entry["is_success"] = np.array([is_success], dtype=bool)
        frames[-1]["done"] = np.array([True], dtype=bool)
        self.episodes.append(frames)

    def _maybe_flush_env(
        self,
        env_idx: int,
        *,
        done_by_term: bool,
        done_by_trunc: bool,
    ) -> None:
        is_success = self._get_episode_success(env_idx)
        if self.only_success:
            if is_success and done_by_term:
                self._flush_env_episode(env_idx, is_success)
                self._reset_env_buffer(env_idx)
            elif done_by_trunc or done_by_term:
                self._reset_env_buffer(env_idx)
        elif done_by_trunc or done_by_term:
            self._flush_env_episode(env_idx, is_success)
            self._reset_env_buffer(env_idx)

    def append_chunk_episode_data(
        self,
        *,
        rollout_result: RolloutResult,
        chunk_actions,
        obs_list,
        terminations,
        truncations,
        infos_list,
    ) -> None:
        """Record per-step LeRobot frames from one env chunk interaction."""
        chunk_size = len(obs_list) if isinstance(obs_list, (list, tuple)) else 1
        num_envs = self.num_envs
        num_chunks = self.num_action_chunks
        action_dim = self.action_dim

        actions_arr = self._reshape_chunk_actions(
            chunk_actions,
            num_envs=num_envs,
            num_chunks=num_chunks,
            action_dim=action_dim,
        )
        intervene_flags = rollout_result.intervene_flags
        if intervene_flags is not None:
            intervene_flags = self._to_numpy(intervene_flags).reshape(
                num_envs, num_chunks
            )
        expert_actions = rollout_result.forward_inputs.get("action", None)
        if expert_actions is not None:
            expert_actions = self._reshape_chunk_actions(
                expert_actions,
                num_envs=num_envs,
                num_chunks=num_chunks,
                action_dim=action_dim,
            )

        for step_idx in range(chunk_size):
            step_obs = (
                obs_list[step_idx] if isinstance(obs_list, (list, tuple)) else obs_list
            )
            step_term = (
                terminations[:, step_idx]
                if getattr(terminations, "ndim", 1) > 1
                else terminations
            )
            step_trunc = (
                truncations[:, step_idx]
                if getattr(truncations, "ndim", 1) > 1
                else truncations
            )
            step_info = (
                copy.deepcopy(infos_list[step_idx])
                if isinstance(infos_list, (list, tuple))
                else copy.deepcopy(infos_list)
            )
            if isinstance(step_info, dict):
                self._inject_expert_into_step_info(
                    step_info,
                    expert_actions,
                    intervene_flags,
                    step_idx=step_idx,
                    step_term=step_term,
                    step_trunc=step_trunc,
                )

            for env_idx in range(num_envs):
                done_by_term = self._scalar_flag(step_term, env_idx, num_envs)
                done_by_trunc = self._scalar_flag(step_trunc, env_idx, num_envs)
                env_done = done_by_term or done_by_trunc
                env_obs, env_info = self._resolve_step_obs_info(
                    step_obs=step_obs,
                    step_info=step_info,
                    env_idx=env_idx,
                    env_done=env_done,
                )

                if self._bool_from_env_info(env_info, "record_reset"):
                    self._env_buffers[env_idx] = []
                    self._episode_success[env_idx] = False
                    self._segment_ids[env_idx] = 0
                    self._pending_obs[env_idx] = None
                    self._pending_info[env_idx] = None
                    self._env_buffers[env_idx].append(
                        {
                            "_pending_seed": True,
                            "obs": env_obs,
                            "info": env_info if isinstance(env_info, dict) else {},
                        }
                    )
                    continue

                if self._bool_from_env_info(env_info, "pre_record"):
                    continue

                if self._bool_from_env_info(env_info, "segment_advance"):
                    self._segment_ids[env_idx] += 1

                frame_obs = env_obs
                frame_info = env_info
                buf = self._env_buffers[env_idx]
                if buf and buf[0].get("_pending_seed"):
                    seed = buf.pop(0)
                    frame_obs = seed["obs"]
                    frame_info = seed.get("info", {})

                image, wrist_image, extra_view_image, state, task = (
                    self._extract_obs_image_state(frame_obs)
                )
                if state is None or actions_arr is None:
                    continue

                if isinstance(frame_info, dict):
                    self._normalize_intervene_in_info(frame_info, action_dim)

                env_action = actions_arr[
                    env_idx, min(step_idx, actions_arr.shape[1] - 1)
                ]
                intervene_flag = False
                if isinstance(frame_info, dict) and (
                    "intervene_flag" in frame_info and "intervene_action" in frame_info
                ):
                    if self._intervene_flag_from_info(frame_info):
                        intervene_flag = True
                        env_action = self._to_numpy(frame_info["intervene_action"])

                if isinstance(task, str) and task == "unknown task":
                    _, _, _, _, task_from_info = self._extract_obs_image_state(
                        frame_obs
                    )
                    if task_from_info != "unknown task":
                        task = task_from_info

                frame: dict[str, Any] = {
                    "state": np.asarray(state).astype(np.float32),
                    "actions": np.asarray(env_action).astype(np.float32).flatten(),
                    "task": task,
                    "is_success": np.array([False], dtype=bool),
                    "done": np.array([False], dtype=bool),
                    "intervene_flag": np.array([intervene_flag], dtype=bool),
                    "segment_id": np.array(
                        [self._segment_ids[env_idx]], dtype=np.uint8
                    ),
                }
                if image is not None:
                    frame["image"] = self._to_uint8(np.asarray(image))
                for key, img in self._expand_multi_view_images(
                    "wrist_image", wrist_image
                ).items():
                    frame[key] = self._to_uint8(np.asarray(img))
                for key, img in self._expand_multi_view_images(
                    "extra_view_image", extra_view_image
                ).items():
                    frame[key] = self._to_uint8(np.asarray(img))

                step_success = self._extract_success_from_info(frame_info)
                if step_success is not None:
                    frame["_frame_success"] = step_success
                self._update_episode_success(env_idx, frame_info)
                self._env_buffers[env_idx].append(frame)

                if env_done:
                    self._maybe_flush_env(
                        env_idx,
                        done_by_term=done_by_term,
                        done_by_trunc=done_by_trunc,
                    )

    def drain_episodes(self) -> list[list[dict[str, Any]]]:
        """Return and clear completed episodes collected since the last drain."""
        episodes = self.episodes
        self.episodes = []
        return episodes

    def clear(self):
        super().clear()
        self._reset_episode_state()


def convert_trajectories_to_batch(
    trajectories: list[Trajectory],
) -> dict[str, torch.Tensor]:
    """
    convert a list of trajectories to a batch dict, the shape of the batch is [T, B, ...].
    """
    if not trajectories:
        return {}

    batch: dict[str, torch.Tensor] = {}

    # -------- obs / forward_inputs: dict[str, Tensor] --------
    if trajectories[0].curr_obs:
        all_keys: set[str] = set()
        for traj in trajectories:
            all_keys.update(traj.curr_obs.keys())
        batch["curr_obs"] = {}
        for key in all_keys:
            tensors = [
                traj.curr_obs[key] for traj in trajectories if key in traj.curr_obs
            ]
            if tensors:
                batch["curr_obs"][key] = torch.cat(tensors, dim=1)

    if trajectories[0].next_obs:
        all_keys: set[str] = set()
        for traj in trajectories:
            all_keys.update(traj.next_obs.keys())
        batch["next_obs"] = {}
        for key in all_keys:
            tensors = [
                traj.next_obs[key] for traj in trajectories if key in traj.next_obs
            ]
            if tensors:
                batch["next_obs"][key] = torch.cat(tensors, dim=1)

    if trajectories[0].forward_inputs:
        all_keys: set[str] = set()
        for traj in trajectories:
            all_keys.update(traj.forward_inputs.keys())
        batch["forward_inputs"] = {}
        for key in all_keys:
            tensors = [
                traj.forward_inputs[key]
                for traj in trajectories
                if key in traj.forward_inputs
            ]
            if tensors:
                batch["forward_inputs"][key] = torch.cat(tensors, dim=1)

    # -------- tensor fields --------
    reference_trajectory = trajectories[0]
    for field_name in reference_trajectory.__dataclass_fields__.keys():
        if not isinstance(getattr(reference_trajectory, field_name), torch.Tensor):
            continue
        field_list = [
            getattr(traj, field_name)
            for traj in trajectories
            if getattr(traj, field_name) is not None
        ]
        if field_list:
            batch[field_name] = torch.cat(field_list, dim=1)

    return batch
