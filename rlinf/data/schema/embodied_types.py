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

"""Embodied foundational data structures (env/step/trajectory types)."""

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from rlinf.utils.nested_dict_process import cat_list_of_dict_tensor, put_tensor_device


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
            final_obs_or_obs = [
                final_obs if final_obs is not None else env_output["obs"]
                for env_output, final_obs in zip(env_outputs, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        return EnvOutput(
            obs=merged_obs,
            final_obs=merged_final_obs,
            dones=_merge_optional_tensor_field("dones"),
            terminations=_merge_optional_tensor_field("terminations"),
            truncations=_merge_optional_tensor_field("truncations"),
            rewards=_merge_optional_tensor_field("rewards"),
            intervene_actions=_merge_optional_tensor_field(
                "intervene_actions", allow_partial_none=True, fill_value=0.0
            ),
            intervene_flags=_merge_optional_tensor_field(
                "intervene_flags", allow_partial_none=True, fill_value=False
            ),
            rlt_switch_flags=_merge_optional_tensor_field(
                "rlt_switch_flags", allow_partial_none=True, fill_value=False
            ),
        ).to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs": self.prepare_observations(self.obs),
            "final_obs": (
                self.prepare_observations(self.final_obs)
                if self.final_obs is not None
                else None
            ),
            "dones": self.dones,
            "terminations": self.terminations,
            "truncations": self.truncations,
            "rewards": self.rewards,
            "env_infos": self.env_infos,
            "intervene_actions": self.intervene_actions,
            "intervene_flags": self.intervene_flags,
            "rlt_switch_flags": self.rlt_switch_flags,
        }


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
class PolicyOutput:
    """Policy/rollout-worker outputs for one embodied communication round."""

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
    def merge(
        outputs: list["PolicyOutput"],
    ) -> "PolicyOutput":
        def _merge_optional_tensor(field_name: str) -> torch.Tensor | None:
            values = [getattr(output, field_name) for output in outputs]
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                raise ValueError(
                    f"Inconsistent field '{field_name}': some shards are None while others are tensors."
                )
            return torch.cat(values, dim=0)

        forward_inputs_list = [output.forward_inputs for output in outputs]
        merged_forward_inputs = (
            {}
            if all(not forward_inputs for forward_inputs in forward_inputs_list)
            else cat_list_of_dict_tensor(forward_inputs_list)
        )
        return PolicyOutput(
            actions=_merge_optional_tensor("actions"),
            prev_logprobs=_merge_optional_tensor("prev_logprobs"),
            prev_values=_merge_optional_tensor("prev_values"),
            bootstrap_values=_merge_optional_tensor("bootstrap_values"),
            intervene_flags=_merge_optional_tensor("intervene_flags"),
            forward_inputs=merged_forward_inputs,
            versions=_merge_optional_tensor("versions"),
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

    max_episode_length: int = 0
    model_weights_id: str = ""
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
        if ref_tensor.shape[0] > traj_len:
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
            field_mask[step_idx == 0] = True
            valid_mask = step_idx >= 1
            mask_idx = epoch_idx[valid_mask] * epoch_len + (step_idx[valid_mask] - 1)
            valid_original_indices = original_indices[valid_mask]
            valid_mask_idx = mask_idx < len(mask)
            field_mask[valid_original_indices[valid_mask_idx]] = mask[
                mask_idx[valid_mask_idx]
            ].to(dtype=torch.bool)
            return field_mask
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


def convert_trajectories_to_batch(
    trajectories: list[Trajectory],
) -> dict[str, torch.Tensor]:
    """Convert trajectory list into a `[T, B, ...]` batch dictionary."""
    if not trajectories:
        return {}

    batch: dict[str, torch.Tensor] = {}

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


__all__ = [
    "ChunkStepResult",
    "PolicyOutput",
    "EnvOutput",
    "RTCActionResponse",
    "RTCRequest",
    "Trajectory",
    "convert_trajectories_to_batch",
    "get_model_weights_id",
]
