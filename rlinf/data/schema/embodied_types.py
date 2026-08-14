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

import numpy as np
import torch

from rlinf.utils.nested_dict_process import put_tensor_device


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
    """Action-only response returned to an environment worker."""

    actions: torch.Tensor

    def __post_init__(self) -> None:
        if self.actions is not None:
            self.actions = self.actions.cpu().contiguous()


@dataclass(frozen=True)
class TrajectoryKey:
    """Identity of one action chunk produced by a logical environment source."""

    step_id: int
    epoch_id: int
    env_rank: int
    stage_id: int
    chunk_id: int


@dataclass(frozen=True)
class TrajectorySource:
    """Trajectory key and batch size carried by one routed source shard."""

    key: TrajectoryKey
    size: int
    offset: int = 0


@dataclass(kw_only=True)
class EnvResult:
    """Environment outcome associated with one policy output."""

    rewards: torch.Tensor | None = None
    dones: torch.Tensor | None = None
    terminations: torch.Tensor | None = None
    truncations: torch.Tensor | None = None
    intervene_actions: torch.Tensor | None = None
    intervene_flags: torch.Tensor | None = None
    rlt_switch_flags: torch.Tensor | None = None
    reward_model_output: torch.Tensor | None = None
    reward_assign_lengths: list[int] | None = None
    episode_data: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "rewards",
            "dones",
            "terminations",
            "truncations",
            "intervene_actions",
            "intervene_flags",
            "rlt_switch_flags",
            "reward_model_output",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, value.cpu().contiguous())

    @classmethod
    def from_env_output(
        cls,
        env_output: EnvOutput,
        reward_model_output: torch.Tensor | None = None,
    ) -> "EnvResult":
        """Build a transport result from an environment output."""
        return cls(
            rewards=env_output.rewards,
            dones=env_output.dones,
            terminations=env_output.terminations,
            truncations=env_output.truncations,
            intervene_actions=env_output.intervene_actions,
            intervene_flags=env_output.intervene_flags,
            rlt_switch_flags=env_output.rlt_switch_flags,
            reward_model_output=reward_model_output,
        )


@dataclass(kw_only=True)
class PolicyCompletion:
    """Environment outcome completed by a subsequent policy request."""

    sources: list[TrajectorySource]
    env_result: EnvResult
    next_obs: dict[str, Any]
    requires_inference: bool

    def __post_init__(self) -> None:
        self.next_obs = put_tensor_device(self.next_obs, "cpu")


@dataclass(kw_only=True)
class PolicyInput:
    """Policy inference input and optional preceding environment outcome."""

    obs: dict[str, Any]
    rlt_switch_flags: torch.Tensor | None = None
    intervene_flags: torch.Tensor | None = None
    sources: list[TrajectorySource] = field(default_factory=list)
    completions: list[PolicyCompletion | None] = field(default_factory=list)
    request_sizes: list[int] = field(default_factory=list)
    is_last: bool = False

    def __post_init__(self) -> None:
        self.obs = put_tensor_device(self.obs, "cpu")
        for name in ("rlt_switch_flags", "intervene_flags"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.cpu().contiguous())


@dataclass(kw_only=True)
class DummyPolicyInput(PolicyInput):
    """Policy request whose actions are supplied without model inference."""

    actions: torch.Tensor

    def __post_init__(self) -> None:
        """Move the request payload to CPU for transport."""
        super().__post_init__()
        self.actions = self.actions.cpu().contiguous()


@dataclass(kw_only=True)
class EmbodiedRolloutResult:
    """Policy inference data retained for trajectory construction."""

    actions: torch.Tensor
    forward_inputs: dict[str, Any]
    bootstrap_values: torch.Tensor | None = None
    prev_logprobs: torch.Tensor | None = None
    prev_values: torch.Tensor | None = None
    intervene_flags: torch.Tensor | None = None
    versions: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.actions = self.actions.cpu().contiguous()
        self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")
        for field_name in (
            "bootstrap_values",
            "prev_logprobs",
            "prev_values",
            "intervene_flags",
            "versions",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, value.cpu().contiguous())


def split_batch_value(value: Any, split_sizes: list[int]) -> list[Any]:
    """Split a recursively nested batch on its leading dimension."""
    if value is None:
        return [None] * len(split_sizes)
    if isinstance(value, torch.Tensor):
        return [chunk.contiguous() for chunk in torch.split(value, split_sizes, dim=0)]
    if isinstance(value, np.ndarray):
        return list(np.split(value, np.cumsum(split_sizes)[:-1], axis=0))
    if isinstance(value, dict):
        chunks = [{} for _ in split_sizes]
        for key, item in value.items():
            for chunk, split_item in zip(chunks, split_batch_value(item, split_sizes)):
                chunk[key] = split_item
        return chunks
    if isinstance(value, list):
        offset = 0
        chunks = []
        for size in split_sizes:
            chunks.append(value[offset : offset + size])
            offset += size
        return chunks
    if isinstance(value, (bool, float, int, str)):
        return [value] * len(split_sizes)
    raise TypeError(f"Unsupported batch value: {type(value)}")


def merge_batch_values(values: list[Any]) -> Any:
    """Merge recursively nested batches on their leading dimension."""
    if not values:
        raise ValueError("Cannot merge an empty list of batch values.")
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Cannot merge present and absent batch values.")

    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0)
    if isinstance(first, np.ndarray):
        return np.concatenate(values, axis=0)
    if isinstance(first, dict):
        if any(value.keys() != first.keys() for value in values[1:]):
            raise ValueError("Cannot merge batch dictionaries with different keys.")
        return {
            key: merge_batch_values([value[key] for value in values]) for key in first
        }
    if isinstance(first, list):
        return [item for value in values for item in value]
    if isinstance(first, (bool, float, int, str)):
        return values
    raise TypeError(f"Unsupported batch value: {type(first)}")


def split_episode_data(
    data: dict[str, Any] | None, split_sizes: list[int]
) -> list[dict[str, Any] | None]:
    """Split online LeRobot chunk data without splitting its time dimension."""
    if data is None:
        return [None] * len(split_sizes)

    def split_steps(values: list[Any]) -> list[list[Any]]:
        chunks = [[] for _ in split_sizes]
        for value in values:
            for chunk, split_item in zip(chunks, split_batch_value(value, split_sizes)):
                chunk.append(split_item)
        return chunks

    return [
        {
            "chunk_actions": chunk_actions,
            "obs_list": obs_list,
            "terminations": terminations,
            "truncations": truncations,
            "infos_list": infos_list,
        }
        for chunk_actions, obs_list, terminations, truncations, infos_list in zip(
            split_batch_value(data["chunk_actions"], split_sizes),
            split_steps(data["obs_list"]),
            split_batch_value(data["terminations"], split_sizes),
            split_batch_value(data["truncations"], split_sizes),
            split_steps(data["infos_list"]),
        )
    ]


def merge_episode_data(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge online LeRobot chunk data without merging its time dimension."""
    if not data:
        raise ValueError("Cannot merge an empty list of episode data.")

    def merge_steps(values: list[list[Any]]) -> list[Any]:
        lengths = {len(value) for value in values}
        if len(lengths) != 1:
            raise ValueError("Cannot merge episode data with different chunk lengths.")
        return [merge_batch_values(list(items)) for items in zip(*values)]

    return {
        "chunk_actions": merge_batch_values([value["chunk_actions"] for value in data]),
        "obs_list": merge_steps([value["obs_list"] for value in data]),
        "terminations": merge_batch_values([value["terminations"] for value in data]),
        "truncations": merge_batch_values([value["truncations"] for value in data]),
        "infos_list": merge_steps([value["infos_list"] for value in data]),
    }


def split_trajectory_sources(
    sources: list[TrajectorySource], split_sizes: list[int]
) -> list[list[TrajectorySource]]:
    """Split routed source metadata along the batch dimension."""
    if not sources:
        return [[] for _ in split_sizes]
    result = [[] for _ in split_sizes]
    source_index = 0
    source_offset = 0
    for result_index, split_size in enumerate(split_sizes):
        remaining = split_size
        while remaining:
            source = sources[source_index]
            take_size = min(remaining, source.size - source_offset)
            result[result_index].append(
                TrajectorySource(
                    key=source.key,
                    size=take_size,
                    offset=source.offset + source_offset,
                )
            )
            source_offset += take_size
            remaining -= take_size
            if source_offset == source.size:
                source_index += 1
                source_offset = 0
    return result


def split_env_result(env_result: EnvResult, split_sizes: list[int]) -> list[EnvResult]:
    """Split an environment result on its batch dimension."""
    fields = {
        name: split_batch_value(getattr(env_result, name), split_sizes)
        for name in env_result.__dataclass_fields__
        if name != "episode_data"
    }
    episodes = split_episode_data(env_result.episode_data, split_sizes)
    return [
        EnvResult(
            **{name: values[index] for name, values in fields.items()},
            episode_data=episodes[index],
        )
        for index in range(len(split_sizes))
    ]


def split_policy_input(
    policy_input: PolicyInput, split_sizes: list[int]
) -> list[PolicyInput]:
    """Split a policy input on its batch dimension."""
    if len(policy_input.completions) > 1:
        raise ValueError("A producer policy input cannot contain merged completions.")
    source_splits = split_trajectory_sources(policy_input.sources, split_sizes)
    rlt_splits = split_batch_value(policy_input.rlt_switch_flags, split_sizes)
    intervene_splits = split_batch_value(policy_input.intervene_flags, split_sizes)
    completion = policy_input.completions[0] if policy_input.completions else None
    if completion is None:
        completion_splits = [None] * len(split_sizes)
    else:
        completion_sources = split_trajectory_sources(completion.sources, split_sizes)
        env_results = split_env_result(completion.env_result, split_sizes)
        next_observations = split_batch_value(completion.next_obs, split_sizes)
        completion_splits = [
            PolicyCompletion(
                sources=completion_sources[index],
                env_result=env_results[index],
                next_obs=next_observations[index],
                requires_inference=completion.requires_inference,
            )
            for index in range(len(split_sizes))
        ]
    input_type = (
        DummyPolicyInput if isinstance(policy_input, DummyPolicyInput) else PolicyInput
    )
    action_splits = (
        split_batch_value(policy_input.actions, split_sizes)
        if isinstance(policy_input, DummyPolicyInput)
        else [None] * len(split_sizes)
    )
    return [
        input_type(
            obs=obs,
            rlt_switch_flags=rlt_splits[index],
            intervene_flags=intervene_splits[index],
            sources=source_splits[index],
            completions=[completion_splits[index]],
            request_sizes=[split_sizes[index]],
            is_last=policy_input.is_last,
            **(
                {"actions": action_splits[index]}
                if isinstance(policy_input, DummyPolicyInput)
                else {}
            ),
        )
        for index, obs in enumerate(split_batch_value(policy_input.obs, split_sizes))
    ]


def merge_policy_inputs(policy_inputs: list[PolicyInput]) -> PolicyInput:
    """Merge routed policy inputs in source order."""
    if not policy_inputs:
        raise ValueError("Cannot merge an empty list of policy inputs.")
    dummy_inputs = [
        policy_input
        for policy_input in policy_inputs
        if isinstance(policy_input, DummyPolicyInput)
    ]
    if dummy_inputs and len(dummy_inputs) != len(policy_inputs):
        raise ValueError("Cannot merge inferred and dummy policy inputs.")

    def get_batch_size(obs: dict[str, Any]) -> int:
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, (torch.Tensor, np.ndarray)):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from policy input observations.")

    observations = [policy_input.obs for policy_input in policy_inputs]

    def merge_optional_tensor(field_name: str) -> torch.Tensor | None:
        values = [getattr(item, field_name) for item in policy_inputs]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            reference = next(value for value in values if value is not None)
            values = [
                value
                if value is not None
                else torch.full(
                    (get_batch_size(obs), *reference.shape[1:]),
                    False,
                    dtype=reference.dtype,
                )
                for obs, value in zip(observations, values)
            ]
        return merge_batch_values(values)

    sources: list[TrajectorySource] = []
    for policy_input in policy_inputs:
        for source in policy_input.sources:
            if (
                sources
                and sources[-1].key == source.key
                and sources[-1].offset + sources[-1].size == source.offset
            ):
                sources[-1] = TrajectorySource(
                    key=source.key,
                    size=sources[-1].size + source.size,
                    offset=sources[-1].offset,
                )
            else:
                sources.append(source)

    input_type = DummyPolicyInput if dummy_inputs else PolicyInput
    return input_type(
        obs=merge_batch_values(observations),
        rlt_switch_flags=merge_optional_tensor("rlt_switch_flags"),
        intervene_flags=merge_optional_tensor("intervene_flags"),
        sources=sources,
        completions=[
            completion
            for policy_input in policy_inputs
            for completion in policy_input.completions
        ],
        request_sizes=[
            size
            for policy_input in policy_inputs
            for size in policy_input.request_sizes
        ],
        is_last=policy_inputs[0].is_last,
        **(
            {"actions": merge_batch_values([item.actions for item in dummy_inputs])}
            if dummy_inputs
            else {}
        ),
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
    "DummyPolicyInput",
    "EmbodiedRolloutResult",
    "EnvOutput",
    "EnvResult",
    "PolicyCompletion",
    "PolicyInput",
    "PolicyOutput",
    "TrajectoryKey",
    "TrajectorySource",
    "RTCActionResponse",
    "RTCRequest",
    "Trajectory",
    "convert_trajectories_to_batch",
    "get_model_weights_id",
    "merge_batch_values",
    "merge_episode_data",
    "merge_policy_inputs",
    "split_batch_value",
    "split_episode_data",
    "split_policy_input",
    "split_trajectory_sources",
]
