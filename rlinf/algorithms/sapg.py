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

from collections.abc import Callable
from dataclasses import dataclass

import torch


# SAPG utilities adapted from jayeshs999/sapg; see NOTICE for the MIT license.
def create_sinusoidal_encoding(
    values: torch.Tensor, dim: int, n: float = 100.0
) -> torch.Tensor:
    """Encode scalars into ``dim``-dimensional sinusoidal vectors.

    Args:
        values: 1-dimensional tensor of values to encode.
        dim: Output dimensionality, must be even.
        n: Base of the geometric frequency schedule.

    Returns:
        Tensor of shape ``[*values.shape, dim]``.
    """
    assert dim % 2 == 0, f"Sinusoidal encoding dim must be even, got {dim}."
    denom = n ** (
        2 * torch.arange(dim // 2, dtype=torch.float32, device=values.device) / dim
    )
    return torch.cat(
        [
            torch.sin(values.unsqueeze(-1) / denom),
            torch.cos(values.unsqueeze(-1) / denom),
        ],
        dim=-1,
    )


@dataclass
class SAPGExploration:
    """Per-environment exploration coefficients for SAPG.

    Attributes:
        embedding: ``[num_envs, embd_size]`` sinusoidal embedding concatenated
            onto each environment's observation.
        intrinsic_coef: ``[num_envs]`` per-block entropy coefficient.
        num_blocks: Number of exploration blocks (chunks).
        block_size: Number of environments per block.
    """

    embedding: torch.Tensor
    intrinsic_coef: torch.Tensor
    num_blocks: int
    block_size: int

    @property
    def embd_size(self) -> int:
        return self.embedding.shape[-1]

    @property
    def num_envs(self) -> int:
        return self.embedding.shape[0]


def build_sapg_exploration(
    num_envs: int,
    num_blocks: int,
    embd_size: int = 16,
    intrinsic_reward_type: str = "entropy",
    intrinsic_coef_scale: float = 1.0,
    intrinsic_coef_start: float = 0.5,
    intrinsic_coef_end: float = 0.0,
    embedding_start: float = 50.0,
    embedding_end: float = 0.0,
    encoding_base: float = 100.0,
    device: torch.device | str = "cpu",
) -> SAPGExploration:
    """Assign every environment an exploration coefficient and its embedding.

    Mirrors the reference implementation: environments are split into
    ``num_blocks`` contiguous blocks. The final block is the leader. The
    conditioning values and entropy coefficients decrease across blocks.

    Args:
        num_envs: Total number of parallel environments.
        num_blocks: Number of exploration chunks; must divide ``num_envs``.
        embd_size: Dimensionality of the sinusoidal embedding (even).
        intrinsic_reward_type: ``"entropy"`` for entropy regularization or
            ``"none"`` to disable it.
        intrinsic_coef_scale: Multiplier applied to the coefficient schedule.
        intrinsic_coef_start: Coefficient of the first block.
        intrinsic_coef_end: Coefficient of the last block.
        embedding_start: Value encoded for the first block.
        embedding_end: Value encoded for the last block.
        encoding_base: Base of the sinusoidal frequency schedule.
        device: Device for the returned tensors.

    Returns:
        A populated :class:`SAPGExploration`.
    """
    assert num_blocks >= 1, f"num_blocks must be >= 1, got {num_blocks}."
    assert num_envs % num_blocks == 0, (
        f"num_envs ({num_envs}) must be divisible by num_blocks ({num_blocks})."
    )
    block_size = num_envs // num_blocks
    env_block_ids = torch.arange(num_blocks, device=device).repeat_interleave(
        block_size
    )
    embedding_genvec = torch.linspace(
        embedding_start, embedding_end, num_blocks, device=device
    )[env_block_ids]
    embedding = create_sinusoidal_encoding(
        embedding_genvec, embd_size, n=encoding_base
    ).to(device)

    if intrinsic_reward_type == "entropy":
        intrinsic_coef = (
            torch.linspace(
                intrinsic_coef_start, intrinsic_coef_end, num_blocks, device=device
            )[env_block_ids]
            * intrinsic_coef_scale
        )
    elif intrinsic_reward_type == "none":
        intrinsic_coef = torch.zeros(num_envs, device=device)
    else:
        raise ValueError(
            f"Unsupported intrinsic_reward_type: {intrinsic_reward_type}. "
            "Expected 'entropy' or 'none'."
        )

    return SAPGExploration(
        embedding=embedding,
        intrinsic_coef=intrinsic_coef,
        num_blocks=num_blocks,
        block_size=block_size,
    )


def sample_repeat_idxs(
    num_blocks: int, off_policy_ratio: int, generator: torch.Generator | None = None
) -> list[int]:
    """Choose which blocks take part in one aggregation round.

    The unshifted copy (0) is always first. Positive shifts are sampled
    without replacement. Shift k relabels behaviour block k - 1 to leader.

    Args:
        num_blocks: Total number of exploration blocks.
        off_policy_ratio: Number of follower blocks to aggregate, clipped to
            the available number of followers.
        generator: Optional RNG for reproducibility.

    Returns:
        List of shifts, starting at 0.
    """
    num_repeat = min(num_blocks, int(off_policy_ratio) + 1)
    if num_repeat <= 1:
        return [0]
    follower_idxs = (
        torch.randperm(num_blocks - 1, generator=generator)[: num_repeat - 1] + 1
    )
    return [0] + follower_idxs.tolist()


def sapg_block_ids(
    states: torch.Tensor, block_embeddings: torch.Tensor
) -> torch.Tensor:
    """Read block IDs from the trailing exploration codes in states."""
    code = states[..., -block_embeddings.shape[-1] :]
    distance = (code.unsqueeze(-2) - block_embeddings.to(code)).square().sum(-1)
    return distance.argmin(dim=-1)


def sapg_follower_indices(
    block_ids: torch.Tensor, num_blocks: int, repeat_idxs: list[int]
) -> torch.Tensor:
    """Select the environments from follower blocks chosen by reference shifts.

    Each actor shard must contain equal numbers of environments per block.
    Equal counts keep the training batch size and FSDP calls aligned across ranks.
    """
    counts = torch.bincount(block_ids, minlength=num_blocks)
    if not torch.all(counts == counts[0]):
        raise ValueError(
            "Expected equal SAPG block counts in each actor shard, "
            f"found {counts.tolist()}."
        )
    selected = torch.zeros_like(block_ids, dtype=torch.bool)
    for shift in repeat_idxs[1:]:
        selected |= block_ids == shift - 1
    return selected.nonzero(as_tuple=True)[0]


def sapg_one_step_returns(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute the leader's one-step targets for follower transitions."""
    return rewards + gamma * next_values.masked_fill(dones, 0)


def sapg_entropy_coef_for_states(
    states: torch.Tensor,
    num_blocks: int,
    embd_size: int,
    intrinsic_coef_scale: float = 1.0,
    intrinsic_coef_start: float = 0.5,
    intrinsic_coef_end: float = 0.0,
    embedding_start: float = 50.0,
    embedding_end: float = 0.0,
    encoding_base: float = 100.0,
) -> torch.Tensor:
    """Get entropy coefficients for the policy conditions in the supplied states."""
    spec = build_sapg_exploration(
        num_envs=num_blocks,
        num_blocks=num_blocks,
        embd_size=embd_size,
        intrinsic_coef_scale=intrinsic_coef_scale,
        intrinsic_coef_start=intrinsic_coef_start,
        intrinsic_coef_end=intrinsic_coef_end,
        embedding_start=embedding_start,
        embedding_end=embedding_end,
        encoding_base=encoding_base,
        device=states.device,
    )
    return spec.intrinsic_coef[sapg_block_ids(states, spec.embedding)].to(states.dtype)


def relabel_sapg_batch(
    batch: dict,
    followers: torch.Tensor,
    leader_code: torch.Tensor,
    gamma: float,
    score_values: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Append leader copies of selected follower transitions to a rollout.

    The copies keep the behaviour log probabilities and actions. Their value
    baselines and one-step returns use the frozen leader. Boundary tensors
    retain their extra time step for the standard RLinf batch processor.
    """
    states = batch["forward_inputs"]["states"]
    raw_next_states = batch.pop("next_obs")["states"][:, followers]
    batch.pop("curr_obs", None)
    leader_states = states[:, followers].clone()
    leader_states[..., -leader_code.numel() :] = leader_code.to(states)
    next_codes = leader_code.to(raw_next_states).expand(*raw_next_states.shape[:2], -1)
    leader_next_states = torch.cat((raw_next_states, next_codes), dim=-1)
    values, next_values = score_values(
        torch.cat((leader_states, leader_next_states), dim=0)
    ).chunk(2, dim=0)
    dones = batch["dones"][1:, followers]
    returns = sapg_one_step_returns(
        batch["rewards"][:, followers], next_values, dones, gamma
    )

    def append_selected(fields: dict) -> None:
        for key, value in fields.items():
            if isinstance(value, torch.Tensor):
                fields[key] = torch.cat((value, value[:, followers]), dim=1)
            elif isinstance(value, dict):
                append_selected(value)

    original_envs = states.shape[1]
    append_selected(batch)
    batch["forward_inputs"]["states"][:, original_envs:] = leader_states
    batch["prev_values"][:-1, original_envs:] = values
    batch["prev_values"][-1, original_envs:] = next_values[-1]
    batch["returns"][:, original_envs:] = returns
    batch["advantages"][:, original_envs:] = returns - values
