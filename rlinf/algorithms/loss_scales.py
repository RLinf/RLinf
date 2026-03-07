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


import torch

from rlinf.algorithms.registry import register_loss_scale


@register_loss_scale("group_level")
def group_scale(context, batch):
    """Apply the outer group-level normalization factor to advantages.

    This function handles the top-level `1 / G` normalization in GRPO.

    Concretely, it rescales the current advantages so that the effective batch
    contribution matches the configured actor global batch size after dynamic
    batches are merged across data-parallel workers.

    Args:
        context (dict): Shared scaling context built in `run_training`.
        batch (dict): Dynamic rollout batch containing `idx_to_traj` and `advantages`.

    Returns:
        dict: The input batch with group-normalized advantages.
    """
    folding_scale = context["folding_scale"]
    assert "group_level" not in folding_scale, (
        "`group_level` loss scaling can only be applied once. Apply the "
        "group-level factor before any `agent_level` or `turn_level` factor."
    )
    context["folding_scale"].append("group_level")

    num_sequence = len(batch["idx_to_traj"])
    dp_world_size = context.get("data_parallel_world_size", 1)
    # Convert the local dynamic-turn count back to the effective global batch
    # normalization used by the actor update.
    group_scale = num_sequence * dp_world_size / context["actor_global_batch_size"]
    batch["advantages"] *= group_scale
    return batch


@register_loss_scale("agent_level")
def agent_scale(
    context: dict,
    batch: dict[str, torch.Tensor],
) -> dict:
    """Apply per-agent and uniform per-turn scaling inside each trajectory.

    This stage contributes:
    - the `1 / A_i` factor that normalizes by the number of agents in trajectory `i`.
    - the temporary `1 / T_{i,a}` factor that uniformly weights each turn of the same agent `a` in trajectory `i`.

    Here, one trajectory corresponds to sampled rollout `i`, and one
    `sub_traj` corresponds to agent `a` inside that trajectory. Every turn of
    the same agent receives the same weight at this stage. If `turn_level` is
    applied afterwards, that later step refines the uniform per-turn weight
    into the token-proportional weighting.

    Args:
        context (dict): Shared scaling context built in `run_training`.
        batch (dict[str, torch.Tensor]): Dynamic rollout batch containing
            `idx_to_traj`, `extra:idx_to_sub_traj`, and `loss_scales`.

    Returns:
        dict: The input batch with updated `loss_scales`.
    """
    folding_scale = context["folding_scale"]
    assert "group_level" in folding_scale and "agent_level" not in folding_scale, (
        "`agent_level` loss scaling requires `group_level` to be applied first, "
        "and it can only be applied once."
    )
    context["folding_scale"].append("agent_level")

    idx_to_sub_traj = batch["extra:idx_to_sub_traj"].tolist()
    traj_to_idx = {}
    # Group all flattened turns by trajectory i.
    for idx, traj in enumerate(batch["idx_to_traj"]):
        if traj not in traj_to_idx:
            traj_to_idx[traj] = []
        traj_to_idx[traj].append(idx)

    for traj, traj_idxes in traj_to_idx.items():
        sub_traj_to_idx = {}
        # Inside one trajectory, group turns by sub-trajectory / agent a.
        for idx in traj_idxes:
            sub_traj = idx_to_sub_traj[idx]
            if sub_traj not in sub_traj_to_idx:
                sub_traj_to_idx[sub_traj] = []
            sub_traj_to_idx[sub_traj].append(idx)

        for sub_traj_idxes in sub_traj_to_idx.values():
            for idx in sub_traj_idxes:
                # Apply 1 / A_i and a uniform 1 / T_{i,a} across the turns of the same agent.
                batch["loss_scales"][idx] *= (
                    1 / len(sub_traj_to_idx) / len(sub_traj_idxes)
                )
    return batch


@register_loss_scale("turn_level")
def turn_scale(
    context: dict,
    batch: dict[str, torch.Tensor],
) -> dict:
    """Refine uniform per-turn weights into token-proportional turn weights.

    This function must run after `agent_scale`. At that point,
    `loss_scales` already contains the `1 / A_i` and `1 / T_{i,a}` factors.
    `turn_scale` converts the uniform turn weighting into the token-level
    weighting:

    `1 / A_i * 1 / T_{i,a}` becomes
    `1 / A_i * |o_t^{i,a}| / sum_t |o_t^{i,a}|`.

    Because the actor loss is still reduced over valid tokens afterwards, this
    produces the desired per-agent normalization by the total token count
    `sum_t |o_t^{i,a}|`.

    Args:
        context (dict): Shared scaling context built in `run_training`.
        batch (dict[str, torch.Tensor]): Dynamic rollout batch containing
            `idx_to_traj`, `extra:idx_to_sub_traj`, `response_mask`, and
            `loss_scales`.

    Returns:
        dict: The input batch with token-proportional `loss_scales`.
    """
    folding_scale = context["folding_scale"]
    assert (
        "group_level" in folding_scale
        and "agent_level" in folding_scale
        and "turn_level" not in folding_scale
    ), (
        "`turn_level` loss scaling requires both `group_level` and "
        "`agent_level` to be applied first, and it can only be applied once."
    )
    context["folding_scale"].append("turn_level")

    idx_to_sub_traj = batch["extra:idx_to_sub_traj"].tolist()
    traj_to_idx = {}
    # Group all flattened turns by trajectory i.
    for idx, traj in enumerate(batch["idx_to_traj"]):
        if traj not in traj_to_idx:
            traj_to_idx[traj] = []
        traj_to_idx[traj].append(idx)

    for traj, traj_idxes in traj_to_idx.items():
        sub_traj_to_idx = {}
        # Inside one trajectory, group turns by sub-trajectory / agent a.
        for idx in traj_idxes:
            sub_traj = idx_to_sub_traj[idx]
            if sub_traj not in sub_traj_to_idx:
                sub_traj_to_idx[sub_traj] = []
            sub_traj_to_idx[sub_traj].append(idx)

        for sub_traj_idxes in sub_traj_to_idx.values():
            # Count how many valid response tokens belong to each turn t of the
            # same agent a, then normalize by the agent's total token count.
            masked_counts = [
                batch["response_mask"][idx].sum().item() for idx in sub_traj_idxes
            ]
            masked_count_all = sum(masked_counts)
            for i, idx in enumerate(sub_traj_idxes):
                # `agent_scale` already applied 1 / T_{i,a}. Multiply by
                # T_{i,a} * |o_t| / sum_t |o_t| so the combined factor becomes
                # |o_t| / sum_t |o_t|.
                batch["loss_scales"][idx] *= (
                    1 * len(sub_traj_idxes) * masked_counts[i] / masked_count_all
                )
    return batch
