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

from typing import Callable, Optional

import torch

from rlinf.algorithms.registry import register_loss_scale


@register_loss_scale("group")
def do_group_scale(context, batch):
    folding_scale = context["folding_scale"]
    assert "group" not in folding_scale, "folding_scale cannot be True before do_group_scale"
    context["folding_scale"].append("group")

    num_sequence = len(batch["idx_to_traj"])
    group_scale = num_sequence / context["actor_global_batch_size"]
    batch["advantages"] *= group_scale
    return batch

@register_loss_scale("sub_traj")
def do_sub_traj_scale(
    context: dict,
    batch: dict[str, torch.Tensor],
) -> dict:
    folding_scale = context["folding_scale"]
    assert "group" in folding_scale and "sub_traj" not in folding_scale
    context["folding_scale"].append("sub_traj")

    idx_to_sub_traj = batch["extra:idx_to_sub_traj"].tolist()
    traj_to_idx = {}
    for idx, traj in enumerate(batch["idx_to_traj"]):
        if traj not in traj_to_idx:
            traj_to_idx[traj] = []
        traj_to_idx[traj].append(idx)

    for traj, traj_idxes in traj_to_idx.items():
        sub_traj_to_idx = {}
        for idx in traj_idxes:
            sub_traj = idx_to_sub_traj[idx]
            if sub_traj not in sub_traj_to_idx:
                sub_traj_to_idx[sub_traj] = []
            sub_traj_to_idx[sub_traj].append(idx)

        for sub_traj_idxes in sub_traj_to_idx.values():
            for idx in sub_traj_idxes:
                batch["loss_scales"][idx] *= (
                    1
                    # apply sub_traj
                    / len(sub_traj_to_idx)
                    # apply turn size in sub_traj
                    / len(sub_traj_idxes)
                )
    return batch

@register_loss_scale("turn-in-sub_traj")
def do_turn_in_sub_traj_scale(
    context: dict,
    batch: dict[str, torch.Tensor],
) -> dict:
    folding_scale = context["folding_scale"]
    assert "group" in folding_scale and "sub_traj" in folding_scale and "turn-in-sub_traj" not in folding_scale
    context["folding_scale"].append("turn-in-sub_traj")

    idx_to_sub_traj = batch["extra:idx_to_sub_traj"].tolist()
    traj_to_idx = {}
    for idx, traj in enumerate(batch["idx_to_traj"]):
        if traj not in traj_to_idx:
            traj_to_idx[traj] = []
        traj_to_idx[traj].append(idx)

    for traj, traj_idxes in traj_to_idx.items():
        sub_traj_to_idx = {}
        for idx in traj_idxes:
            sub_traj = idx_to_sub_traj[idx]
            if sub_traj not in sub_traj_to_idx:
                sub_traj_to_idx[sub_traj] = []
            sub_traj_to_idx[sub_traj].append(idx)

        for sub_traj_idxes in sub_traj_to_idx.values():
            masked_counts = [
                batch["response_mask"][idx].sum().item()
                for idx in sub_traj_idxes
            ]
            masked_count_all = sum(masked_counts)
            for i, idx in enumerate(sub_traj_idxes):
                batch["loss_scales"][idx] *= (
                    1
                    # turn-level -> sub_traj-level
                    * len(sub_traj_idxes)
                    * masked_counts[i] / masked_count_all
                )
    return batch


def group_and_sub_traj_level_scale(
    context: dict,
    batch: dict[str, torch.Tensor],
) -> dict:
    folding_scale = context["folding_scale"]
    enable_scale_of_group = context.get("enable_scale_of_group", False)
    if enable_scale_of_group:
        assert "group" not in folding_scale
        context["folding_scale"].append("group")

    assert "sub_traj" not in folding_scale
    context["folding_scale"].append("sub_traj")

    idx_to_sub_traj = batch["extra:idx_to_sub_traj"].tolist()
    num_sequence = len(batch["idx_to_traj"])
    traj_to_idx = {}
    for idx, traj in enumerate(batch["idx_to_traj"]):
        if traj not in traj_to_idx:
            traj_to_idx[traj] = []
        traj_to_idx[traj].append(idx)

    for traj, traj_idxes in traj_to_idx.items():
        sub_traj_to_idx = {}
        for idx in traj_idxes:
            sub_traj = idx_to_sub_traj[idx]
            if sub_traj not in sub_traj_to_idx:
                sub_traj_to_idx[sub_traj] = []
            sub_traj_to_idx[sub_traj].append(idx)

        for sub_traj_idxes in sub_traj_to_idx.values():
            masked_counts = [
                batch["response_mask"][idx].sum().item()
                for idx in sub_traj_idxes
            ]
            masked_count_all = sum(masked_counts)
            for i, idx in enumerate(sub_traj_idxes):
                if enable_scale_of_group:
                    batch["loss_scales"][idx] *= (
                        1
                        # revert turn-level
                        * num_sequence
                        # apply expectation of sample and group
                        / context["actor_global_batch_size"]
                        # apply sub_traj
                        / len(sub_traj_to_idx)
                        # apply turn size in sub_traj
                        # / len(sub_traj_idxes)
                        # turn-level -> sub_traj-level
                        # * len(sub_traj_idxes)
                        * masked_counts[i] / masked_count_all
                    )
                else:
                    batch["loss_scales"][idx] *= (
                        1
                        # # revert turn-level
                        # * num_sequence
                        # # apply expectation of sample and group
                        # / context["actor_global_batch_size"]
                        # apply sub_traj
                        / len(sub_traj_to_idx)
                        # apply turn size in sub_traj
                        # / len(sub_traj_idxes)
                        # turn-level -> sub_traj-level
                        # * len(sub_traj_idxes)
                        * masked_counts[i] / masked_count_all
                    )
    return batch

def group_and_sub_traj_and_turn_scale(
    context: dict,
    batch: dict[str, torch.Tensor],
) -> dict:
    folding_scale = context["folding_scale"]
    enable_scale_of_group = context.get("enable_scale_of_group", False)
    if enable_scale_of_group:
        assert "group" not in folding_scale, "enable_scale_of_group cannot be True if folding_scale is True"
        context["folding_scale"].append("group")
    assert "sub_traj" not in folding_scale
    context["folding_scale"].append("sub_traj")
    assert "turn" not in folding_scale
    context["folding_scale"].append("turn")

    idx_to_sub_traj = batch["extra:idx_to_sub_traj"].tolist()
    num_sequence = len(batch["idx_to_traj"])
    traj_to_idx = {}
    for idx, traj in enumerate(batch["idx_to_traj"]):
        if traj not in traj_to_idx:
            traj_to_idx[traj] = []
        traj_to_idx[traj].append(idx)

    for traj, traj_idxes in traj_to_idx.items():
        sub_traj_to_idx = {}
        for idx in traj_idxes:
            sub_traj = idx_to_sub_traj[idx]
            if sub_traj not in sub_traj_to_idx:
                sub_traj_to_idx[sub_traj] = []
            sub_traj_to_idx[sub_traj].append(idx)

        for sub_traj_idxes in sub_traj_to_idx.values():
            for idx in sub_traj_idxes:
                if enable_scale_of_group:
                    batch["loss_scales"][idx] *= (
                        1
                        # revert turn-level
                        * num_sequence
                        # apply expectation of sample and group
                        / context["actor_global_batch_size"]
                        # apply sub_traj
                        / len(sub_traj_to_idx)
                        # apply turn size in sub_traj
                        / len(sub_traj_idxes)
                    )
                else:
                    batch["loss_scales"][idx] *= (
                        1
                        # # revert turn-level
                        # * num_sequence
                        # # apply expectation of sample and group
                        # / context["actor_global_batch_size"]
                        # apply sub_traj
                        / len(sub_traj_to_idx)
                        # apply turn size in sub_traj
                        / len(sub_traj_idxes)
                    )
    return batch