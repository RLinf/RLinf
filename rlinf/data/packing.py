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

"""Best-Fit Decreasing (BFD) sequence packing for SFT training.

Packs multiple sequences into a single row so flash-attention can derive
``cu_seqlens`` from ``position_ids``, matching the pattern used by
``rlinf.hybrid_engines.fsdp.utils.pack_fsdp_input``.
"""

from __future__ import annotations

from typing import Any

import torch

from rlinf.data.datasets.item import SftDatasetItem


def best_fit_decreasing(
    lengths: list[int], max_bin_capacity: int
) -> list[list[int]]:
    """Pack items into bins using the Best-Fit Decreasing algorithm.

    Args:
        lengths: Length of each item (by original index).
        max_bin_capacity: Maximum total length per bin.

    Returns:
        A list of bins, where each bin is a list of original indices.
    """
    indexed = sorted(enumerate(lengths), key=lambda x: -x[1])

    bins: list[list[int]] = []
    bin_remaining: list[int] = []

    for orig_idx, length in indexed:
        if length > max_bin_capacity:
            bins.append([orig_idx])
            bin_remaining.append(0)
            continue

        best_bin = -1
        best_remaining = max_bin_capacity + 1

        for b_idx, rem in enumerate(bin_remaining):
            if rem >= length and rem < best_remaining:
                best_bin = b_idx
                best_remaining = rem

        if best_bin >= 0:
            bins[best_bin].append(orig_idx)
            bin_remaining[best_bin] -= length
        else:
            bins.append([orig_idx])
            bin_remaining.append(max_bin_capacity - length)

    return bins


def sft_packed_collate_fn(
    data_list: list[SftDatasetItem],
    max_seq_length: int,
    pad_token_id: int = 0,
) -> list[dict[str, Any]]:
    """Collate SFT items into packed batches using BFD bin-packing.

    Each returned dict represents one packed row (B=1) consumable by a model
    with flash-attention that derives ``cu_seqlens`` from ``position_ids``.

    Args:
        data_list: A batch of ``SftDatasetItem`` from the DataLoader.
        max_seq_length: Maximum packed sequence length (bin capacity).
        pad_token_id: Token id used for padding.

    Returns:
        A list of dicts, one per bin::

            {
                "prompt":             [1, max_seq_length],
                "position_ids":       [1, max_seq_length],
                "attention_mask":     None,
                "labels":            [1, max_seq_length],
                "multi_modal_inputs": dict,
                "num_sequences":      int,
            }
    """
    # --- materialise tensors ------------------------------------------------
    items: list[dict[str, Any]] = []
    for it in data_list:
        p = (
            it.prompt
            if isinstance(it.prompt, torch.Tensor)
            else torch.as_tensor(it.prompt, dtype=torch.long)
        )
        if p.dim() == 2 and p.size(0) == 1:
            p = p.squeeze(0)
        assert p.dim() == 1

        am = getattr(it, "attention_mask", None)
        am = (
            am
            if isinstance(am, torch.Tensor)
            else torch.as_tensor(am, dtype=torch.long)
        )
        if am.dim() == 2 and am.size(0) == 1:
            am = am.squeeze(0)

        lm = getattr(it, "label_mask", None)
        lm = (
            lm
            if isinstance(lm, torch.Tensor)
            else torch.as_tensor(lm, dtype=torch.bool)
        )
        if lm.dim() == 2 and lm.size(0) == 1:
            lm = lm.squeeze(0)

        # Effective length = number of valid (attended) tokens
        valid_len = int(am.sum().item()) if am is not None else p.numel()
        # Extract the valid (non-padded) tokens – they are right-aligned in sft_collate,
        # so the last `valid_len` tokens are the actual content.
        p_valid = p[-valid_len:] if valid_len < p.numel() else p
        lm_valid = lm[-valid_len:] if valid_len < lm.numel() else lm

        items.append(
            {
                "input_ids": p_valid,       # [valid_len]
                "label_mask": lm_valid,     # [valid_len]
                "length": valid_len,
                "multi_modal_inputs": it.multi_modal_inputs,
            }
        )

    lengths = [it["length"] for it in items]

    # --- BFD packing --------------------------------------------------------
    bins = best_fit_decreasing(lengths, max_seq_length)

    # --- build packed batches -----------------------------------------------
    packed_batches: list[dict[str, Any]] = []
    for bin_indices in bins:
        seq_ids_list: list[torch.Tensor] = []
        seq_pos_list: list[torch.Tensor] = []
        seq_labels_list: list[torch.Tensor] = []

        for seq_order, orig_idx in enumerate(bin_indices):
            ids = items[orig_idx]["input_ids"]
            lm = items[orig_idx]["label_mask"]
            seq_len = ids.numel()

            # position_ids: independent 0..seq_len-1 per sequence
            pos = torch.arange(seq_len, dtype=torch.long)

            # labels: clone input_ids, mask prompt tokens with -100
            labels = ids.clone()
            labels[lm] = -100

            # Cross-sequence boundary guard: for all sequences except the last
            # in the bin, set the last token's label to -100 so the HF
            # shift-left loss does not predict into the next sequence.
            if seq_order < len(bin_indices) - 1:
                labels[-1] = -100

            seq_ids_list.append(ids)
            seq_pos_list.append(pos)
            seq_labels_list.append(labels)

        # Concatenate all sequences, then pad to max_seq_length
        cat_ids = torch.cat(seq_ids_list)
        cat_pos = torch.cat(seq_pos_list)
        cat_labels = torch.cat(seq_labels_list)

        total_len = cat_ids.numel()
        pad_len = max_seq_length - total_len

        if pad_len > 0:
            cat_ids = torch.cat(
                [cat_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]
            )
            cat_pos = torch.cat(
                [cat_pos, torch.zeros(pad_len, dtype=torch.long)]
            )
            cat_labels = torch.cat(
                [cat_labels, torch.full((pad_len,), -100, dtype=torch.long)]
            )
        elif pad_len < 0:
            # Truncate to max_seq_length (should be rare with correct BFD)
            cat_ids = cat_ids[:max_seq_length]
            cat_pos = cat_pos[:max_seq_length]
            cat_labels = cat_labels[:max_seq_length]

        # Merge multi_modal_inputs from all items in the bin
        mm_list = [
            items[i]["multi_modal_inputs"]
            for i in bin_indices
            if items[i]["multi_modal_inputs"] is not None
        ]
        multi_modal_inputs: dict[str, Any] = {}
        if mm_list:
            for k in mm_list[0].keys():
                vals = [m[k] for m in mm_list]
                multi_modal_inputs[k] = (
                    torch.cat(vals, dim=0)
                    if isinstance(vals[0], torch.Tensor)
                    else vals
                )

        packed_batches.append(
            {
                "prompt": cat_ids.unsqueeze(0),          # [1, M]
                "position_ids": cat_pos.unsqueeze(0),    # [1, M]
                "attention_mask": None,
                "labels": cat_labels.unsqueeze(0),       # [1, M]
                "multi_modal_inputs": multi_modal_inputs,
                "num_sequences": len(bin_indices),
            }
        )

    return packed_batches
