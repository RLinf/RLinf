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

"""Unit tests for rlinf.data.packing (BFD packing & sft_packed_collate_fn)."""

import torch

from rlinf.data.datasets.item import SftDatasetItem
from rlinf.data.packing import best_fit_decreasing, sft_packed_collate_fn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sft_item(
    seq_len: int,
    prompt_len: int,
    idx: int = 0,
    start_token: int = 100,
) -> SftDatasetItem:
    """Create a minimal SftDatasetItem for testing.

    ``prompt_len`` tokens are prompt (label_mask=True), the rest are answer.
    """
    input_ids = torch.arange(start_token, start_token + seq_len, dtype=torch.long)
    attention_mask = torch.ones(seq_len, dtype=torch.long)
    # label_mask: True for prompt tokens (will be masked to -100), False for answer
    label_mask = torch.zeros(seq_len, dtype=torch.bool)
    label_mask[:prompt_len] = True

    return SftDatasetItem(
        prompt=input_ids,
        length=seq_len,
        answer="dummy",
        idx=idx,
        attention_mask=attention_mask,
        label_mask=label_mask,
    )


# ---------------------------------------------------------------------------
# Tests for best_fit_decreasing
# ---------------------------------------------------------------------------

class TestBestFitDecreasing:
    def test_basic(self):
        """Items that fit two-per-bin should be paired optimally."""
        lengths = [3, 5, 2, 7, 1]
        bins = best_fit_decreasing(lengths, max_bin_capacity=8)

        # Every bin's total length should be <= 8
        for b in bins:
            assert sum(lengths[i] for i in b) <= 8

        # All indices accounted for
        flat = sorted(i for b in bins for i in b)
        assert flat == [0, 1, 2, 3, 4]

    def test_single_long(self):
        """An item longer than capacity gets its own bin."""
        lengths = [10, 3, 2]
        bins = best_fit_decreasing(lengths, max_bin_capacity=8)

        # The item with length 10 must be alone
        for b in bins:
            if 0 in b:
                assert b == [0]
                break

    def test_all_same_length(self):
        """Items of equal length should pack evenly."""
        lengths = [4, 4, 4, 4]
        bins = best_fit_decreasing(lengths, max_bin_capacity=8)
        assert len(bins) == 2
        for b in bins:
            assert sum(lengths[i] for i in b) <= 8

    def test_empty(self):
        """Empty input yields empty output."""
        assert best_fit_decreasing([], max_bin_capacity=10) == []

    def test_exact_fit(self):
        """Items that exactly fill bins should not leave waste."""
        lengths = [5, 3, 5, 3]
        bins = best_fit_decreasing(lengths, max_bin_capacity=8)
        assert len(bins) == 2
        for b in bins:
            assert sum(lengths[i] for i in b) == 8


# ---------------------------------------------------------------------------
# Tests for sft_packed_collate_fn
# ---------------------------------------------------------------------------

class TestSftPackedCollateFn:
    def test_shapes(self):
        """All output tensors should have shape [1, max_seq_length]."""
        max_seq = 32
        items = [_make_sft_item(10, 4, idx=0), _make_sft_item(8, 3, idx=1)]
        result = sft_packed_collate_fn(items, max_seq_length=max_seq, pad_token_id=0)

        assert isinstance(result, list)
        for pb in result:
            assert pb["prompt"].shape == (1, max_seq)
            assert pb["position_ids"].shape == (1, max_seq)
            assert pb["labels"].shape == (1, max_seq)
            assert pb["attention_mask"] is None
            assert isinstance(pb["num_sequences"], int)

    def test_labels_boundary(self):
        """For multi-sequence bins, the last token of non-final seqs has label=-100."""
        max_seq = 64
        items = [
            _make_sft_item(10, 3, idx=0, start_token=100),
            _make_sft_item(8, 2, idx=1, start_token=200),
        ]
        result = sft_packed_collate_fn(items, max_seq_length=max_seq, pad_token_id=0)

        # With total length 18 < 64, both should fit in one bin
        assert len(result) == 1
        pb = result[0]
        assert pb["num_sequences"] == 2

        labels = pb["labels"].squeeze(0)  # [max_seq]

        # Position 9 (last token of first seq, 0-indexed) should be -100
        # because it's the boundary between seq 0 and seq 1.
        assert labels[9].item() == -100

        # Padding region should be all -100
        assert (labels[18:] == -100).all()

    def test_position_ids_reset(self):
        """position_ids should reset to 0 at each sequence boundary."""
        max_seq = 64
        items = [
            _make_sft_item(10, 3, idx=0),
            _make_sft_item(8, 2, idx=1),
        ]
        result = sft_packed_collate_fn(items, max_seq_length=max_seq, pad_token_id=0)
        assert len(result) == 1

        pos = result[0]["position_ids"].squeeze(0)

        # First sequence: 0,1,...,9
        assert pos[0].item() == 0
        assert pos[9].item() == 9

        # Second sequence starts at offset 10 and resets to 0
        assert pos[10].item() == 0
        assert pos[17].item() == 7

    def test_prompt_mask_preserved(self):
        """Prompt tokens (label_mask=True) should have labels=-100."""
        max_seq = 32
        prompt_len = 5
        total_len = 12
        items = [_make_sft_item(total_len, prompt_len, idx=0, start_token=50)]
        result = sft_packed_collate_fn(items, max_seq_length=max_seq, pad_token_id=0)

        assert len(result) == 1
        labels = result[0]["labels"].squeeze(0)

        # First prompt_len tokens should be -100 (masked)
        assert (labels[:prompt_len] == -100).all()
        # Answer tokens should NOT be -100
        assert (labels[prompt_len:total_len] != -100).all()

    def test_multi_modal_inputs_merged(self):
        """multi_modal_inputs from different items should be cat'd along dim=0."""
        max_seq = 64
        item0 = _make_sft_item(10, 3, idx=0)
        item0.multi_modal_inputs = {"pixel_values": torch.randn(2, 3, 224, 224)}
        item1 = _make_sft_item(8, 2, idx=1)
        item1.multi_modal_inputs = {"pixel_values": torch.randn(1, 3, 224, 224)}

        result = sft_packed_collate_fn([item0, item1], max_seq_length=max_seq)
        assert len(result) == 1
        assert result[0]["multi_modal_inputs"]["pixel_values"].shape[0] == 3

    def test_single_item_per_bin(self):
        """An item that fills the entire capacity should still work."""
        max_seq = 10
        items = [_make_sft_item(10, 4, idx=0)]
        result = sft_packed_collate_fn(items, max_seq_length=max_seq, pad_token_id=0)

        assert len(result) == 1
        assert result[0]["num_sequences"] == 1
        # No padding needed
        labels = result[0]["labels"].squeeze(0)
        assert labels.shape[0] == max_seq


class TestBackwardCompat:
    def test_no_packing_config(self):
        """When packing is not enabled, worker should use sft_collate_fn.

        This is a sanity check: sft_collate_fn should still produce the
        expected dict keys (no position_ids, has label_mask).
        """
        from rlinf.data.datasets import sft_collate_fn

        items = [_make_sft_item(10, 4, idx=0), _make_sft_item(10, 3, idx=1)]
        batch = sft_collate_fn(items)

        assert "prompt" in batch
        assert "attention_mask" in batch
        assert "label_mask" in batch
        assert batch["prompt"].shape[0] == 2  # B=2
