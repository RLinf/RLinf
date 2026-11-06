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

"""Megatron packing contracts shared by inference and training."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _packing_process(rank, tp_size, cp_size, init_file):
    from megatron.core import parallel_state

    from rlinf.hybrid_engines.megatron.utils import (
        postprocess_packed_seqs,
        postprocess_ragged_seqs,
        preprocess_packed_seqs,
        preprocess_ragged_seqs,
    )

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=tp_size * cp_size,
        timeout=timedelta(seconds=60),
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )
    try:
        lengths = [101, 201, 301]
        rows = [torch.arange(n).float() for n in lengths]
        dense = torch.stack(
            [torch.nn.functional.pad(row, (0, 8192 - len(row))) for row in rows]
        )
        mask = torch.arange(8192)[None, :] < torch.tensor(lengths)[:, None]
        packed, params = preprocess_ragged_seqs(rows)
        reference, reference_params = preprocess_packed_seqs(dense, mask)
        torch.testing.assert_close(packed, reference)
        torch.testing.assert_close(params.cu_seqlens_q, reference_params.cu_seqlens_q)
        labels = dense.roll(-1, dims=1)
        label_mask = mask.roll(-1, dims=1)
        label_mask[:, -1] = False
        reference_labels, _ = preprocess_packed_seqs(labels, mask)
        reference_label_mask, _ = preprocess_packed_seqs(label_mask, mask)
        new_labels, _ = preprocess_ragged_seqs(
            [torch.cat((row[1:], row.new_zeros(1))) for row in rows]
        )
        new_label_mask, _ = preprocess_ragged_seqs(
            [torch.arange(n) < n - 1 for n in lengths]
        )
        torch.testing.assert_close(new_label_mask, reference_label_mask)
        torch.testing.assert_close(
            new_labels[new_label_mask], reference_labels[reference_label_mask]
        )
        current = packed.clone().requires_grad_()
        restored = postprocess_ragged_seqs(current, params, lengths)
        sum(row.square().sum() for row in restored).backward()
        previous = reference.clone().requires_grad_()
        padded = postprocess_packed_seqs(previous, reference_params, mask, 3, 8192)
        padded.square().sum().backward()
        torch.testing.assert_close(current.grad, previous.grad)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize(("tp_size", "cp_size"), [(1, 1), (4, 1), (1, 2)])
def test_ragged_packing_matches_dense_and_gradients(tmp_path, tp_size, cp_size):
    pytest.importorskip("megatron.core")
    mp.spawn(
        _packing_process,
        args=(tp_size, cp_size, str(tmp_path / "rendezvous")),
        nprocs=tp_size * cp_size,
        join=True,
    )
