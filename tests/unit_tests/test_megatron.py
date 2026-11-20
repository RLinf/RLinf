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

"""Megatron forward input contracts, exercised without accelerators."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch


def _parallel_groups(monkeypatch, tp, cp, rank):
    pytest.importorskip("megatron.core")
    from megatron.core import parallel_state

    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: tp
    )
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: cp)
    monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: rank)
    return SimpleNamespace(
        tp=SimpleNamespace(size=lambda: tp),
        cp=SimpleNamespace(size=lambda: cp, rank=lambda: rank),
    )


def _batch():
    ids = torch.arange(1, 49).reshape(3, 16)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[0] = True
    mask[1, 3:12] = True
    mask[2, :5] = True
    return ids, mask


def _expected_shard(ids, mask, tp, cp, rank):
    parts = []
    alignment = tp * cp * 2 if cp > 1 else tp
    for row, valid in zip(ids, mask):
        values = row[valid].tolist()
        values += [0] * (-len(values) % alignment)
        if cp > 1:
            size = len(values) // (2 * cp)
            values = (
                values[rank * size : (rank + 1) * size]
                + values[(2 * cp - rank - 1) * size : (2 * cp - rank) * size]
            )
        parts.extend(values)
    return torch.tensor([parts], dtype=ids.dtype)


@pytest.mark.parametrize("tp,cp", [(1, 1), (2, 1), (1, 2), (2, 2), (2, 4)])
@pytest.mark.parametrize("padding", [None, 16, 32])
@pytest.mark.parametrize("pre_process", [False, True])
def test_model_owned_packing_matches_local_labels(
    monkeypatch, tp, cp, padding, pre_process
):
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
        preprocess_packed_seqs as bridge_pack,
    )

    from rlinf.hybrid_engines.megatron.utils import (
        prepare_full_model_inputs,
        preprocess_packed_seqs,
    )

    ids, mask = _batch()
    for rank in range(cp):
        groups = _parallel_groups(monkeypatch, tp, cp, rank)
        tokens, model_mask, params = prepare_full_model_inputs(
            ids, mask, pre_process, padding
        )
        assert torch.equal(tokens[:3, :16], ids)
        assert torch.equal(model_mask[:3, :16], mask)
        actual, internal_params = bridge_pack(tokens, model_mask, pg_collection=groups)
        expected = _expected_shard(ids, mask, tp, cp, rank)
        if padding is not None:
            expected = torch.nn.functional.pad(
                expected, (0, padding * 3 // cp - expected.shape[1])
            )
        assert torch.equal(actual, expected)
        assert torch.equal(params.cu_seqlens_q, internal_params.cu_seqlens_q)
        assert actual.shape[1] * cp == params.cu_seqlens_q[-1]
        labels, _ = preprocess_packed_seqs(ids, mask, padding_seqlen=padding)
        assert torch.equal(actual, labels)


@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("padding", [None, 16, 32])
@pytest.mark.parametrize("pre_process", [False, True])
def test_custom_forward_preserves_tokens_labels_and_gradients(
    monkeypatch, internal, padding, pre_process
):
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
        preprocess_packed_seqs as bridge_pack,
    )

    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        MegatronModelManager,
    )
    from rlinf.hybrid_engines.megatron.utils import prepare_full_model_inputs

    groups = _parallel_groups(monkeypatch, 2, 1, 0)
    ids, mask = _batch()
    # A CPU backend implements the vendor input/output contract; RLinf's
    # preparation, label packing and output restoration all run unchanged.

    class Backend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_process = pre_process if internal else True
            self.post_process = True
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(
            self, input_ids, attention_mask, position_ids, packed_seq_params, **kwargs
        ):
            if internal:
                assert torch.equal(input_ids[:3, :16], ids)
                assert kwargs["image_grid_thw"] is grid
                input_ids, _ = bridge_pack(
                    input_ids, attention_mask, pg_collection=groups
                )
            else:
                assert attention_mask is None
            return input_ids.float().unsqueeze(-1) * self.weight

    grid = torch.tensor([[1, 2, 2]])
    model = Backend()
    manager = MegatronModelManager.__new__(MegatronModelManager)
    if internal:
        manager.custom_forward = partial(
            manager.custom_forward, prepare_inputs=prepare_full_model_inputs
        )

    def process(logits, labels):
        assert torch.equal(logits[..., 0].detach(), labels.float() * 2)
        return {"scores": logits[..., 0]}

    result = manager.custom_forward(
        model,
        ids,
        mask,
        torch.zeros_like(ids),
        sequence_parallel=False,
        logits_processor=process,
        logits_processor_args={"labels": ids},
        padding_seqlen=padding,
        image_grid_thw=grid,
    )["scores"]
    assert torch.equal(result[mask], ids[mask].float() * 2)
    assert torch.equal(result[~mask], torch.zeros_like(result[~mask]))
    result.sum().backward()
    assert model.weight.grad.item() == ids[mask].sum().item()


@pytest.mark.parametrize("cp", [1, 2, 4])
@pytest.mark.parametrize("padding", [None, 16, 32])
def test_bridge_multimodal_forward_splits_once_on_cpu(monkeypatch, cp, padding):
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel

    from rlinf.hybrid_engines.megatron.utils import prepare_full_model_inputs

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda *args: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    ids, mask = _batch()

    class LanguageBackend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.rotary_pos_emb = SimpleNamespace(is_thd_format=False)

        def embedding(self, input_ids, position_ids):
            return input_ids.float().T.unsqueeze(-1) * self.weight

        def forward(self, decoder_input, **kwargs):
            return decoder_input.transpose(0, 1)

    for rank in range(cp):
        groups = _parallel_groups(monkeypatch, 2, cp, rank)
        model = Qwen3VLModel.__new__(Qwen3VLModel)
        torch.nn.Module.__init__(model)
        model.pg_collection = groups
        model.config = SimpleNamespace(sequence_parallel=False, spatial_merge_size=2)
        model.pre_process = True
        model.use_dist_train = False
        model.image_token_id = 1000
        model.video_token_id = 1001
        model.vision_start_token_id = 1002
        model.square_merge_size = 4
        model.language_model = LanguageBackend()
        tokens, model_mask, params = prepare_full_model_inputs(ids, mask, True, padding)
        output = model(
            input_ids=tokens, attention_mask=model_mask, packed_seq_params=params
        )
        expected = _expected_shard(ids, mask, 2, cp, rank)
        if padding is not None:
            expected = torch.nn.functional.pad(
                expected, (0, padding * 3 // cp - expected.shape[1])
            )
        assert torch.equal(output[..., 0], expected.float() * 2)
        output.sum().backward()
        assert model.language_model.weight.grad.item() == expected.sum().item()
