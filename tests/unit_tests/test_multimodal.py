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

"""Multimodal row alignment, cache budgets, and bounded image prefetch."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rlinf.data.schema.reasoning_results import DynamicRolloutResult, RolloutResult


@pytest.mark.parametrize("prefetch_batches", [0, 2, 8])
def test_image_prefetch_preserves_order_and_gradients(prefetch_batches, monkeypatch):
    from rlinf.utils.multimodal import MultimodalBatchPreparer

    consumed = []
    source_ids = []
    # Simulate distinct host copies without depending on a GPU for this test.
    monkeypatch.setattr(torch.Tensor, "cpu", lambda tensor: tensor.clone())

    def batches():
        for index in range(5):
            consumed.append(index)
            ids = torch.tensor([[index + 1, 9]])
            source_ids.append(ids)
            yield {
                "input_ids": ids,
                "attention_mask": torch.ones(1, 2),
                "multi_modal_inputs": [
                    {
                        "pixel_values": torch.full((2, 3), float(index)),
                        "image_grid_thw": torch.tensor([[1, 1, 2]]),
                    }
                ],
            }

    # Each result is 48 bytes; the byte limit admits at most two batches.
    preparer = MultimodalBatchPreparer(
        "unused", prefetch_batches=prefetch_batches, prefetch_bytes=96
    )
    weight = torch.tensor(0.5, requires_grad=True)
    with preparer.iterator_context(batches()) as iterator:
        for index, batch in enumerate(iterator):
            assert batch["input_ids"] is source_ids[index]
            inputs = preparer.for_forward(batch, "cpu")
            assert torch.equal(inputs["pixel_values"], torch.full((2, 3), float(index)))
            if prefetch_batches:
                assert (
                    len(consumed) <= index + 3
                )  # Two admitted, one compact lookahead.
            (inputs["pixel_values"] * weight).sum().backward()
            assert "_prepared_multimodal_inputs" not in batch
    assert weight.grad.item() == 6 * sum(range(5))


@pytest.mark.parametrize("failure", ["prepare", "consumer", "budget"])
def test_image_prefetch_propagates_failure_and_joins_threads(failure):
    import threading

    from rlinf.utils.multimodal import MultimodalBatchPreparer

    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2),
        "multi_modal_inputs": [],
    }
    if failure == "prepare":
        batch["multi_modal_inputs"] = [
            {
                "format": "legacy",
                "pixel_values": torch.ones(1),
                "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
            }
        ]
    preparer = MultimodalBatchPreparer(
        "unused",
        token_space=True,
        prefetch_batches=2,
        prefetch_bytes=1 if failure == "budget" else 4096,
    )
    error = {
        "prepare": "requires a captured image row",
        "budget": "exceeding",
        "consumer": "model forward",
    }[failure]
    with pytest.raises(ValueError, match=error):
        with preparer.iterator_context(iter([batch, batch])) as iterator:
            next(iterator)
            raise ValueError("model forward failed")
    assert not any(
        thread.name.startswith("rlinf-images") for thread in threading.enumerate()
    )


def test_static_image_rows_survive_grouping():
    rows = RolloutResult(
        num_sequence=2,
        group_size=1,
        prompt_ids=[[1], [2]],
        prompt_lengths=[1, 1],
        response_ids=[[3], [4]],
        response_lengths=[1, 1],
        is_end=[True, True],
        multi_modal_inputs=[None, {"image": "second"}],
    )
    groups = RolloutResult.split_result_list_by_group([rows])
    assert [g.multi_modal_inputs for g in groups] == [[None], [{"image": "second"}]]
    assert (
        RolloutResult.merge_result_list(groups).multi_modal_inputs
        == rows.multi_modal_inputs
    )

    rows.multi_modal_inputs = [{"image": "unknown row"}]
    for operation in (
        RolloutResult.split_result_list_by_group,
        RolloutResult.merge_result_list,
    ):
        with pytest.raises(ValueError, match="one entry per sequence"):
            operation([rows])


def test_batch_merge_pads_rows_for_a_text_only_batch():
    """A batch without images must still contribute one slot per row."""

    def batch(payloads):
        data = {
            "input_ids": [torch.arange(n) for n in (3, 5, 6)],
            "idx_to_traj": [0, 1, 2],
            "prompt_lengths": torch.ones(3, dtype=torch.int32),
            "response_lengths": torch.ones(3, dtype=torch.int32),
            "is_end": torch.zeros(3, dtype=torch.bool),
        }
        if payloads is not None:
            data["multi_modal_inputs"] = payloads
        return data

    with_images = batch([None, {"tag": torch.tensor([1.0])}, None])
    for order in ([batch(None), with_images], [with_images, batch(None)]):
        merged = DynamicRolloutResult.merge_batches(
            [dict(item) for item in order], group_size=3, adjust_traj_indices=True
        )
        payloads = merged["multi_modal_inputs"]
        assert len(payloads) == len(merged["input_ids"]) == 6
        image_row = 4 if "multi_modal_inputs" not in order[0] else 1
        assert [i for i, entry in enumerate(payloads) if entry is not None] == [
            image_row
        ]
        assert DynamicRolloutResult.pack_traj_batch({}, merged) is merged

    for merge in (RolloutResult.merge_batches, DynamicRolloutResult.merge_batches):
        with pytest.raises(ValueError, match="one entry per sequence"):
            merge([batch([{}])])


@pytest.mark.parametrize(
    "settings,token_space,error",
    [
        ({}, False, None),
        ({"cache_bytes": 1024}, False, None),
        ({"prefetch_batches": 2, "prefetch_bytes": 4096}, True, None),
        ({"cache_bytes": -1}, False, "cache_bytes"),
        ({"prefetch_batches": True}, True, "prefetch_batches"),
        ({"prefetch_bytes": 1.5}, True, "prefetch_bytes"),
        ({"prefetch_batches": 1}, True, "positive prefetch_bytes"),
        (
            {"prefetch_batches": 1, "prefetch_bytes": 4096},
            False,
            "enable_shared_multi_modal_processor",
        ),
    ],
)
def test_multimodal_learner_budgets_without_rpent(settings, token_space, error):
    from omegaconf import OmegaConf

    from rlinf.config import validate_multimodal_learner_cfg

    cfg = OmegaConf.create(
        {
            "actor": {"multimodal": settings},
            "rollout": {"sglang": {"enable_shared_multi_modal_processor": token_space}},
        }
    )
    if error:
        with pytest.raises(ValueError, match=error):
            validate_multimodal_learner_cfg(cfg)
    else:
        snapshot = OmegaConf.to_container(cfg)
        validate_multimodal_learner_cfg(cfg)
        assert OmegaConf.to_container(cfg) == snapshot


@pytest.mark.parametrize("backend,cp", [("megatron", 2), ("fsdp", 1)])
def test_image_replay_rejects_unsupported_learner(backend, cp):
    from omegaconf import OmegaConf

    from rlinf.config import validate_multimodal_learner_cfg

    cfg = OmegaConf.create(
        {
            "actor": {
                "expect_multi_modal_inputs": True,
                "training_backend": backend,
                "model": {"context_parallel_size": cp},
            }
        }
    )
    with pytest.raises(ValueError, match="actor.expect_multi_modal_inputs"):
        validate_multimodal_learner_cfg(cfg)


def test_release_capture_replays_positions_and_verifies_shared_pixels(tmp_path):
    from rlinf.hybrid_engines.sglang.multimodal.replay import (
        REPLAY_FORMAT,
        store_pixels,
    )
    from rlinf.utils.multimodal import materialize_multimodal_inputs

    pixels = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    data = {
        "format": REPLAY_FORMAT,
        "prompt_token_ids": [10, 11, 12],
        "image_spans": [[1, 2]],
        "pixel_values": pixels,
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "mrope_positions": torch.tensor([[0, 1, 2]] * 3),
        "mrope_position_delta": torch.tensor([0]),
    }
    store_pixels(data, tmp_path)
    ids = torch.tensor([[10, 11, 12, 13, 0]])

    def replay():
        return materialize_multimodal_inputs(
            [data],
            "unused",
            "cpu",
            input_ids=ids,
            attention_mask=ids.ne(0),
            token_space=True,
        )

    result = replay()
    assert torch.equal(result["pixel_values"], pixels.bfloat16())
    assert result["image_input_mask"].tolist() == [[False, True, False, False, False]]
    assert result["multimodal_position_ids"][:, 0].tolist() == [[0, 1, 2, 3, 0]] * 3
    ids[0, 0] = 2
    with pytest.raises(ValueError, match="prompt"):
        replay()
    ids[0, 0] = 10
    Path(data["pixel_refs"][0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        replay()


def test_release_capture_checks_sampled_history_and_request_id():
    from rlinf.hybrid_engines.sglang.multimodal.replay import (
        CAPTURE_KEY,
        capture_file,
        generate_with_capture,
        save_capture,
    )

    rid = "a" * 32
    pixels = torch.ones(4, 2)

    async def generate(input_ids, image_data, sampling_params, return_logprob):
        capture = sampling_params["custom_params"][CAPTURE_KEY]
        output = SimpleNamespace(
            input_ids=input_ids,
            mrope_positions=torch.arange(len(input_ids)).repeat(3, 1),
            mrope_position_delta=0,
            mm_items=[
                SimpleNamespace(
                    offsets=[(1, 1)],
                    feature=pixels,
                    image_grid_thw=torch.tensor([[1, 2, 2]]),
                )
            ],
        )
        save_capture(capture_file(capture["ticket"], rid), output)
        return {"meta_info": {"id": rid}, "output_ids": [23]}

    def sample(ids, history=None):
        return asyncio.run(
            generate_with_capture(
                generate,
                ids,
                ["image"],
                {},
                {"image_spans": [[1, 2]], "history": history},
                False,
            )
        )

    _, _, history = sample([10, 11, 12])
    _, row, _ = sample([10, 11, 12, 23, 24], history)
    assert row["prompt_token_ids"] == [10, 11, 12, 23, 24]
    with pytest.raises(ValueError, match="sampled token history"):
        sample([10, 11, 12, 99, 24], history)
    pixels.zero_()
    with pytest.raises(ValueError, match="pixels"):
        sample([10, 11, 12, 23, 24], history)


def test_mrope_verification_removes_hook_after_failure():
    pytest.importorskip("megatron.bridge")
    from rlinf.hybrid_engines.megatron.qwen_vl_inputs import verify_qwen_vl_positions

    class BridgeModel(torch.nn.Module):
        __module__ = "megatron.bridge.models.qwen_vl.test"

        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Identity()

    model = BridgeModel()
    positions = torch.zeros(3, 1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="M-RoPE positions different"):
        with verify_qwen_vl_positions(model, positions, torch.ones(1, 2)):
            model.language_model(position_ids=positions + 1)
    assert not model.language_model._forward_pre_hooks


def test_bridge_replays_literal_media_ids_as_text():
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index

    from rlinf.hybrid_engines.megatron.qwen_vl_inputs import qwen_vl_replay_inputs

    class EchoIDs(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            return input_ids

    class BridgeModel(torch.nn.Module):
        __module__ = "megatron.bridge.models.qwen_vl.test"

    model = BridgeModel()
    model.image_token_id, model.vision_start_token_id = 2, 1
    model.language_model = EchoIDs()
    model.language_model.embedding = EchoIDs()
    # One real 2x2 image, then a literal image triplet, a pad and a trailing start.
    ids = torch.tensor([[1, 2, 2, 2, 2, 3, 1, 2, 3, 2, 1]])
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 1:5] = True
    expected = torch.tensor(
        [
            [0, 1, 1, 1, 1, 3, 4, 5, 6, 7, 8],
            [0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8],
        ]
    ).unsqueeze(1)
    inputs = {
        "input_ids": ids,
        "image_input_mask": mask,
        "attention_mask": torch.ones_like(mask),
    }
    with qwen_vl_replay_inputs(model, expected, inputs) as replay:
        geometry = replay["input_ids"]
        positions, _ = get_rope_index(
            2, 2, 4, 1, geometry, image_grid_thw=torch.tensor([[1, 4, 4]])
        )
        assert torch.equal(geometry == model.image_token_id, mask)
        assert torch.equal(model.language_model.embedding(input_ids=geometry), ids)
        assert torch.equal(
            model.language_model(input_ids=geometry, position_ids=positions), ids
        )
    assert not model.language_model.embedding._forward_pre_hooks
