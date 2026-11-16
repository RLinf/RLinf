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

"""Checkpoint and online weight conversion contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rlinf.utils.ckpt_convertor.openpi.jax_to_openpi import _unwrap_value_leaves


def test_jax_value_wrappers_are_unwrapped():
    kernel = np.zeros((4, 2), dtype=np.float32)
    bias = np.ones(2, dtype=np.float32)
    tree = {
        "PaliGemma": {
            "img": {"pos_embedding": {"value": kernel}},
            "llm": {"layers": {"Dense_0": {"kernel": {"value": kernel}}}},
        },
        "action_in_proj": {"kernel": {"value": kernel}, "bias": {"value": bias}},
    }

    unwrapped = _unwrap_value_leaves(tree)

    assert unwrapped["PaliGemma"]["img"]["pos_embedding"] is kernel
    assert unwrapped["PaliGemma"]["llm"]["layers"]["Dense_0"]["kernel"] is kernel
    assert unwrapped["action_in_proj"]["kernel"] is kernel
    assert unwrapped["action_in_proj"]["bias"] is bias


def _assert_tensor_bits_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().contiguous().reshape(-1).view(torch.uint8),
        expected.detach().contiguous().reshape(-1).view(torch.uint8),
    )


@pytest.mark.parametrize(
    "load_mtp,sglang_options,accepted",
    [
        (None, {}, True),
        (True, {}, True),
        (False, {}, True),
        (True, {"speculative_algorithm": None}, True),
        (
            False,
            {
                "speculative_algorithm": "NEXTN",
                "speculative_draft_model_path": "/old/checkpoint",
            },
            True,
        ),
        (
            True,
            {
                "speculative_algorithm": "STANDALONE",
                "speculative_draft_model_path": "/old/checkpoint",
            },
            True,
        ),
        ("false", {}, False),
    ],
)
def test_qwen35_mtp_loading_needs_only_one_switch(load_mtp, sglang_options, accepted):
    from omegaconf import OmegaConf

    from rlinf.config import validate_rollout_cfg

    actor = OmegaConf.create(
        {
            "training_backend": "megatron",
            "megatron": {"mbridge": True},
            "model": {"tensor_model_parallel_size": 2},
        }
    )
    if load_mtp is not None:
        actor.model.load_mtp = load_mtp
    rollout = OmegaConf.create(
        {
            "model": {"model_type": "qwen3_5", "model_path": "/unused/checkpoint"},
            "sglang": sglang_options,
            "vllm": {},
            "enforce_eager": False,
            "tensor_parallel_size": 4,
            "return_logprobs": True,
        }
    )
    if accepted:
        result = validate_rollout_cfg(rollout, OmegaConf.create({}), actor)
        for name, value in sglang_options.items():
            assert result.sglang[name] == value
        assert actor.model.get("load_mtp", True) is (load_mtp is not False)
    else:
        with pytest.raises(AssertionError, match="load_mtp"):
            validate_rollout_cfg(rollout, OmegaConf.create({}), actor)


def _qwen35_config(tp):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hidden_size=4,
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=2,
            ffn_hidden_size=8,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
        ),
        vision_config=SimpleNamespace(
            hidden_size=8,
            num_heads=4,
            intermediate_size=8,
            spatial_merge_size=2,
        ),
        reshard_tp_size=tp,
        reshard_pp_size=1,
        enable_dp_attention=False,
    )


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("mtp", [False, True])
def test_qwen35_gated_queries_keep_each_head_with_its_gate(tp, mtp):
    from rlinf.utils.convertor.utils import Qwen35Convertor

    groups = range(4 - 4 // tp, 4)
    fused, queries, keys, values = [], [], [], []
    for group in groups:
        q0, q1 = [
            torch.full((2, 4), 10 * group + i, dtype=torch.float32) for i in (1, 2)
        ]
        g0, g1 = q0 + 100, q1 + 100
        key, value = q0 + 200, q0 + 300
        fused.extend((q0, q1, g0, g1, key, value))
        queries.extend((q0, g0, q1, g1))
        keys.append(key)
        values.append(value)
    source = (
        "language_model.mtp.layers.0.mtp_model_layer."
        if mtp
        else "language_model.decoder.layers.3."
    )
    target = (
        "mtp.layers.0.self_attn." if mtp else "model.language_model.layers.3.self_attn."
    )
    result = Qwen35Convertor(_qwen35_config(tp), strict=True).convert(
        {
            source + "self_attention.linear_qkv.weight": torch.cat(fused),
        }
    )
    for name, expected in (("q", queries), ("k", keys), ("v", values)):
        _assert_tensor_bits_equal(
            result[target + name + "_proj.weight"], torch.cat(expected)
        )


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_qwen35_gdn_target_shards_and_zero_centered_norm(tp):
    from rlinf.utils.convertor.utils import Qwen35Convertor

    parts = [
        torch.full((size // tp, 4), value, dtype=torch.float32)
        for size, value in zip((8, 8, 16, 16, 8, 8), (1, 2, 3, 4, 5, 6), strict=True)
    ]
    source = "language_model.decoder.layers.0.self_attention."
    result = Qwen35Convertor(_qwen35_config(tp), strict=True).convert(
        {
            source + "in_proj.weight": torch.cat(parts),
            source + "out_norm.weight": torch.tensor([-0.25, 0.5]),
        }
    )
    target = "model.language_model.layers.0.linear_attn."
    _assert_tensor_bits_equal(
        result[target + "in_proj_qkv.weight"], torch.cat(parts[:3])
    )
    for name, part in zip(("z", "b", "a"), parts[3:], strict=True):
        _assert_tensor_bits_equal(result[target + f"in_proj_{name}.weight"], part)
    _assert_tensor_bits_equal(result[target + "norm.weight"], torch.tensor([0.75, 1.5]))


def test_qwen35_vision_qkv_bias_and_gelu_mlp_are_preserved():
    from rlinf.utils.convertor.utils import Qwen35Convertor

    # Two local heads, each containing Q, K, V with two coordinates.
    bias = torch.tensor(
        [
            1.0,
            2.0,
            11.0,
            12.0,
            21.0,
            22.0,
            3.0,
            4.0,
            13.0,
            14.0,
            23.0,
            24.0,
        ]
    )
    fc1 = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    result = Qwen35Convertor(_qwen35_config(2), strict=True).convert(
        {
            "vision_model.decoder.layers.0.self_attention.linear_qkv.bias": bias,
            "vision_model.decoder.layers.0.mlp.linear_fc1.weight": fc1,
            "vision_model.merger.linear_fc2.bias": torch.arange(4, dtype=torch.float32),
        }
    )
    _assert_tensor_bits_equal(
        result["model.visual.blocks.0.attn.qkv.bias"],
        torch.tensor(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                11.0,
                12.0,
                13.0,
                14.0,
                21.0,
                22.0,
                23.0,
                24.0,
            ]
        ),
    )
    _assert_tensor_bits_equal(
        result["model.visual.blocks.0.mlp.linear_fc1.weight"], fc1
    )
    assert "model.visual.merger.linear_fc2.bias" in result
    assert not any("gate_proj" in key for key in result)


def test_qwen35_mtp_projection_stays_replicated_and_unknown_keys_fail():
    from rlinf.utils.convertor.utils import Qwen35Convertor

    converter = Qwen35Convertor(_qwen35_config(4), strict=True)
    fusion = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    result = converter.convert({"language_model.mtp.layers.0.eh_proj.weight": fusion})
    _assert_tensor_bits_equal(result["mtp.fc.weight"], fusion)
    with pytest.raises(KeyError, match="Unmapped key"):
        converter.convert({"vision_model.unmapped.weight": torch.ones(1)})


@pytest.mark.parametrize("tp, rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
@pytest.mark.parametrize("fused", [False, True])
def test_sglang_native_qkv_full_and_local_weights_match_bitwise(tp, rank, fused):
    linear = pytest.importorskip("sglang.srt.layers.linear")
    pytest.importorskip("sglang.srt.models.qwen3_5")
    from rlinf.hybrid_engines.sglang.common.sgl_scheduler import (
        patch_sglang_presharded_weights,
    )

    layer = linear.QKVParallelLinear(
        hidden_size=4,
        head_size=2,
        total_num_heads=8,
        total_num_kv_heads=4,
        bias=True,
        params_dtype=torch.bfloat16,
        tp_rank=rank,
        tp_size=tp,
    )
    model = torch.nn.ModuleList([layer])
    full_weights = torch.arange(128, dtype=torch.bfloat16).reshape(32, 4)
    full_bias = torch.arange(32, dtype=torch.bfloat16)
    full_weights[0, 0] = -0.0
    reference = []
    for param, full in ((layer.weight, full_weights), (layer.bias, full_bias)):
        param.weight_loader(param, full)
        reference.append(param.detach().clone())

    patch_sglang_presharded_weights(model)
    patch_sglang_presharded_weights(model)
    layer.use_presharded_weights = True
    for param, full, expected in zip(
        (layer.weight, layer.bias), (full_weights, full_bias), reference, strict=True
    ):
        local = [part.chunk(tp, dim=0)[rank] for part in full.split([16, 8, 8])]
        with torch.no_grad():
            param.fill_(float("nan"))
        if fused:
            param.weight_loader(param, torch.cat(local))
        else:
            for shard_id, part in zip("qkv", local, strict=True):
                param.weight_loader(param, part, shard_id)
        _assert_tensor_bits_equal(param, expected)

    layer.use_presharded_weights = False
    for param, full, expected in zip(
        (layer.weight, layer.bias), (full_weights, full_bias), reference, strict=True
    ):
        param.weight_loader(param, full)
        _assert_tensor_bits_equal(param, expected)


@pytest.fixture
def weight_sync_receiver(monkeypatch):
    native = pytest.importorskip("sglang.srt.managers.scheduler")
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    from rlinf.hybrid_engines.sglang.common.sgl_scheduler import Scheduler
    from rlinf.scheduler import Worker
    from rlinf.utils.placement import RolloutSyncMode

    def make(expected, buckets, *, multimodal=False, header=True):
        model = torch.nn.Module()
        for name, value in expected.items():
            model.register_parameter(name, torch.nn.Parameter(torch.zeros_like(value)))
        events, replies = [], []

        def load_weights(weights):
            for name, value in weights:
                if name.startswith("mtp."):
                    continue
                events.append(("load", name))
                default_weight_loader(model.get_parameter(name), value)

        model.load_weights = load_weights
        messages = []
        for index, bucket in enumerate(buckets):
            payload = dict(bucket)
            if index == 0 and header:
                payload["bucket_length"] = torch.tensor([len(buckets)])
            messages.append(payload)
        pending = iter(messages)

        def recv(*args, async_op=False, **kwargs):
            payload = next(pending)
            events.append(("recv", async_op))
            if async_op:
                return SimpleNamespace(wait=lambda: payload)
            return payload

        def flush_cache(self):
            events.append(("flush", tuple(self.waiting_queue)))
            return False

        scheduler = object.__new__(Scheduler)
        scheduler.tp_worker = SimpleNamespace(
            worker=SimpleNamespace(model_runner=SimpleNamespace(model=model))
        )
        scheduler._rlinf_worker = SimpleNamespace(
            recv=recv, send=lambda message, *args: replies.append(message)
        )
        scheduler._actor_group_name = "actor"
        scheduler.actor_weight_rank = 0
        scheduler.cfg = SimpleNamespace(rollout=SimpleNamespace(enforce_eager=False))
        scheduler.model_config = SimpleNamespace(is_multimodal=multimodal)
        scheduler.rollout_sync_mode = RolloutSyncMode.DISAGGREGATED
        scheduler.weight_reload = "sync"
        scheduler.is_weight_offloaded = False
        scheduler.weight_norm_dict = None
        scheduler.running_batch = SimpleNamespace(reqs=[], is_empty=lambda: True)
        scheduler.waiting_queue = []
        scheduler.chunked_req = None
        scheduler.enable_overlap = False
        scheduler._sync_draft_model = None
        monkeypatch.setattr(native.Scheduler, "flush_cache", flush_cache)
        monkeypatch.setattr(
            Worker.torch_platform, "synchronize", lambda: events.append(("sync", None))
        )
        return scheduler, model, events, replies

    return make


@pytest.mark.parametrize("idle", [False, True])
def test_sglang_weight_sync_loads_when_cache_flush_is_declined(
    idle, weight_sync_receiver, monkeypatch
):
    from sglang.srt.managers.scheduler import Scheduler as NativeScheduler

    from rlinf.hybrid_engines.sglang.common.io_struct import SyncHFWeightInput

    expected = {"weight": torch.tensor([[-0.0, 1.0], [2.0, 3.0]])}
    scheduler, model, _, replies = weight_sync_receiver(
        expected, [expected], multimodal=True
    )
    monkeypatch.setattr(NativeScheduler, "is_fully_idle", lambda self: idle)
    scheduler.sync_hf_weight(SyncHFWeightInput())
    _assert_tensor_bits_equal(model.weight, expected["weight"])
    assert replies == []


@pytest.mark.parametrize(
    "multimodal,bucket_count,header",
    [(False, 1, False)]
    + [(mm, count, True) for mm in (False, True) for count in (1, 2, 4)],
)
@pytest.mark.parametrize("pause_mode", [None, "drain", "in_place", "retract"])
def test_sglang_weight_sync_common_bucket_lifecycle(
    multimodal, bucket_count, header, pause_mode, weight_sync_receiver, monkeypatch
):
    from sglang.srt.managers import mm_utils

    from rlinf.hybrid_engines.sglang.common.io_struct import SyncHFWeightInput

    expected = {
        f"weight_{index}": torch.tensor([[-0.0, index + 1.0], [2.0, 3.0]])
        for index in range(bucket_count)
    }
    scheduler, model, events, replies = weight_sync_receiver(
        expected,
        [{name: value} for name, value in expected.items()],
        multimodal=multimodal,
        header=header,
    )
    embedding_cache = {"image": object()}
    monkeypatch.setattr(mm_utils, "embedding_cache", embedding_cache)
    if pause_mode in ("in_place", "retract"):
        scheduler.waiting_queue.append(object())
    waiting_queue = scheduler.waiting_queue
    waiting_requests = list(waiting_queue)
    scheduler.sync_hf_weight(SyncHFWeightInput(pause_mode=pause_mode))
    for name, value in expected.items():
        _assert_tensor_bits_equal(model.get_parameter(name), value)
    assert events[0] == ("sync", None)
    flush_events = [event for event in events if event[0] == "flush"]
    if pause_mode == "in_place":
        assert flush_events == []
    else:
        assert flush_events == [("flush", ())]
        assert events.index(("flush", ())) > max(
            i for i, event in enumerate(events) if event[0] == "load"
        )
    assert scheduler.waiting_queue is waiting_queue
    assert scheduler.waiting_queue == waiting_requests
    assert bool(embedding_cache) is (not multimodal)
    assert replies == []
    assert [event[1] for event in events if event[0] == "recv"] == [False] + [True] * (
        bucket_count - 1
    )


def test_sglang_weight_sync_propagates_loading_failures(weight_sync_receiver):
    from rlinf.hybrid_engines.sglang.common.io_struct import SyncHFWeightInput

    expected = {"weight": torch.tensor([[-0.0, 1.0], [2.0, 3.0]])}
    bucket = {"weight": torch.ones(3, 2)}
    scheduler, _, events, replies = weight_sync_receiver(
        expected, [bucket], multimodal=True
    )
    with pytest.raises(AssertionError):
        scheduler.sync_hf_weight(SyncHFWeightInput())
    assert replies == []
    assert not any(event[0] == "flush" for event in events)


@pytest.mark.parametrize("shared_storage", [False, True])
def test_sglang_weight_sync_loads_mtp_and_refreshes_shared_weights(
    shared_storage, weight_sync_receiver
):
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    from rlinf.hybrid_engines.sglang.common.io_struct import SyncHFWeightInput

    expected = {
        "embedding": torch.tensor([[-0.0, 1.0], [2.0, 3.0]]),
        "head": torch.tensor([[4.0, 5.0], [6.0, 7.0]]),
    }
    mtp_weight = torch.tensor([[8.0, 9.0], [10.0, 11.0]])
    scheduler, target, events, replies = weight_sync_receiver(
        expected, [{"mtp.fc": mtp_weight}, expected], multimodal=True
    )
    draft = torch.nn.Module()
    draft.fc = torch.nn.Parameter(torch.zeros_like(mtp_weight))
    for name in expected:
        source = target.get_parameter(name)
        draft.register_parameter(
            name,
            source if shared_storage else torch.nn.Parameter(torch.zeros_like(source)),
        )

    def load_draft_weights(weights):
        for name, value in weights:
            events.append(("draft_load", name))
            default_weight_loader(draft.get_parameter(name.removeprefix("mtp.")), value)

    target.get_embed_and_head = lambda: (target.embedding, target.head)
    draft.get_embed_and_head = lambda: (draft.embedding, draft.head)
    draft.load_weights = load_draft_weights
    scheduler._sync_draft_model = draft
    scheduler.sync_hf_weight(SyncHFWeightInput())

    for model in (target, draft):
        for name, value in expected.items():
            _assert_tensor_bits_equal(model.get_parameter(name), value)
    _assert_tensor_bits_equal(draft.fc, mtp_weight)
    assert [event for event in events if event[0] == "draft_load"] == [
        ("draft_load", "mtp.fc")
    ]
    assert replies == []
