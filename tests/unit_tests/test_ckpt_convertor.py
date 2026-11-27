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


@pytest.mark.parametrize(
    "path,value",
    [
        ("actor.model.pipeline_model_parallel_size", 2),
        ("rollout.pipeline_parallel_size", 2),
        ("actor.model.context_parallel_size", 2),
        ("actor.model.virtual_pipeline_model_parallel_size", 2),
        ("actor.model.expert_model_parallel_size", 2),
        ("actor.megatron.mbridge", False),
        ("rollout.rollout_backend", "vllm"),
        ("rollout.enforce_eager", True),
        ("rollout.sglang.enable_dp_attention", True),
        ("rollout.sglang.mm_enable_dp_encoder", True),
        ("rollout.sglang.enable_dp_lm_head", True),
        ("rollout.sglang.enable_ep_moe", True),
        ("rollout.sglang.ep_size", 2),
        ("rollout.sglang.moe_dense_tp_size", 1),
        ("rollout.sglang.quantization", "fp8"),
        ("rollout.sglang.speculative_draft_model_quantization", "fp8"),
        ("rollout.sglang.speculative_eagle_topk", 2),
    ],
)
def test_qwen35_config_preserves_parallel_and_backend_options(path, value):
    from omegaconf import OmegaConf

    from rlinf.config import validate_rollout_cfg

    cfg = OmegaConf.create(
        {
            "actor": {
                "training_backend": "megatron",
                "megatron": {"mbridge": True},
                "model": {"load_mtp": True},
            },
            "rollout": {
                "model": {"model_type": "qwen3_5", "model_path": "/unused/checkpoint"},
                "sglang": {},
                "vllm": {},
                "enforce_eager": False,
                "tensor_parallel_size": 4,
                "return_logprobs": True,
            },
        }
    )
    OmegaConf.update(cfg, path, value)
    cfg.rollout = validate_rollout_cfg(cfg.rollout, OmegaConf.create({}), cfg.actor)
    assert OmegaConf.select(cfg, path) == value


@pytest.mark.parametrize("vision_dp", [None, False, True])
@pytest.mark.parametrize("load_mtp", [False, True])
def test_qwen35_server_args_preserve_vision_dp(vision_dp, load_mtp, monkeypatch):
    native = pytest.importorskip("sglang.srt.server_args")
    from omegaconf import OmegaConf

    from rlinf.config import validate_rollout_cfg
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

    # ServerArgs initialization probes hardware and reads the checkpoint.
    monkeypatch.setattr(native.ServerArgs, "__post_init__", lambda self: None)
    worker = object.__new__(SGLangWorker)
    worker._rank = 1
    # The port-lock service normally lives in the scheduler process.
    worker._port_lock = SimpleNamespace(acquire=lambda port: True)
    worker.weight_reload = "sync"
    worker._cfg = OmegaConf.create(
        {
            "actor": {
                "training_backend": "megatron",
                "model": {"load_mtp": load_mtp},
            },
            "runner": {"resume_dir": None},
            "rollout": {
                "model": {
                    "model_type": "qwen3_5",
                    "model_path": "/unused/checkpoint",
                    "precision": "bf16",
                },
                "sglang": {}
                if vision_dp is None
                else {"mm_enable_dp_encoder": vision_dp},
                "enforce_eager": False,
                "tensor_parallel_size": 4,
                "return_logprobs": True,
                "validate_weight": False,
                "cuda_graph_max_bs": 8,
                "max_running_requests": 8,
            },
        }
    )
    worker._cfg_rollout = validate_rollout_cfg(
        worker._cfg.rollout, OmegaConf.create({}), worker._cfg.actor
    )
    args = worker._build_server_args()
    assert args["mm_enable_dp_encoder"] is (vision_dp is True)
    assert args["speculative_algorithm"] == ("NEXTN" if load_mtp else None)
    assert args["speculative_draft_model_path"] == (
        "/unused/checkpoint" if load_mtp else None
    )


def _qwen35_config(tp, vision_dp=False):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            share_embeddings_and_output_weights=False,
            num_moe_experts=None,
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
        rollout_vision_tp_size=1 if vision_dp else tp,
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


@pytest.fixture
def weight_sync_cpu_group(tmp_path):
    torch.distributed.init_process_group(
        "gloo", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1
    )
    try:
        yield torch.distributed.group.WORLD
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("vision_dp", [False, True])
@pytest.mark.parametrize("category", ["attn", "dense"])
def test_qwen35_vision_layout_and_native_loading_match_bitwise(
    tp, vision_dp, category, weight_sync_cpu_group
):
    linear = pytest.importorskip("sglang.srt.layers.linear")
    pytest.importorskip("megatron.training")
    from rlinf.hybrid_engines.sglang.common.sgl_scheduler import (
        patch_sglang_presharded_weights,
    )
    from rlinf.utils.resharding.mcore_weight_reshard import MegatronCoreWeightReshard
    from rlinf.utils.resharding.reshard_config import ReshardConfig

    settings = _qwen35_config(tp)
    config = ReshardConfig(
        model_type="qwen3_5",
        model_config=settings.model_config,
        vision_config=settings.vision_config,
        reshard_tp_size=tp,
        mm_enable_dp_encoder=vision_dp,
        rollout_moe_dense_tp_size=1,
    )
    vision_tp = 1 if vision_dp else tp
    assert config.rollout_vision_tp_size == vision_tp
    resharder = MegatronCoreWeightReshard(config)
    # Source names, full shapes, receiver names and their native shard axes.
    parameters = {
        "patch_embed.proj.weight": ((8, 3, 2, 2, 2), "patch_embed.proj.weight", None),
        "patch_embed.proj.bias": ((8,), "patch_embed.proj.bias", None),
        "pos_embed.weight": ((128, 8), "pos_embed.weight", 0),
        "decoder.layers.0.self_attention.linear_proj.weight": (
            (8, 8),
            "blocks.0.attn.proj.weight",
            1,
        ),
        "decoder.layers.0.self_attention.linear_proj.bias": (
            (8,),
            "blocks.0.attn.proj.bias",
            None,
        ),
        "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight": (
            (8,),
            "blocks.0.norm1.weight",
            None,
        ),
        "decoder.layers.0.self_attention.linear_qkv.layer_norm_bias": (
            (8,),
            "blocks.0.norm1.bias",
            None,
        ),
        "decoder.layers.0.mlp.linear_fc1.weight": (
            (8, 8),
            "blocks.0.mlp.linear_fc1.weight",
            0,
        ),
        "decoder.layers.0.mlp.linear_fc1.bias": (
            (8,),
            "blocks.0.mlp.linear_fc1.bias",
            0,
        ),
        "decoder.layers.0.mlp.linear_fc2.weight": (
            (8, 8),
            "blocks.0.mlp.linear_fc2.weight",
            1,
        ),
        "decoder.layers.0.mlp.linear_fc2.bias": (
            (8,),
            "blocks.0.mlp.linear_fc2.bias",
            None,
        ),
        "merger.patch_norm.weight": ((8,), "merger.norm.weight", None),
        "merger.patch_norm.bias": ((8,), "merger.norm.bias", None),
        "merger.linear_fc1.weight": ((32, 32), "merger.linear_fc1.weight", 0),
        "merger.linear_fc1.bias": ((32,), "merger.linear_fc1.bias", 0),
        "merger.linear_fc2.weight": ((4, 32), "merger.linear_fc2.weight", 1),
        "merger.linear_fc2.bias": ((4,), "merger.linear_fc2.bias", None),
    }
    source, full_reference = {}, {}
    for name, (shape, target, dim) in parameters.items():
        value = torch.arange(np.prod(shape), dtype=torch.float32).to(torch.bfloat16)
        value[0] = -0.0
        source["vision_model." + name] = value.reshape(shape)
        full_reference["model.visual." + target] = (value.reshape(shape), dim)
    qkv = {}
    for suffix, shape in (("weight", (24, 8)), ("bias", (24,))):
        value = torch.arange(np.prod(shape), dtype=torch.float32).to(torch.bfloat16)
        value[0] = -0.0
        qkv[suffix] = value.reshape(shape)
        source["vision_model.decoder.layers.0.self_attention.linear_qkv." + suffix] = (
            value.reshape(3, 4, 2, *shape[1:]).transpose(0, 1).reshape(shape)
        )
    gathered = config.tp_gather_fn(source, weight_sync_cpu_group, category)
    for rank in range(tp):
        result = resharder.narrow_to_target(gathered, rank)
        vision_rank = rank % vision_tp
        for name, (full, dim) in full_reference.items():
            expected = full if dim is None else full.chunk(vision_tp, dim)[vision_rank]
            _assert_tensor_bits_equal(result[name], expected)

        # Full checkpoint loading is the oracle for SGLang's fused QKV layout.
        layer = linear.QKVParallelLinear(
            hidden_size=8,
            head_size=2,
            total_num_heads=4,
            bias=True,
            params_dtype=torch.bfloat16,
            tp_rank=vision_rank,
            tp_size=vision_tp,
        )
        for suffix, full in qkv.items():
            param = getattr(layer, suffix)
            param.weight_loader(param, full)
        expected = {
            name: param.detach().clone() for name, param in layer.named_parameters()
        }
        patch_sglang_presharded_weights(layer)
        layer.use_presharded_weights = True
        for suffix in qkv:
            param = getattr(layer, suffix)
            with torch.no_grad():
                param.fill_(float("nan"))
            param.weight_loader(
                param, result["model.visual.blocks.0.attn.qkv." + suffix]
            )
            _assert_tensor_bits_equal(param, expected[suffix])


def test_qwen35_vision_dp_keeps_language_and_mtp_shards(weight_sync_cpu_group):
    pytest.importorskip("megatron.training")
    from rlinf.utils.resharding.mcore_weight_reshard import MegatronCoreWeightReshard
    from rlinf.utils.resharding.reshard_config import ReshardConfig

    settings = _qwen35_config(4)
    source = {
        "language_model.decoder.layers.0.self_attention.in_proj.weight": torch.arange(
            256
        ).reshape(64, 4),
        "language_model.decoder.layers.1.self_attention.linear_qkv.weight": torch.arange(
            192
        ).reshape(48, 4),
        "language_model.mtp.layers.0.eh_proj.weight": torch.arange(32).reshape(4, 8),
    }
    outputs = []
    for vision_dp in (False, True):
        config = ReshardConfig(
            model_type="qwen3_5",
            model_config=settings.model_config,
            vision_config=settings.vision_config,
            reshard_tp_size=4,
            mm_enable_dp_encoder=vision_dp,
        )
        resharder = MegatronCoreWeightReshard(config)
        gathered = config.tp_gather_fn(source, weight_sync_cpu_group, "attn")
        outputs.append(
            [resharder.narrow_to_target(gathered, rank) for rank in range(4)]
        )
    for without_dp, with_dp in zip(*outputs, strict=True):
        assert without_dp.keys() == with_dp.keys()
        for name in without_dp:
            _assert_tensor_bits_equal(without_dp[name], with_dp[name])
        assert with_dp[
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        ].shape == (8, 4)
        assert with_dp[
            "model.language_model.layers.1.self_attn.q_proj.weight"
        ].shape == (8, 4)
        _assert_tensor_bits_equal(
            with_dp["mtp.fc.weight"],
            source["language_model.mtp.layers.0.eh_proj.weight"],
        )


def test_qwen35_vision_dp_dimensions_are_independent_of_language_tp():
    from rlinf.utils.convertor.utils import Qwen35Convertor

    config = _qwen35_config(4, vision_dp=True)
    config.vision_config.num_heads = 2
    config.vision_config.intermediate_size = 6
    source = torch.arange(24).reshape(2, 3, 4)
    result = Qwen35Convertor(config, strict=True).convert(
        {
            "vision_model.decoder.layers.0.self_attention.linear_qkv.bias": source.flatten()
        }
    )
    _assert_tensor_bits_equal(
        result["model.visual.blocks.0.attn.qkv.bias"], source.transpose(0, 1).flatten()
    )


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
        scheduler._rlinf_weight_version = 0
        monkeypatch.setattr(native.Scheduler, "flush_cache", flush_cache)
        monkeypatch.setattr(
            Worker.torch_platform, "synchronize", lambda: events.append(("sync", None))
        )
        return scheduler, model, events, replies

    return make


def test_sglang_weight_sync_publishes_new_version(weight_sync_receiver):
    from rlinf.hybrid_engines.sglang.common.io_struct import SyncHFWeightInput

    expected = {"weight": torch.tensor([[1.0, 2.0]])}
    scheduler, model, _, _ = weight_sync_receiver(expected, [expected])
    result = scheduler.sync_hf_weight(SyncHFWeightInput(weight_version=7))
    _assert_tensor_bits_equal(model.weight, expected["weight"])
    assert result.weight_version == scheduler._rlinf_weight_version == 7


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


@pytest.mark.parametrize("prefix", ["", "language_model."])
def test_pipeline_language_layer_offsets(prefix):
    from rlinf.utils.resharding.utils import _apply_layer_offset

    key = prefix + "decoder.layers.0.self_attention.linear_qkv.weight"
    assert _apply_layer_offset(key, 32) == key.replace("layers.0.", "layers.32.")
    for key in (
        "vision_model.decoder.layers.0.weight",
        "language_model.mtp.layers.0.weight",
    ):
        assert _apply_layer_offset(key, 32) == key


def test_qwen35_dp_attention_embedding_and_ffn(weight_sync_cpu_group):
    linear = pytest.importorskip("sglang.srt.layers.linear")
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    pytest.importorskip("megatron.training")
    from rlinf.utils.resharding.mcore_weight_reshard import MegatronCoreWeightReshard
    from rlinf.utils.resharding.reshard_config import ReshardConfig

    settings = _qwen35_config(2)
    cfg = ReshardConfig(
        model_type="qwen3_5",
        model_config=settings.model_config,
        vision_config=settings.vision_config,
        reshard_tp_size=2,
        rollout_full_tp_size=4,
        enable_dp_attention=True,
    )
    resharder = MegatronCoreWeightReshard(cfg)
    embedding = torch.arange(512, dtype=torch.float32).reshape(128, 4)
    gate, up = (
        torch.arange(32, dtype=torch.float32).reshape(8, 4),
        torch.arange(32, 64, dtype=torch.float32).reshape(8, 4),
    )
    gathered = cfg.tp_gather_fn(
        {"language_model.embedding.word_embeddings.weight": embedding},
        weight_sync_cpu_group,
        "attn",
    )
    gathered.update(
        cfg.tp_gather_fn(
            {
                "language_model.decoder.layers.0.mlp.linear_fc1.weight": torch.cat(
                    (gate, up)
                )
            },
            weight_sync_cpu_group,
            "dense",
        )
    )
    for rank in range(4):
        output = resharder.narrow_to_target(gathered, rank % 2, dst_shared_rank=rank)
        embed = VocabParallelEmbedding(128, 4, enable_tp=False)
        embed.use_presharded_weights = True
        embed.weight.weight_loader(
            embed.weight, output["model.language_model.embed_tokens.weight"]
        )
        _assert_tensor_bits_equal(embed.weight, embedding)
        layer = linear.MergedColumnParallelLinear(
            4, [8, 8], bias=False, tp_rank=rank, tp_size=4
        )
        for i, value in enumerate((gate, up)):
            layer.weight.weight_loader(layer.weight, value, i)
        expected = layer.weight.detach().clone()
        layer.use_presharded_weights = True
        with torch.no_grad():
            layer.weight.fill_(float("nan"))
        for i, name in enumerate(("gate", "up")):
            layer.weight.weight_loader(
                layer.weight,
                output[f"model.language_model.layers.0.mlp.{name}_proj.weight"],
                i,
            )
        _assert_tensor_bits_equal(layer.weight, expected)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("head_tp", [2, 4])
@pytest.mark.parametrize("rollout_pp", [1, 2])
def test_qwen35_dp_attention_output_layer_native_bitwise(
    tied, head_tp, rollout_pp, weight_sync_cpu_group, monkeypatch
):
    vocab = pytest.importorskip("sglang.srt.layers.vocab_parallel_embedding")
    native = pytest.importorskip("sglang.srt.models.qwen3_5")
    pytest.importorskip("megatron.training")
    from megatron.core import parallel_state

    from rlinf.utils.resharding.mcore_weight_reshard import MegatronCoreWeightReshard
    from rlinf.utils.resharding.reshard_config import ReshardConfig

    # CPU collectives stand in for the actor TP group; PP transport is covered
    # by the full-model integration run. Exercise the output-layer bucket here.
    for axis in ("tensor_model", "pipeline_model", "expert_model", "expert_tensor"):
        monkeypatch.setattr(
            parallel_state, f"get_{axis}_parallel_world_size", lambda: 1
        )
        monkeypatch.setattr(
            parallel_state,
            f"get_{axis}_parallel_group",
            lambda: weight_sync_cpu_group,
        )
    monkeypatch.setattr(
        parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: None
    )
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)
    settings = _qwen35_config(2)
    settings.model_config.pipeline_model_parallel_size = 1
    settings.model_config.share_embeddings_and_output_weights = tied
    cfg = ReshardConfig(
        model_type="qwen3_5",
        model_config=settings.model_config,
        vision_config=settings.vision_config,
        reshard_tp_size=2,
        reshard_pp_size=rollout_pp,
        rollout_full_tp_size=4,
        rollout_lm_head_tp_size=head_tp,
        enable_dp_attention=True,
    )
    resharder = MegatronCoreWeightReshard(cfg)
    weight = torch.arange(512, dtype=torch.float32).reshape(128, 4)
    gathered = resharder.gather_full_model(
        {"language_model.output_layer.weight": weight}
    )
    for rank in range(4):
        head_rank = rank % head_tp
        for group in ("tensor_model_parallel", "attention_tp"):
            monkeypatch.setattr(vocab, f"get_{group}_rank", lambda: head_rank)
            size_name = "size" if group == "attention_tp" else "world_size"
            monkeypatch.setattr(vocab, f"get_{group}_{size_name}", lambda: head_tp)
        model = native.Qwen3_5ForConditionalGeneration.__new__(
            native.Qwen3_5ForConditionalGeneration
        )
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(tie_word_embeddings=tied)
        model.pp_group = SimpleNamespace(
            is_first_rank=rollout_pp == 1, is_last_rank=True, world_size=rollout_pp
        )
        model.model = torch.nn.Module()
        if tied and rollout_pp == 1:
            model.model.embed_tokens = vocab.VocabParallelEmbedding(
                128, 4, enable_tp=False, params_dtype=torch.float32
            )
            model.lm_head = model.model.embed_tokens
        else:
            model.lm_head = vocab.ParallelLMHead(
                128, 4, params_dtype=torch.float32, use_attn_tp_group=head_tp == 2
            )
        model.load_weights([("lm_head.weight", weight)])
        expected = model.lm_head.weight.detach().clone()
        with torch.no_grad():
            model.lm_head.weight.fill_(float("nan"))
        model.lm_head.use_presharded_weights = True
        output = resharder.narrow_to_target(
            gathered, rank % 2, dst_shared_rank=rank, dst_lm_head_rank=head_rank
        )
        model.load_weights(output.items())
        _assert_tensor_bits_equal(model.lm_head.weight, expected)


@pytest.mark.parametrize(
    "model_type", ["qwen2.5", "qwen3", "qwen3_moe", "deepseek_v3", "glm4_moe_lite"]
)
def test_output_layer_gather_preserves_lm_head_shards(
    model_type, weight_sync_cpu_group
):
    from rlinf.utils.resharding.utils import get_tp_gather_fn

    weight = torch.arange(128, dtype=torch.float32).reshape(32, 4)
    result = get_tp_gather_fn(model_type)(
        {"output_layer.weight": weight}, weight_sync_cpu_group, "lm_head"
    )
    full, spec = result["output_layer.weight"]
    _assert_tensor_bits_equal(full, weight)
    assert spec == ("slice", 0, "lm_head")


@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("mtp", [False, True])
def test_qwen35_gqa_kv_replication_native_bitwise(tp, mtp, weight_sync_cpu_group):
    linear = pytest.importorskip("sglang.srt.layers.linear")
    pytest.importorskip("megatron.training")
    from rlinf.utils.resharding.mcore_weight_reshard import MegatronCoreWeightReshard
    from rlinf.utils.resharding.reshard_config import ReshardConfig

    settings = _qwen35_config(tp)
    settings.model_config.linear_num_key_heads = 8
    cfg = ReshardConfig(
        model_type="qwen3_5",
        model_config=settings.model_config,
        vision_config=settings.vision_config,
        reshard_tp_size=tp,
        mm_enable_dp_encoder=True,
    )
    source, queries, keys, values = [], [], [], []
    for group in range(4):
        parts = [
            torch.arange(8, dtype=torch.float32).reshape(2, 4) + group * 48 + i * 8
            for i in range(6)
        ]
        source.extend(parts)
        queries.extend([parts[0], parts[2], parts[1], parts[3]])
        keys.append(parts[4])
        values.append(parts[5])
    source[0][0, 0] = -0.0
    prefix = (
        "language_model.mtp.layers.0.mtp_model_layer."
        if mtp
        else "language_model.decoder.layers.3."
    )
    target = (
        "mtp.layers.0.self_attn." if mtp else "model.language_model.layers.3.self_attn."
    )
    gathered = cfg.tp_gather_fn(
        {prefix + "self_attention.linear_qkv.weight": torch.cat(source)},
        weight_sync_cpu_group,
        "attn",
    )
    resharder = MegatronCoreWeightReshard(cfg)
    for rank in range(tp):
        layer = linear.QKVParallelLinear(
            4, 2, 16, 4, bias=False, tp_rank=rank, tp_size=tp
        )
        for name, full in zip("qkv", (queries, keys, values), strict=True):
            layer.weight.weight_loader(layer.weight, torch.cat(full), name)
        expected = layer.weight.detach().clone()
        layer.use_presharded_weights = True
        with torch.no_grad():
            layer.weight.fill_(float("nan"))
        output = resharder.narrow_to_target(gathered, rank)
        for name in "qkv":
            layer.weight.weight_loader(
                layer.weight, output[target + name + "_proj.weight"], name
            )
        _assert_tensor_bits_equal(layer.weight, expected)


@pytest.mark.parametrize("vp_size", [None, 1, 2])
def test_virtual_pipeline_buckets_preserve_repeated_names(vp_size, monkeypatch):
    pytest.importorskip("megatron.training")
    from megatron.core import parallel_state

    from rlinf.utils.resharding.mcore_weight_reshard import MegatronCoreWeightReshard
    from rlinf.utils.resharding.reshard_config import ReshardConfig

    settings = _qwen35_config(1)
    cfg = ReshardConfig(
        model_type="qwen3_5",
        model_config=settings.model_config,
        vision_config=settings.vision_config,
    )
    resharder = MegatronCoreWeightReshard(cfg)
    monkeypatch.setattr(
        parallel_state,
        "get_virtual_pipeline_model_parallel_world_size",
        lambda: vp_size,
    )
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1
    )
    chunks = [torch.nn.Linear(2, 2, bias=False) for _ in range(vp_size or 1)]
    buckets = resharder.divide_model_to_bucket(chunks)
    assert len(buckets) == len(chunks)
    for index, bucket in enumerate(buckets):
        if vp_size == 2:
            value, chunk_index = bucket["weight"]
            assert chunk_index == index
        else:
            value = bucket["weight"]
        _assert_tensor_bits_equal(value, chunks[index].weight)


def test_qwen35_provider_keeps_virtual_pipeline_config(tmp_path, monkeypatch):
    pytest.importorskip("megatron.bridge.models.qwen_vl.qwen35_vl_provider")
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider
    from omegaconf import OmegaConf
    from transformers import Qwen3_5Config

    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        MegatronModelManager,
    )

    config = Qwen3_5Config(architectures=["Qwen3_5ForConditionalGeneration"])
    config.save_pretrained(tmp_path)
    worker = SimpleNamespace(
        _cfg=OmegaConf.create(
            {
                "megatron": {"load": str(tmp_path)},
                "model": {"model_type": "qwen3_5", "load_mtp": False},
                "optim": {"use_distributed_optimizer": False},
            }
        ),
        transformer_config=SimpleNamespace(
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
        ),
        _rank=1,
        is_dedicated_critic_model=False,
    )

    class ModelConstructionReached(Exception):
        pass

    def check_provider(provider, **kwargs):
        assert provider.pipeline_model_parallel_size == 2
        assert provider.virtual_pipeline_model_parallel_size == 2
        raise ModelConstructionReached

    monkeypatch.setattr(
        Qwen35VLModelProvider, "provide_distributed_model", check_provider
    )
    with pytest.raises(ModelConstructionReached):
        MegatronModelManager.setup_mbridge_model_and_optimizer(worker)
