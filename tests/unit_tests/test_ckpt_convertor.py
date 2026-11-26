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


def _qwen35_mcore_reshard_rank(rank, source_tp, store_path, check_local=False):
    from datetime import timedelta

    from rlinf.utils.resharding.mcore_weight_reshard import MegatronCoreWeightReshard
    from rlinf.utils.resharding.reshard_config import ReshardConfig

    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method=store_path,
        rank=rank,
        world_size=source_tp,
        timeout=timedelta(seconds=90),
    )
    try:
        # Canonical full Megatron weights; segmented tensors concatenate logical parts.
        layouts = {
            "vision_model.pos_embed.weight": ((128, 8), None, None),
            "vision_model.patch_embed.proj.weight": ((8, 3, 2, 2, 2), None, None),
            "vision_model.decoder.layers.0.self_attention.linear_qkv.weight": (
                (24, 8),
                0,
                None,
            ),
            "vision_model.decoder.layers.0.self_attention.linear_qkv.bias": (
                (24,),
                0,
                None,
            ),
            "vision_model.decoder.layers.0.mlp.linear_fc1.weight": ((8, 8), 0, None),
            "vision_model.merger.linear_fc2.weight": ((4, 32), 1, None),
            "vision_model.merger.linear_fc2.bias": ((4,), None, None),
            "language_model.mtp.layers.0.eh_proj.weight": ((4, 8), 0, None),
            "language_model.decoder.layers.0.self_attention.in_proj.weight": (
                (64, 4),
                0,
                (8, 8, 16, 16, 8, 8),
            ),
            "language_model.decoder.layers.0.self_attention.conv1d.weight": (
                (32, 1, 4),
                0,
                (8, 8, 16),
            ),
            "language_model.decoder.layers.0.self_attention.A_log": ((8,), 0, None),
            "language_model.decoder.layers.0.self_attention.dt_bias": ((8,), 0, None),
            "language_model.decoder.layers.0.self_attention.out_norm.weight": (
                (2,),
                None,
                None,
            ),
            "language_model.decoder.layers.0.self_attention.out_proj.weight": (
                (4, 16),
                1,
                None,
            ),
            "language_model.embedding.word_embeddings.weight": ((32, 4), 0, None),
            "language_model.output_layer.weight": ((32, 4), 0, None),
        }
        for prefix in (
            "language_model.decoder.layers.3.",
            "language_model.mtp.layers.0.mtp_model_layer.",
        ):
            layouts[prefix + "mlp.linear_fc1.weight"] = ((16, 4), 0, (8, 8))
            layouts[prefix + "mlp.linear_fc2.weight"] = ((4, 8), 1, None)
            layouts[prefix + "self_attention.linear_qkv.weight"] = ((48, 4), 0, None)
            layouts[prefix + "self_attention.linear_qkv.layer_norm_weight"] = (
                (4,),
                None,
                None,
            )

        full, local = {}, {}
        for index, (name, (shape, dim, segments)) in enumerate(layouts.items()):
            value = (
                (torch.arange(np.prod(shape)) + index * 100)
                .to(torch.bfloat16)
                .reshape(shape)
            )
            value.reshape(-1)[0] = -0.0
            full[name] = value
            if dim is None:
                local[name] = value.clone()
            elif segments:
                local[name] = torch.cat(
                    [
                        part.chunk(source_tp, dim)[rank]
                        for part in value.split(segments, dim)
                    ],
                    dim,
                ).contiguous()
            else:
                local[name] = value.chunk(source_tp, dim)[rank].contiguous()

        if check_local:
            _check_qwen35_local_shards(rank, source_tp, local, layouts)
            return

        for target_tp in (1, 2, 4):
            settings = _qwen35_config(target_tp)
            config = ReshardConfig(
                model_type="qwen3_5",
                model_config=settings.model_config,
                vision_config=settings.vision_config,
                reshard_weights_format="mcore",
                reshard_tp_size=target_tp,
            )
            resharder = MegatronCoreWeightReshard(config)
            gathered = config.tp_gather_fn(
                local, torch.distributed.group.WORLD, "attn", split_fc1=config.split_fc1
            )
            for target_rank in range(target_tp):
                result = resharder.narrow_to_target(gathered, target_rank)
                assert result.keys() == full.keys()
                for name, value in full.items():
                    _, dim, segments = layouts[name]
                    if dim is None:
                        expected = value
                    elif segments:
                        expected = torch.cat(
                            [
                                part.chunk(target_tp, dim)[target_rank]
                                for part in value.split(segments, dim)
                            ],
                            dim,
                        )
                    else:
                        expected = value.chunk(target_tp, dim)[target_rank]
                    _assert_tensor_bits_equal(result[name], expected)
    finally:
        torch.distributed.destroy_process_group()


def _check_qwen35_local_shards(rank, tp, local, layouts):
    from rlinf.utils.resharding.mcore_weight_reshard import MegatronCoreWeightReshard
    from rlinf.utils.resharding.reshard_config import ReshardConfig

    settings = _qwen35_config(tp)
    original_gather = torch.distributed.all_gather
    calls = []

    def counted_gather(*args, **kwargs):
        calls.append(args[1].numel())
        return original_gather(*args, **kwargs)

    torch.distributed.all_gather = counted_gather
    try:
        for target_format in ("mcore", "sglang"):
            for vision_dp in (False, True):
                config = ReshardConfig(
                    model_type="qwen3_5",
                    model_config=settings.model_config,
                    vision_config=settings.vision_config,
                    reshard_weights_format=target_format,
                    reshard_tp_size=tp,
                    mm_enable_dp_encoder=vision_dp,
                )
                resharder = MegatronCoreWeightReshard(config)
                for version in (0, 1):
                    for name, value in local.items():
                        source = {name: value.clone() if version == 0 else -value}
                        calls.clear()
                        full = config.tp_gather_fn(
                            source,
                            torch.distributed.group.WORLD,
                            "attn",
                            split_fc1=config.split_fc1,
                        )
                        expected = resharder.narrow_to_target(full, rank)
                        full_calls = len(calls)
                        calls.clear()
                        fast = config.tp_gather_fn(
                            source,
                            torch.distributed.group.WORLD,
                            "attn",
                            split_fc1=config.split_fc1,
                            tp_gather_is_identity=True,
                        )
                        actual = resharder.narrow_to_target(fast, rank)
                        assert actual.keys() == expected.keys()
                        for key in expected:
                            _assert_tensor_bits_equal(actual[key], expected[key])
                        required = (
                            name.endswith("eh_proj.weight") and config.split_fc1
                        ) or (
                            name.startswith("vision_model.")
                            and layouts[name][1] is not None
                            and vision_dp
                            and tp > 1
                        )
                        assert len(calls) == (full_calls if required else 0), name
                        # Snapshots must survive later mutations/offload of live weights.
                        source[name].zero_()
                        for key in expected:
                            _assert_tensor_bits_equal(actual[key], expected[key])
    finally:
        torch.distributed.all_gather = original_gather


@pytest.mark.parametrize("source_tp", [1, 2, 4])
def test_qwen35_local_shards_match_gather_and_skip_communication(source_tp, tmp_path):
    pytest.importorskip("megatron.training")
    torch.multiprocessing.spawn(
        _qwen35_mcore_reshard_rank,
        args=(source_tp, (tmp_path / "store").as_uri(), True),
        nprocs=source_tp,
        join=True,
    )


@pytest.mark.parametrize("source_tp", [1, 2, 4])
def test_qwen35_mcore_resharding_bitwise_across_tp_layouts(source_tp, tmp_path):
    pytest.importorskip("megatron.training")
    torch.multiprocessing.spawn(
        _qwen35_mcore_reshard_rank,
        args=(source_tp, (tmp_path / "store").as_uri()),
        nprocs=source_tp,
        join=True,
    )


@pytest.mark.parametrize("use_bridge", [False, True])
def test_megatron_reshard_config_uses_effective_model_config(
    use_bridge, weight_sync_cpu_group
):
    pytest.importorskip("megatron.bridge.models.qwen_vl.qwen35_vl_provider")
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider
    from megatron.core.transformer import TransformerConfig

    from rlinf.utils.resharding.reshard_config import ReshardConfig
    from rlinf.workers.megatron_worker import MegatronWorker

    worker = object.__new__(MegatronWorker)
    worker.mbridge = use_bridge
    worker.transformer_config = TransformerConfig(
        num_layers=4, hidden_size=8, num_attention_heads=4
    )
    if use_bridge:
        worker.provider = Qwen35VLModelProvider(
            num_layers=4,
            hidden_size=8,
            num_attention_heads=4,
            num_query_groups=4,
            kv_channels=2,
            ffn_hidden_size=16,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
        )
    model_config = worker.reshard_model_config
    assert model_config is (
        worker.provider if use_bridge else worker.transformer_config
    )
    for target_format in ("mcore", "sglang"):
        config = ReshardConfig(
            model_type="qwen3_5" if use_bridge else "qwen2.5",
            model_config=model_config,
            vision_config=getattr(model_config, "vision_config", None),
            reshard_weights_format=target_format,
        )
        if use_bridge:
            assert config.vision_config is worker.provider.vision_config
            value = torch.arange(256, dtype=torch.float32).reshape(64, 4)
            name = "language_model.decoder.layers.0.self_attention.in_proj.weight"
            gathered = config.tp_gather_fn(
                {name: value}, weight_sync_cpu_group, "attn", split_fc1=config.split_fc1
            )
            _assert_tensor_bits_equal(gathered[name][0], value)


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


@pytest.mark.parametrize("dp_attention", [False, True])
@pytest.mark.parametrize("ffn_size", [6, 8])
def test_qwen35_ffn_validation_uses_its_target_tp(dp_attention, ffn_size):
    from rlinf.utils.convertor.utils import Qwen35Convertor

    cfg = _qwen35_config(2)
    cfg.enable_dp_attention = dp_attention
    cfg.rollout_full_tp_size = 4
    cfg.model_config.ffn_hidden_size = ffn_size
    if dp_attention and ffn_size == 6:
        with pytest.raises(AssertionError, match="FFN TP=4"):
            Qwen35Convertor(cfg)
    else:
        Qwen35Convertor(cfg)
