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

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from rlinf.quantization.backends import create_quantization_backend as _backend
from rlinf.quantization.backends.nvfp4 import (
    dequantize_nvfp4_blocks,
    fake_quantize_nvfp4_blocks,
    quantize_nvfp4_blocks,
)
from rlinf.quantization.config import (
    BackendCapability,
    TargetManifestEntry,
    parse_quantization_spec,
    resolve_quantization_profile,
)

_SUPPORTED_PROFILES = frozenset({"int8_block128", "nvfp4"})
_NVIDIA_CUDA = torch.cuda.is_available() and torch.version.cuda is not None
_FP8_CUDA = _NVIDIA_CUDA and torch.cuda.get_device_capability() >= (8, 9)
_NVFP4_CAPABILITIES = {(10, 0), (10, 3), (12, 0), (12, 1)}
_NVFP4_GEMM_CUDA = (
    _NVIDIA_CUDA and torch.cuda.get_device_capability() in _NVFP4_CAPABILITIES
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_int8_rtn_rounds_positive_and_negative_midpoints_to_even(dtype):
    from rlinf.quantization.backends.int8 import quantize_int8_blocks

    values = torch.tensor([[-127, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 127]], dtype=dtype)
    quantized, scales = quantize_int8_blocks(values, 128)
    assert quantized.tolist() == [[-127, -2, -2, 0, 0, 2, 2, 127]]
    torch.testing.assert_close(scales, torch.ones(1, 1), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_int8_fake_linear_preserves_recipe_and_gradients(dtype):
    backend = _backend("int8", block_size=128)
    values = torch.linspace(-2, 3, 34).reshape(2, 17).to(dtype).requires_grad_()
    weight = torch.linspace(-3, 2, 51).reshape(3, 17).to(dtype).requires_grad_()
    bias = torch.linspace(-0.07, 0.11, 3).to(dtype).requires_grad_()
    expected = F.linear(
        _reference_fake_int8_blocks(values, 128),
        _reference_fake_int8_blocks(weight, 128),
        bias,
    )
    actual = backend.fake_linear(values, weight, bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_grads = torch.autograd.grad(expected.float().sum(), (values, weight, bias))
    actual_grads = torch.autograd.grad(actual.float().sum(), (values, weight, bias))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    quantized = backend.quantize(weight, role="weight")
    restored = backend.dequantize(quantized)
    assert quantized.logical_shape == tuple(weight.shape)
    assert not quantized.data.requires_grad and not quantized.scales.requires_grad
    assert restored.shape == weight.shape and restored.dtype == torch.float32


def test_int8_backend_retains_model_specific_parameter_dtype_casts():
    backend = _backend("int8", block_size=128)
    values = torch.ones(2, 17)
    weight = torch.ones(3, 17, dtype=torch.bfloat16)
    bias = torch.ones(3, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        backend.fake_linear(values, weight, bias)
    expected = F.linear(
        _reference_fake_int8_blocks(values, 128),
        _reference_fake_int8_blocks(weight, 128).float(),
        bias.float(),
    )
    torch.testing.assert_close(
        backend.fake_linear(values, weight, bias, weight_dtype=values.dtype),
        expected,
        rtol=0,
        atol=0,
    )


def test_int8_backend_rejects_different_logical_widths_with_matching_padding():
    from rlinf.quantization.backends.base import QuantizedTensor

    backend = _backend("int8", block_size=128)
    activation = QuantizedTensor(
        torch.zeros(1, 256, dtype=torch.int8),
        torch.ones(1, 2),
        (1, 129),
        "int8",
        storage="native",
        padded_shape=(1, 256),
    )
    weight = QuantizedTensor(
        torch.zeros(3, 256, dtype=torch.int8),
        torch.ones(3, 2),
        (3, 130),
        "int8",
        storage="native",
        padded_shape=(3, 256),
    )
    with pytest.raises(ValueError, match="logical K"):
        backend.gemm(activation, weight)


def _resolve_pi05_targets(model, selectors):
    from rlinf.quantization.models.openpi.adapter import discover_pi05_targets

    targets = discover_pi05_targets(model)
    config = _config("nvfp4")
    config["targets"] = selectors
    spec = parse_quantization_spec(config)
    capability = BackendCapability("flashinfer_cutlass", "test", _SUPPORTED_PROFILES)
    return resolve_quantization_profile(
        spec, [target.path for target in targets], capability, target_manifest=targets
    ).target_manifest


def test_qat_linear_preserves_master_on_loading_migration_and_unfreeze():
    from rlinf.quantization.layers.linear import FakeQuantLinear

    source = nn.Linear(17, 7).requires_grad_(False)
    layer = FakeQuantLinear.from_linear(source, backend=_backend("int8", block_size=16))
    values = torch.ones(2, 17)
    assert layer.weight is source.weight and layer.bias is source.bias
    with torch.inference_mode():
        before = layer(values)
    values.requires_grad_()
    layer(values).sum().backward()
    assert values.grad is not None and layer.weight.grad is None
    assert tuple(layer.state_dict()) == ("weight", "bias")
    state = {name: value.clone().mul_(2) for name, value in layer.state_dict().items()}
    layer.load_state_dict(state)
    torch.testing.assert_close(layer(values), before * 2, rtol=0, atol=0)
    layer.double()
    layer(values.double())
    layer.weight.requires_grad_()
    layer(values.double()).sum().backward()
    assert torch.count_nonzero(layer.weight.grad) > 0


@pytest.mark.skipif(not _NVIDIA_CUDA, reason="requires CUDA FSDP")
@pytest.mark.parametrize("fmt", ["int8", "nvfp4"])
def test_qat_fsdp_preserves_frozen_prefix_and_trains_expert(tmp_path, fmt):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    from rlinf.quantization.layers.linear import FakeQuantLinear

    if dist.is_initialized():
        pytest.skip("requires its own isolated process group")
    if fmt == "nvfp4" and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("requires FP8 CUDA scales")

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            frozen = nn.Linear(17, 7, device="cuda").requires_grad_(False)
            self.prefix = FakeQuantLinear.from_linear(
                frozen,
                backend=_backend(fmt, block_size=16),
            )
            self.expert = FakeQuantLinear.from_linear(
                nn.Linear(7, 3, device="cuda"),
                backend=_backend(fmt, block_size=16),
            )

        def forward(self, x):
            with torch.no_grad():
                prefix = self.prefix(x)
            return self.expert(prefix)

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1
    )
    try:
        policy = Policy()
        model = FSDP(
            policy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.NO_SHARD,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        values = torch.ones(2, 17, device="cuda")
        first = model(values)
        torch.testing.assert_close(model(values), first, rtol=0, atol=0)
        frozen_before = policy.prefix.weight.detach().clone()
        expert_before = policy.expert.weight.detach().clone()
        first.sum().backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        model(values).sum().backward()
        torch.testing.assert_close(policy.prefix.weight, frozen_before, rtol=0, atol=0)
        assert policy.prefix.weight.grad is None
        assert policy.expert.weight.grad is not None
        assert not torch.equal(policy.expert.weight, expert_before)
        assert set(model.state_dict()) == {
            "prefix.weight",
            "prefix.bias",
            "expert.weight",
            "expert.bias",
        }
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _NVIDIA_CUDA, reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("block_size", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_int8_quantization_matches_block_recipe_and_tail(block_size, dtype):
    from rlinf.quantization.backends.int8 import quantize_int8_blocks
    from rlinf.quantization.kernels.int8 import quantize_int8_cuda_blocks

    values = torch.randn((3, 129), device="cuda", dtype=dtype)
    values[0].zero_()
    values[2].zero_()
    values[2, :8] = torch.tensor(
        [-127, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 127], device="cuda"
    )
    values[2, -1] = -0.25
    actual_q, actual_scales = quantize_int8_cuda_blocks(values, block_size)
    expected_q, expected_scales = quantize_int8_blocks(values, block_size)

    assert actual_q.dtype == torch.int8
    assert actual_q.shape == (3, ((129 + block_size - 1) // block_size) * block_size)
    torch.testing.assert_close(actual_q[:, :129], expected_q, rtol=0, atol=0)
    assert actual_q[2, :8].tolist() == [-127, -2, -2, 0, 0, 2, 2, 127]
    assert torch.count_nonzero(actual_q[:, 129:]) == 0
    torch.testing.assert_close(actual_scales, expected_scales, rtol=1e-6, atol=0)


@pytest.mark.skipif(not _NVIDIA_CUDA, reason="requires NVIDIA CUDA")
@pytest.mark.parametrize(
    "m,k,n,block_size",
    [
        (1, 129, 3, 32),
        (10, 32, 1024, 128),
        (256, 588, 1152, 128),
    ],
)
def test_triton_int8_gemm_matches_integer_reference_without_float_fallback(
    m, k, n, block_size, monkeypatch
):
    from rlinf.quantization.kernels.int8 import (
        int8_block_gemm,
        quantize_int8_cuda_blocks,
    )

    generator = torch.Generator(device="cuda").manual_seed(20261013)
    values = torch.randn((m, k), device="cuda", generator=generator)
    weights = torch.randn((n, k), device="cuda", generator=generator)
    values_q, activation_scales = quantize_int8_cuda_blocks(values, block_size)
    weights_q, weight_scales = quantize_int8_cuda_blocks(weights, block_size)

    def reject_fallback(*args, **kwargs):
        raise AssertionError("INT8 GEMM must not use float linear")

    with monkeypatch.context() as guard:
        guard.setattr(F, "linear", reject_fallback)
        actual = int8_block_gemm(values_q, activation_scales, weights_q, weight_scales)
    expected = _reference_block_gemm(values, weights, block_size)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-5)


@pytest.mark.skipif(not _NVIDIA_CUDA, reason="requires NVIDIA CUDA")
def test_triton_int8_bias_is_added_before_bf16_output_cast():
    from rlinf.quantization.kernels.int8 import (
        int8_block_gemm,
        quantize_int8_cuda_blocks,
    )

    values = torch.ones((3, 129), device="cuda", dtype=torch.bfloat16)
    weights = torch.ones((5, 129), device="cuda", dtype=torch.bfloat16)
    bias = torch.tensor([0.125, -0.5, 0, 1, -2], device="cuda")
    values_q, activation_scales = quantize_int8_cuda_blocks(values)
    weights_q, weight_scales = quantize_int8_cuda_blocks(weights)
    actual = int8_block_gemm(
        values_q,
        activation_scales,
        weights_q,
        weight_scales,
        bias=bias,
        output_dtype=torch.bfloat16,
    )
    expected = (torch.full((3, 5), 129.0, device="cuda") + bias).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _reference_block_gemm(values, weights, block_size):
    from rlinf.quantization.backends.int8 import quantize_int8_blocks

    aq, asc = quantize_int8_blocks(values, block_size)
    wq, wsc = quantize_int8_blocks(weights, block_size)
    m, k = values.shape
    n = weights.shape[0]
    padded_m = max(32, ((m + 7) // 8) * 8)
    padded_n = ((n + 7) // 8) * 8
    padded_k = ((k + block_size - 1) // block_size) * block_size
    aq = F.pad(aq, (0, padded_k - k, 0, padded_m - m))
    wq = F.pad(wq, (0, padded_k - k, 0, padded_n - n))
    output = torch.zeros((m, n), device=values.device, dtype=torch.float32)
    for block, start in enumerate(range(0, padded_k, block_size)):
        product = torch._int_mm(
            aq[:, start : start + block_size].contiguous(),
            wq[:, start : start + block_size].T.contiguous(),
        )[:m, :n]
        output += product.float() * asc[:, block, None] * wsc[None, :, block]
    return output


@pytest.mark.skipif(not _NVIDIA_CUDA, reason="requires NVIDIA CUDA")
def test_native_int8_linear_quantizes_weight_only_when_version_is_loaded():
    from rlinf.quantization.layers.linear import QuantizedLinear

    source = nn.Linear(129, 3, bias=False, device="cuda", dtype=torch.bfloat16)
    source.weight.data.fill_(0.5)
    native = QuantizedLinear.from_linear(
        source, backend=_backend("int8", block_size=128)
    )
    native.load_weights()
    values = torch.ones((2, 129), device="cuda", dtype=torch.bfloat16)
    first = native(values)
    dynamic_activation = native(values * 2)
    torch.testing.assert_close(dynamic_activation, first * 2, rtol=0, atol=0)
    source.weight.data.mul_(2)
    torch.testing.assert_close(native(values), first, rtol=0, atol=0)
    native.load_weights()
    torch.testing.assert_close(native(values), first * 2, rtol=0, atol=0)


def _config(format_name="int8"):
    tensor_recipe = {
        "format": format_name,
        "granularity": "per_block",
        "axis": "input",
        "block_size": 16 if format_name == "nvfp4" else 128,
        "scale": "dynamic",
    }
    return {
        "enabled": True,
        "weight": tensor_recipe.copy(),
        "activation": tensor_recipe.copy(),
        "rounding": "nearest_even",
        "targets": ["encoder.*", "action.proj"],
    }


def _projection_reference(values, weights, bias=None, *, format_name):
    if format_name == "int8":
        return F.linear(
            _reference_fake_int8_blocks(values, 128),
            _reference_fake_int8_blocks(weights, 128),
            bias,
        )
    result = F.linear(
        fake_quantize_nvfp4_blocks(values).bfloat16(),
        fake_quantize_nvfp4_blocks(weights).bfloat16(),
    ).float()
    if bias is not None:
        result = result + bias.float()
    return result.to(values.dtype)


@pytest.mark.parametrize("stage", ["actor", "rollout"])
@pytest.mark.parametrize(
    "quantization_format,block_size",
    [
        ("nvfp4", 16),
        ("int8", 128),
    ],
)
def test_pi05_adapter_applies_explicit_recipe_to_all_projections(
    stage, quantization_format, block_size
):
    from rlinf.quantization.models.openpi.adapter import build_adapter

    model = _TinyPi05()
    model.img.eval().requires_grad_(False)
    model.llm.requires_grad_(False)
    model.img.stem._fsdp_wrap_name = "patch"
    parameters = dict(model.named_parameters())
    training_scope = {name: p.requires_grad for name, p in parameters.items()}
    keys = tuple(model.state_dict())
    targets = _resolve_pi05_targets(model, ("*",))
    adapter = build_adapter()
    options = {
        "backend": _backend(
            quantization_format,
            block_size=block_size,
        )
    }
    if stage == "actor":
        selected = adapter.apply_actor(model, targets, **options)
    else:
        selected = adapter.apply_rollout(model, targets, **options)
    assert selected == targets
    assert tuple(model.state_dict()) == keys
    assert all(
        dict(model.named_parameters())[key] is weight
        for key, weight in parameters.items()
    )
    assert not model.img.training
    assert {
        name: p.requires_grad for name, p in model.named_parameters()
    } == training_scope
    assert model.img.stem._fsdp_wrap_name == "patch"
    for projection in (
        model.llm.proj,
        model.llm.ff,
        model.img.stem,
        model.img.encoder.attn,
        model.img.head,
        model.action_in_proj,
        model.action_out_proj,
        model.time_mlp_in,
        model.time_mlp_out,
    ):
        assert projection.backend.quantization_format == quantization_format
        assert projection.backend.block_size == block_size
    if stage == "rollout":
        with pytest.raises(RuntimeError, match="not loaded"):
            model.llm.proj(torch.ones(1, 4))


def test_pi05_same_float_master_checkpoint_loads_after_fresh_format_selection():
    from rlinf.quantization.models.openpi.adapter import build_adapter

    master = _TinyPi05().state_dict()
    adapter = build_adapter()
    for mode in ("bf16", "int8_qat", "nvfp4_qat", "int8_native", "nvfp4_native"):
        model = _TinyPi05().bfloat16()
        if mode != "bf16":
            format_name, stage = mode.split("_")
            apply = adapter.apply_actor if stage == "qat" else adapter.apply_rollout
            apply(
                model,
                _resolve_pi05_targets(model, ("*",)),
                backend=_backend(
                    format_name, block_size=16 if format_name == "nvfp4" else 128
                ),
            )
        model.load_state_dict(master, strict=True)
        assert tuple(model.state_dict()) == tuple(master)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, master[key].bfloat16(), rtol=0, atol=0)


def test_config_disabled_is_noop():
    from rlinf.quantization.config import QuantizationSpec

    assert parse_quantization_spec(None) == QuantizationSpec(enabled=False)
    assert parse_quantization_spec({"enabled": False}) == QuantizationSpec(
        enabled=False
    )


@pytest.mark.parametrize("format_name", ["int8", "nvfp4"])
def test_config_resolves_targets_and_stable_shape_sensitive_identity(format_name):
    spec = parse_quantization_spec(_config(format_name))
    backend_name = "flashinfer_cutlass" if format_name == "nvfp4" else "triton_int8"
    capability = BackendCapability(backend_name, "test", _SUPPORTED_PROFILES)
    manifest = [
        TargetManifestEntry("encoder.q", "encoder.q.weight", (3, 17), "linear"),
        TargetManifestEntry("action.proj", "action.proj.weight", (4, 17), "linear"),
    ]
    profile = resolve_quantization_profile(
        spec, [x.path for x in manifest], capability, target_manifest=manifest
    )
    assert profile.targets == ("action.proj", "encoder.q")
    assert (
        profile.config_hash
        == resolve_quantization_profile(
            spec,
            [x.path for x in reversed(manifest)],
            capability,
            target_manifest=manifest,
        ).config_hash
    )
    manifest[1] = TargetManifestEntry(
        "action.proj", "action.proj.weight", (4, 18), "linear"
    )
    assert (
        profile.config_hash
        != resolve_quantization_profile(
            spec, [x.path for x in manifest], capability, target_manifest=manifest
        ).config_hash
    )


@pytest.mark.parametrize(
    "format_name,section,field,value,match",
    [
        ("int8", None, "rounding", "away_from_zero", "rounding"),
        ("int8", "weight", "granularity", "per_channel", "granularity"),
        ("int8", "weight", "block_size", 0, "block_size"),
        ("int8", "activation", "axis", "output", "not executable"),
        ("nvfp4", "weight", "block_size", 128, "not executable"),
        ("nvfp4", "activation", "format", "fp8", "not executable"),
        ("nvfp4", "weight", "scale", "static", "not executable"),
    ],
)
def test_config_rejects_unsupported_recipes(format_name, section, field, value, match):
    config = _config(format_name)
    (config if section is None else config[section])[field] = value
    with pytest.raises(ValueError, match=match):
        spec = parse_quantization_spec(config)
        resolve_quantization_profile(
            spec,
            ["encoder.q", "action.proj"],
            BackendCapability("flashinfer_cutlass", "test", _SUPPORTED_PROFILES),
        )


def test_config_rejects_zero_layer_match_and_unsupported_recipe():
    spec = parse_quantization_spec(_config())
    capability = BackendCapability(
        backend="test",
        version="1",
        supported_profiles=_SUPPORTED_PROFILES,
    )
    with pytest.raises(ValueError, match="matched no layers"):
        resolve_quantization_profile(
            spec, available_targets=["value.proj"], capability=capability
        )
    with pytest.raises(ValueError, match="not supported"):
        resolve_quantization_profile(
            spec,
            available_targets=["encoder.q", "action.proj"],
            capability=BackendCapability(
                backend="test", version="1", supported_profiles=frozenset()
            ),
        )
    with pytest.raises(ValueError, match="backend"):
        resolve_quantization_profile(
            parse_quantization_spec(_config("nvfp4")),
            ["encoder.q", "action.proj"],
            capability,
        )


def test_config_embodied_entry_rejects_incomplete_runtime_configuration():
    from rlinf.quantization.config import validate_qat_config

    cfg = OmegaConf.create({"qat": _config(), "actor": {}, "rollout": {}})
    with pytest.raises(ValueError, match="runner.task_type"):
        validate_qat_config(cfg)


def test_math_nearest_even_rounding_and_int8_saturation():
    from rlinf.quantization.backends.int8 import quantize_int8_values

    values = torch.tensor([-129.5, -128.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 127.5])
    assert quantize_int8_values(values, torch.tensor(1.0)).tolist() == [
        -128,
        -128,
        -2,
        -2,
        0,
        0,
        2,
        2,
        127,
    ]


@pytest.mark.parametrize("k", [1, 128, 129])
def test_int8_block_scales_cover_tail_zero_and_dynamic_rows(k):
    from rlinf.quantization.backends.int8 import quantize_int8_blocks

    values = torch.ones((3, k), dtype=torch.bfloat16)
    values[0, -1] = 2
    values[1].mul_(4)
    values[2].zero_()
    quantized, scales = quantize_int8_blocks(values, 128)
    assert quantized.shape == values.shape and quantized.dtype == torch.int8
    assert scales.shape == (3, (k + 127) // 128) and scales.dtype == torch.float32
    assert scales[0, -1].item() == pytest.approx(2 / 127)
    assert scales[1, -1].item() == pytest.approx(4 / 127)
    assert quantized[0, -1].item() == 127
    assert torch.count_nonzero(quantized[2]) == 0
    torch.testing.assert_close(scales[2], torch.full_like(scales[2], 1e-12))
    _, updated_scales = quantize_int8_blocks(values * 4, 128)
    torch.testing.assert_close(updated_scales[:2], scales[:2] * 4, rtol=0, atol=0)


class _TinyPi05(nn.Module):
    def __init__(self):
        super().__init__()
        from rlinf.models.embodiment.openpi.modules.gemma import FeedForward

        self.llm = nn.Module()
        self.llm.proj = nn.Linear(4, 4)
        self.llm.ff = FeedForward(4, 8)
        self.img = nn.Module()
        self.img.stem = nn.Conv2d(3, 4, kernel_size=2, stride=2)
        self.img.encoder = nn.Module()
        self.img.encoder.attn = nn.MultiheadAttention(4, 2, batch_first=True)
        self.img.head = nn.Linear(4, 4)
        self.action_in_proj = nn.Linear(4, 4)
        self.action_out_proj = nn.Linear(4, 4)
        self.time_mlp_in = nn.Linear(4, 4)
        self.time_mlp_out = nn.Linear(4, 4)
        self.value_head = nn.Linear(4, 1)


def test_pi05_inventory_includes_functional_and_module_weights():
    from rlinf.quantization.models.openpi.adapter import discover_pi05_targets

    targets = {target.path: target for target in discover_pi05_targets(_TinyPi05())}
    assert targets["llm.proj"].weight_key == "llm.proj.weight"
    assert targets["llm.ff.w_gating"].kind == "functional_gating"
    assert targets["llm.ff.w_linear"].kind == "functional_down"
    assert targets["img.stem"].kind == "patch_conv"
    assert targets["img.encoder.attn.in_proj_weight"].kind == "mha_qkv"
    assert targets["img.encoder.attn.out_proj"].kind == "mha_out_proj"
    assert targets["action_in_proj"].kind == "linear"
    assert targets["time_mlp_out"].kind == "linear"
    assert "value_head" not in targets


def test_pi05_inventory_rejects_missing_vision_attention_family():
    from rlinf.quantization.models.openpi.adapter import discover_pi05_targets

    model = _TinyPi05()
    del model.img.encoder.attn
    with pytest.raises(ValueError, match="mha_qkv"):
        discover_pi05_targets(model)


@pytest.mark.skipif(not _NVIDIA_CUDA, reason="requires NVIDIA CUDA")
def test_pi05_native_full_scope_executes_without_float_linear(monkeypatch):
    from rlinf.quantization.models.openpi.adapter import (
        apply_pi05_rollout_quantization,
        load_pi05_rollout_weights,
    )

    model = _TinyPi05().cuda()
    apply_pi05_rollout_quantization(
        model,
        targets=_resolve_pi05_targets(model, ("*",)),
        backend=_backend("int8", block_size=128),
    )
    load_pi05_rollout_weights(model)

    def reject_float_linear(*args, **kwargs):
        raise AssertionError("selected native projection must use INT8 GEMM")

    with monkeypatch.context() as guard:
        guard.setattr(F, "linear", reject_float_linear)
        values = torch.ones((2, 4), device="cuda")
        assert torch.isfinite(model.llm.proj(values)).all()
        assert torch.isfinite(model.llm.ff(values)).all()
        patches = model.img.stem(torch.ones((2, 3, 4, 4), device="cuda"))
        tokens = patches.flatten(2).transpose(1, 2)
        attended, _ = model.img.encoder.attn(tokens, tokens, tokens)
        assert torch.isfinite(model.img.head(attended)).all()
        for name in (
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ):
            assert torch.isfinite(getattr(model, name)(values)).all()


def test_pi05_all_layer_request_fails_atomically_for_unsupported_mha():
    from rlinf.quantization.models.openpi.adapter import apply_pi05_actor_quantization

    model = _TinyPi05()
    model.img.encoder.attn = nn.MultiheadAttention(4, 2, batch_first=True, dropout=0.1)
    original_ids = {name: id(module) for name, module in model.named_modules()}
    with pytest.raises(ValueError, match="SigLIP"):
        apply_pi05_actor_quantization(
            model,
            targets=_resolve_pi05_targets(model, ("*",)),
            backend=_backend("int8", block_size=2),
        )
    assert {name: id(module) for name, module in model.named_modules()} == original_ids


def test_pi05_selector_zero_match_and_linear_only_replacement():
    from rlinf.quantization.layers.linear import FakeQuantLinear
    from rlinf.quantization.models.openpi.adapter import apply_pi05_actor_quantization

    model = _TinyPi05()
    before_keys = set(model.state_dict())
    model.action_in_proj._fsdp_wrap_name = "action_in_proj"
    with pytest.raises(ValueError, match="matched no layers"):
        apply_pi05_actor_quantization(
            model,
            targets=_resolve_pi05_targets(model, ("llm.missing",)),
            backend=_backend("int8", block_size=128),
        )
    targets = apply_pi05_actor_quantization(
        model,
        targets=_resolve_pi05_targets(model, ("action_in_proj",)),
        backend=_backend("int8", block_size=128),
    )
    assert [target.path for target in targets] == ["action_in_proj"]
    assert isinstance(model.action_in_proj, FakeQuantLinear)
    assert model.action_in_proj._fsdp_wrap_name == "action_in_proj"
    assert set(model.state_dict()) == before_keys


def test_pi05_feed_forward_adapter_replaces_selected_projection():
    from rlinf.quantization.models.openpi.actor import FakeQuantFeedForward
    from rlinf.quantization.models.openpi.adapter import apply_pi05_actor_quantization

    model = _TinyPi05()
    original_keys = set(model.state_dict())
    selected = apply_pi05_actor_quantization(
        model,
        targets=_resolve_pi05_targets(model, ("llm.ff.w_gating",)),
        backend=_backend("int8", block_size=2),
    )
    assert [target.path for target in selected] == ["llm.ff.w_gating"]
    assert isinstance(model.llm.ff, FakeQuantFeedForward)
    assert model.llm.ff.quantize_gating
    assert not model.llm.ff.quantize_down
    assert set(model.state_dict()) == original_keys


@pytest.mark.parametrize("format_name", ["int8", "nvfp4"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pi05_feed_forward_quantizes_separate_gate_up_and_down(format_name, dtype):
    from rlinf.models.embodiment.openpi.modules.gemma import FeedForward
    from rlinf.models.embodiment.openpi.modules.utils import gelu_glu
    from rlinf.quantization.models.openpi.actor import FakeQuantFeedForward

    source = FeedForward(17, 20).to(dtype)
    with torch.no_grad():
        source.w_gating[1].mul_(0.001)
    qat = FakeQuantFeedForward.from_feed_forward(
        source,
        backend=_backend(format_name, block_size=16 if format_name == "nvfp4" else 128),
    )
    values = torch.randn(2, 17, dtype=dtype, requires_grad=True)
    gate = _projection_reference(values, source.w_gating[0].T, format_name=format_name)
    up = _projection_reference(values, source.w_gating[1].T, format_name=format_name)
    expected = _projection_reference(
        gelu_glu(gate, up), source.w_linear.T, format_name=format_name
    )
    actual = qat(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    assert torch.count_nonzero(source.w_gating.grad) > 0
    assert torch.count_nonzero(source.w_linear.grad) > 0
    assert values.grad is not None


@pytest.mark.skipif(not _NVFP4_GEMM_CUDA, reason="requires NVFP4 GEMM CUDA")
@pytest.mark.parametrize(
    "m,n,k,dtype",
    [
        (1, 3, 17, torch.float32),
        (40, 33, 588, torch.bfloat16),
    ],
)
def test_nvfp4_cutlass_uses_logical_m_and_aligned_n(monkeypatch, m, n, k, dtype):
    flashinfer = pytest.importorskip(
        "flashinfer", reason="optional FlashInfer is not installed"
    )

    from rlinf.quantization.kernels.nvfp4 import nvfp4_gemm

    backend = _backend("nvfp4", block_size=16)
    activation = backend.quantize(
        torch.randn(m, k, device="cuda"), role="activation", native=True
    )
    weight = backend.quantize(
        torch.randn(n, k, device="cuda"), role="weight", native=True
    )
    bias = torch.randn(n, device="cuda")
    original = flashinfer.mm_fp4

    def reference():
        output = original(
            activation.data,
            weight.data.T,
            activation.scales,
            weight.scales.T,
            alpha=activation.global_scale * weight.global_scale,
            out_dtype=torch.bfloat16,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cutlass",
            use_nvfp4=True,
        )
        return (output[:m, :n].float() + bias).to(dtype)

    expected = reference()
    observed = []

    def observe(a, b, a_descale, b_descale, **kwargs):
        observed.append(
            (
                tuple(a.shape),
                tuple(b.shape),
                a_descale.data_ptr(),
                b_descale.data_ptr(),
                kwargs["backend"],
            )
        )
        return original(a, b, a_descale, b_descale, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(flashinfer, "mm_fp4", observe)
        actual = nvfp4_gemm(activation, weight, bias, dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert observed == [
        (
            (m, activation.padded_shape[1] // 2),
            (weight.padded_shape[1] // 2, (n + 31) // 32 * 32),
            activation.scales.data_ptr(),
            weight.scales.data_ptr(),
            "cutlass",
        )
    ]


@pytest.mark.skipif(not _NVFP4_GEMM_CUDA, reason="requires NVFP4 GEMM CUDA")
@pytest.mark.parametrize("m,n,k", [(1, 33, 17), (40, 256, 4096)])
def test_nvfp4_gemm_autotune_preserves_values_and_complete_writes(monkeypatch, m, n, k):
    flashinfer = pytest.importorskip(
        "flashinfer", reason="optional FlashInfer is not installed"
    )

    backend = _backend("nvfp4", block_size=16)
    generator = torch.Generator(device="cuda").manual_seed(20261119)
    weight = backend.quantize(
        torch.randn(n, k, device="cuda", generator=generator),
        role="weight",
        native=True,
    )
    bias = torch.randn(n, device="cuda", generator=generator)
    inputs = [
        torch.randn(m, k, device="cuda", generator=generator),
        torch.randn(m, k, device="cuda", generator=generator) * 3,
        torch.zeros(m, k, device="cuda"),
    ]
    prepared = [
        backend.quantize(values, role="activation", native=True) for values in inputs
    ]

    original = flashinfer.mm_fp4
    expected = []
    with flashinfer.autotune(False, skip_ops={"fp4_gemm"}):
        for activation in prepared:
            output = original(
                activation.data[:m],
                weight.data[: ((n + 31) // 32) * 32].T,
                activation.scales,
                weight.scales.T,
                alpha=activation.global_scale * weight.global_scale,
                out_dtype=torch.bfloat16,
                block_size=16,
                use_8x4_sf_layout=False,
                backend="cutlass",
                use_nvfp4=True,
            )
            expected.append(output[:m, :n].float() + bias)
    autotune = flashinfer.autotune
    tuning = []
    calls = []

    def observe_tuning(enabled, **kwargs):
        tuning.append(enabled)
        return autotune(enabled, **kwargs)

    def observe(a, b, a_descale, b_descale, **kwargs):
        assert kwargs["backend"] == "cutlass"
        # Initialize after tuning so another candidate cannot hide partial writes.
        output = torch.full(
            (a.shape[0], b.shape[1]),
            torch.nan,
            device=a.device,
            dtype=kwargs["out_dtype"],
        )
        kwargs["out"] = output
        actual = original(a, b, a_descale, b_descale, **kwargs)
        assert actual.data_ptr() == output.data_ptr()
        assert torch.isfinite(output).all()
        calls.append(tuple(output.shape))
        return actual

    with monkeypatch.context() as guard:
        guard.setattr(flashinfer, "autotune", observe_tuning)
        activation = prepared[0]
        with backend.autotune():
            backend.gemm(activation, weight, bias=bias)
        guard.setattr(flashinfer, "mm_fp4", observe)
        for activation, reference in zip(prepared, expected, strict=True):
            actual = backend.gemm(activation, weight, bias=bias)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert tuning == [True]
    assert calls == [(m, (n + 31) // 32 * 32)] * len(inputs)


@pytest.mark.parametrize("format_name", ["int8", "nvfp4"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pi05_vision_patch_and_attention_match_quantized_projections(
    format_name, dtype
):
    from rlinf.quantization.models.openpi.actor import (
        FakeQuantPatchConv2d,
        FakeQuantVisionMultiheadAttention,
    )

    backend = _backend(format_name, block_size=16 if format_name == "nvfp4" else 128)
    patch = nn.Conv2d(3, 4, 2, stride=2).to(dtype)
    attention = nn.MultiheadAttention(4, 2, batch_first=True).to(dtype)
    qat_patch = FakeQuantPatchConv2d.from_conv2d(patch, backend=backend)
    qat_attention = FakeQuantVisionMultiheadAttention.from_mha(
        attention, backend=backend
    )
    values = torch.randn(2, 3, 4, 4, dtype=dtype, requires_grad=True)
    patches = F.unfold(values, kernel_size=2, stride=2).transpose(1, 2)
    tokens = _projection_reference(
        patches, patch.weight.reshape(4, -1), patch.bias, format_name=format_name
    )
    actual_patches = qat_patch(values)
    torch.testing.assert_close(
        actual_patches, tokens.transpose(1, 2).reshape(2, 4, 2, 2), rtol=0, atol=0
    )
    qkv = _projection_reference(
        tokens,
        attention.in_proj_weight,
        attention.in_proj_bias,
        format_name=format_name,
    )
    q, k, v = [
        part.reshape(2, 4, 2, 2).transpose(1, 2) for part in qkv.chunk(3, dim=-1)
    ]
    probabilities = torch.softmax(q @ k.transpose(-1, -2) / (2**0.5), dim=-1)
    context = (probabilities @ v).transpose(1, 2).reshape(2, 4, 4)
    expected = _projection_reference(
        context,
        attention.out_proj.weight,
        attention.out_proj.bias,
        format_name=format_name,
    )
    actual_tokens = actual_patches.flatten(2).transpose(1, 2)
    actual, weights = qat_attention(actual_tokens, actual_tokens, actual_tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(weights, probabilities.mean(dim=1), rtol=0, atol=0)
    actual.float().square().mean().backward()
    for weight in (patch.weight, attention.in_proj_weight, attention.out_proj.weight):
        assert torch.count_nonzero(weight.grad) > 0
    assert values.grad is not None


def test_pi05_vision_mha_rejects_unimplemented_cross_attention():
    from rlinf.quantization.models.openpi.actor import FakeQuantVisionMultiheadAttention

    source = nn.MultiheadAttention(4, 2, batch_first=True, dropout=0.0)
    qat = FakeQuantVisionMultiheadAttention.from_mha(
        source, backend=_backend("int8", block_size=2)
    )
    with pytest.raises(ValueError, match="self-attention"):
        qat(torch.randn(1, 3, 4), torch.randn(1, 3, 4), torch.randn(1, 3, 4))


def test_pi05_vision_mha_matches_plain_attention_when_quantization_disabled():
    from rlinf.quantization.models.openpi.actor import FakeQuantVisionMultiheadAttention

    source = nn.MultiheadAttention(8, 2, batch_first=True, dropout=0.0)
    source.eval()
    qat = FakeQuantVisionMultiheadAttention.from_mha(
        source,
        quantize_qkv=False,
        quantize_out=False,
        backend=_backend("int8", block_size=4),
    )
    values = torch.randn(2, 5, 8)
    expected, expected_weights = source(values, values, values)
    actual, actual_weights = qat(values, values, values)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_weights, expected_weights, atol=1e-5, rtol=1e-5)


def test_pi05_siglip_small_model_runs_qat_patch_attention_and_head():
    from rlinf.models.embodiment.openpi.modules.siglip import SigLIPViT
    from rlinf.quantization.layers.linear import FakeQuantLinear
    from rlinf.quantization.models.openpi.actor import (
        FakeQuantPatchConv2d,
        FakeQuantVisionMultiheadAttention,
    )

    vision = SigLIPViT(variant="mu/14", pool_type="none", num_classes=4)
    vision.stem = FakeQuantPatchConv2d.from_conv2d(
        vision.stem,
        backend=_backend("int8", block_size=128),
    )
    block = vision.encoder.layers[0]
    block.attn = FakeQuantVisionMultiheadAttention.from_mha(
        block.attn,
        backend=_backend("int8", block_size=128),
    )
    vision.head = FakeQuantLinear.from_linear(
        vision.head,
        backend=_backend("int8", block_size=128),
    )
    # SigLIP's zero-initialized classifier would block backbone gradients.
    with torch.no_grad():
        vision.head.weight.fill_(0.1)
    tokens, _ = vision(torch.randn(1, 224, 224, 3))
    assert tokens.shape == (1, 256, 4)
    tokens.square().mean().backward()
    assert torch.count_nonzero(vision.stem.weight.grad) > 0
    assert torch.count_nonzero(block.attn.in_proj_weight.grad) > 0
    assert torch.count_nonzero(vision.head.weight.grad) > 0


def test_backend_factory_rejects_unsupported_formats():
    with pytest.raises(ValueError, match="quantization format.*not supported"):
        _backend("fp8", block_size=16)


def _reference_fake_int8_blocks(values, block_size):
    from rlinf.quantization.backends.int8 import quantize_int8_blocks

    quantized, scales = quantize_int8_blocks(values, block_size)
    k = values.shape[-1]
    flat_scale = scales.repeat_interleave(block_size, dim=-1)[:, :k]
    restored = (quantized.reshape(-1, k).float() * flat_scale).reshape(values.shape)
    return values + (restored.to(values.dtype) - values).detach()


def test_nvfp4_e2m1_values_midpoints_and_two_level_scales():
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    values = torch.stack(
        [
            torch.cat([magnitudes, -magnitudes]),
            torch.cat([midpoints, -midpoints]),
            torch.full((16,), 2688.0),
        ]
    )
    codes, scales, global_scale = quantize_nvfp4_blocks(values)
    assert codes.dtype == torch.uint8 and codes.shape == values.shape
    assert scales.dtype == torch.float8_e4m3fn and scales.shape == (3, 1)
    assert global_scale.dtype == torch.float32 and global_scale.shape == ()
    torch.testing.assert_close(global_scale, torch.tensor(1.0), rtol=0, atol=0)
    torch.testing.assert_close(
        scales.float(), torch.tensor([[1.0], [1.0], [448.0]]), rtol=0, atol=0
    )
    expected_codes = torch.cat([torch.arange(8), torch.arange(8) + 8]).to(torch.uint8)
    torch.testing.assert_close(codes[0], expected_codes, rtol=0, atol=0)
    midpoint_codes = torch.tensor([0, 2, 2, 4, 4, 6, 6, 7], dtype=torch.uint8)
    torch.testing.assert_close(codes[1, :8], midpoint_codes, rtol=0, atol=0)
    torch.testing.assert_close(codes[1, 8:], midpoint_codes + 8, rtol=0, atol=0)
    restored = dequantize_nvfp4_blocks(codes, scales, global_scale, logical_k=16)
    torch.testing.assert_close(restored[[0, 2]], values[[0, 2]], rtol=0, atol=0)

    restored = dequantize_nvfp4_blocks(
        torch.full((1, 16), 3, dtype=torch.uint8),
        torch.tensor([[0.029296875]], dtype=torch.float8_e4m3fn),
        torch.tensor(2.734375 / 2688.0, dtype=torch.float32),
        logical_k=16,
    )
    torch.testing.assert_close(
        restored, torch.full((1, 16), 4.470347994356416e-05), rtol=0, atol=0
    )


@pytest.mark.parametrize("zero", [False, True])
def test_nvfp4_tail_and_zero_preserve_logical_shape_and_global_scale(zero):
    values = (
        torch.zeros((3, 17)) if zero else torch.arange(51).reshape(3, 17).float() - 19
    )
    codes, scales, global_scale = quantize_nvfp4_blocks(values)
    padded_codes, padded_scales, padded_global_scale = quantize_nvfp4_blocks(
        F.pad(values, (0, 15))
    )
    torch.testing.assert_close(codes, padded_codes[:, :17], rtol=0, atol=0)
    torch.testing.assert_close(scales.float(), padded_scales.float(), rtol=0, atol=0)
    torch.testing.assert_close(global_scale, padded_global_scale, rtol=0, atol=0)
    actual = dequantize_nvfp4_blocks(padded_codes, padded_scales, global_scale, 17)
    assert actual.shape == values.shape
    if zero:
        assert global_scale == 1
        assert torch.count_nonzero(codes) == torch.count_nonzero(scales.float()) == 0
        torch.testing.assert_close(actual, values, rtol=0, atol=0)


def test_nvfp4_linear_preserves_bf16_forward_and_master_gradients_under_autocast():
    from rlinf.quantization.layers.linear import FakeQuantLinear

    source = nn.Linear(17, 3)
    source.weight.data.copy_(torch.linspace(-1000, 1000, 51).reshape(3, 17))
    source.bias.data.copy_(torch.tensor([0.03, -0.07, 0.11]))
    layer = FakeQuantLinear.from_linear(
        source, backend=_backend("nvfp4", block_size=16)
    )
    values = torch.linspace(-1000, 1000, 34).reshape(2, 17).requires_grad_()
    activation = fake_quantize_nvfp4_blocks(values).detach().bfloat16()
    weight = fake_quantize_nvfp4_blocks(source.weight).detach().bfloat16()
    expected = (activation @ weight.T).float() + source.bias.detach()
    with torch.autocast("cpu", dtype=torch.float16):
        actual = layer(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    actual.backward(gradient)
    torch.testing.assert_close(
        values.grad, (gradient.bfloat16() @ weight).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        source.weight.grad, (gradient.bfloat16().T @ activation).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(source.bias.grad, gradient.sum(0), rtol=0, atol=0)


def _pack_native(values):
    pytest.importorskip("triton", reason="optional Triton is not installed")
    return _backend("nvfp4", block_size=16).quantize(
        values, role="activation", native=True
    )


def _native_dequantize(packed):
    pytest.importorskip("flashinfer", reason="optional FlashInfer is not installed")
    from flashinfer.quantization import e2m1_and_ufp8sf_scale_to_float

    rows, k = packed.logical_shape
    values = e2m1_and_ufp8sf_scale_to_float(
        packed.data.cpu(),
        packed.scales.cpu(),
        packed.global_scale.cpu().reshape(1),
        sf_vec_size=16,
        ufp8_type=1,
        is_sf_swizzled_layout=True,
        is_sf_8x4_layout=False,
    )
    return values[:rows, :k]


@pytest.mark.skipif(not _FP8_CUDA, reason="requires an FP8-capable NVIDIA CUDA device")
@pytest.mark.parametrize("rows,k", [(3, 17), (129, 65)])
def test_nvfp4_native_pack_matches_logical_values_with_tail_and_scale_padding(rows, k):
    generator = torch.Generator(device="cuda").manual_seed(20261014)
    values = torch.randn((rows, k), device="cuda", generator=generator)
    midpoints = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device="cuda"
    )
    values[0, :16] = torch.cat([midpoints, -midpoints])
    values[1].fill_(2688)
    values[2, :16] = 1e-30
    backend = _backend("nvfp4", block_size=16)
    packed = _pack_native(values)
    expected = fake_quantize_nvfp4_blocks(values).cpu()
    logical = backend.quantize(values, role="activation")
    torch.testing.assert_close(
        backend.dequantize(logical).bfloat16().float().cpu(), expected, rtol=0, atol=0
    )
    torch.testing.assert_close(
        backend.dequantize(packed).bfloat16().float().cpu(), expected, rtol=0, atol=0
    )
    native = _native_dequantize(packed).to(torch.bfloat16).float()
    torch.testing.assert_close(native, expected, rtol=0, atol=0)
    assert torch.count_nonzero(native[2, :16]) == 0


@pytest.mark.skipif(
    not _NVFP4_GEMM_CUDA, reason="requires SM100/SM120-family NVFP4 GEMM"
)
@pytest.mark.parametrize("m,n,k", [(512, 1152, 588), (1, 3, 17)])
def test_nvfp4_gemm_executes_native_provider_and_matches_packed_reference(
    m, n, k, monkeypatch
):
    flashinfer = pytest.importorskip(
        "flashinfer", reason="optional FlashInfer is not installed"
    )
    from rlinf.quantization.kernels.nvfp4 import nvfp4_gemm

    generator = torch.Generator(device="cuda").manual_seed(20261014)
    values = torch.randn((m, k), device="cuda", generator=generator)
    weights = torch.randn((n, k), device="cuda", generator=generator)
    activation_pack, weight_pack = _pack_native(values), _pack_native(weights)
    native_mm = flashinfer.mm_fp4
    used_native = False

    def observe_mm(*args, **kwargs):
        nonlocal used_native
        used_native = True
        return native_mm(*args, **kwargs)

    monkeypatch.setattr(flashinfer, "mm_fp4", observe_mm)
    actual = nvfp4_gemm(activation_pack, weight_pack)
    assert used_native
    activation = _native_dequantize(activation_pack).cuda()
    weight = _native_dequantize(weight_pack).cuda()
    expected = (activation @ weight.T).to(torch.bfloat16).float()
    assert actual.shape == (m, n) and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.125)


@pytest.mark.skipif(
    not _NVFP4_GEMM_CUDA, reason="requires SM100/SM120-family NVFP4 GEMM"
)
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_nvfp4_gemm_adds_bias_in_fp32_after_bf16_rounding(output_dtype):
    pytest.importorskip("flashinfer", reason="optional FlashInfer is not installed")
    from rlinf.quantization.kernels.nvfp4 import nvfp4_gemm

    activation_pack = _pack_native(torch.ones((1, 17), device="cuda"))
    weight_pack = _pack_native(torch.ones((3, 17), device="cuda"))
    bias = torch.tensor([0.03, -0.07, 0.11], device="cuda")
    rounded = nvfp4_gemm(activation_pack, weight_pack, output_dtype=torch.bfloat16)
    actual = nvfp4_gemm(
        activation_pack, weight_pack, bias=bias, output_dtype=output_dtype
    )
    expected = (rounded.float() + bias).to(output_dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if output_dtype == torch.float32:
        assert not torch.equal(actual, expected.to(torch.bfloat16).float())


@pytest.mark.skipif(not _FP8_CUDA, reason="requires an FP8-capable NVIDIA CUDA device")
def test_nvfp4_gemm_rejects_cross_device_scale():
    from dataclasses import replace

    from rlinf.quantization.kernels.nvfp4 import nvfp4_gemm

    activation = _pack_native(torch.ones((1, 17), device="cuda"))
    weight = _pack_native(torch.ones((3, 17), device="cuda"))
    with pytest.raises(ValueError, match="device"):
        nvfp4_gemm(activation, replace(weight, global_scale=weight.global_scale.cpu()))


def _modelopt_quantizer():
    pytest.importorskip("triton", reason="optional Triton is not installed")
    pytest.importorskip("modelopt", reason="optional ModelOpt is not installed")
    from modelopt.torch.quantization.config import QuantizerAttributeConfig
    from modelopt.torch.quantization.nn import TensorQuantizer

    return TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(2, 1),
            block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        )
    ).cuda()


@pytest.mark.skipif(not _FP8_CUDA, reason="requires an FP8-capable NVIDIA CUDA device")
@pytest.mark.parametrize("rows,k", [(1, 17), (512, 588), (16, 2048)])
@pytest.mark.parametrize("magnitude", [0.001, 1.0, 1000.0])
def test_nvfp4_matches_independent_dynamic_modelopt_tensor_quantizer(
    rows, k, magnitude
):
    quantizer = _modelopt_quantizer()
    generator = torch.Generator(device="cuda").manual_seed(42)
    values = torch.randn((rows, k), device="cuda", generator=generator) * magnitude
    canonical = values.to(torch.bfloat16)
    padded = F.pad(canonical, (0, ((k + 15) // 16) * 16 - k))
    expected = quantizer(padded)[:, :k].float()
    torch.testing.assert_close(
        fake_quantize_nvfp4_blocks(values), expected, rtol=0, atol=0
    )
    native = _native_dequantize(_pack_native(values)).to(torch.bfloat16).float().cuda()
    torch.testing.assert_close(native, expected, rtol=0, atol=0)


@pytest.mark.skipif(not _FP8_CUDA, reason="requires an FP8-capable NVIDIA CUDA device")
def test_nvfp4_preserves_values_below_modelopt_reference_floor():
    quantizer = _modelopt_quantizer()
    values = torch.full((1, 16), 1e-6, dtype=torch.float32, device="cuda")
    actual = fake_quantize_nvfp4_blocks(values)
    reference = quantizer(values.to(torch.bfloat16))
    assert torch.count_nonzero(reference) == 0
    assert torch.count_nonzero(actual) == values.numel()
    native = _native_dequantize(_pack_native(values)).to(torch.bfloat16).float().cuda()
    torch.testing.assert_close(native, actual, rtol=0, atol=0)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2
    or not _NVFP4_GEMM_CUDA
    or torch.cuda.get_device_capability(1) not in _NVFP4_CAPABILITIES,
    reason="requires two SM100/SM120-family CUDA devices",
)
def test_nvfp4_pack_and_gemm_use_input_device_and_restore_current_device():
    pytest.importorskip("flashinfer", reason="optional FlashInfer is not installed")
    from rlinf.quantization.kernels.nvfp4 import nvfp4_gemm

    with torch.cuda.device(0):
        values = torch.ones((1, 17), device="cuda:1")
        weights = torch.ones((3, 17), device="cuda:1")
        activation, weight = _pack_native(values), _pack_native(weights)
        for packed in (activation, weight):
            assert (
                packed.data.device
                == packed.scales.device
                == packed.global_scale.device
                == values.device
            )
        assert torch.cuda.current_device() == 0
        actual = nvfp4_gemm(activation, weight)
        assert actual.device == values.device and torch.cuda.current_device() == 0
        torch.testing.assert_close(
            actual, torch.full((1, 3), 17.0, device=values.device), rtol=0, atol=0
        )
