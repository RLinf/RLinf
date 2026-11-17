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

import asyncio
import copy
import importlib
import inspect
import math
import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.algorithms.losses import compute_decoupled_ppo_actor_loss
from rlinf.quantization.algorithms.correction import (
    correction_loss_kwargs,
    parse_correction_spec,
)
from rlinf.quantization.backends.int8 import Int8Backend
from rlinf.quantization.config import BackendCapability, validate_qat_config


class CpuInt8Backend(Int8Backend):
    """Use reference packing and floating GEMM for CPU lifecycle tests."""

    def __init__(self):
        super().__init__(block_size=128)
        self.quantizations = []

    def quantize(self, values, *, role, native=False):
        self.quantizations.append(role)
        logical = super().quantize(values, role=role)
        if not native:
            return logical
        data = torch.nn.functional.pad(
            logical.data, (0, -values.shape[-1] % self.block_size)
        )
        return replace(
            logical, data=data, storage="native", padded_shape=tuple(data.shape)
        )

    def gemm(self, activation, weight, *, bias=None, output_dtype=torch.float32):
        return torch.nn.functional.linear(
            self.dequantize(activation),
            self.dequantize(weight),
            None if bias is None else bias.float(),
        ).to(output_dtype)

    def probe(self, device):
        return BackendCapability("cpu_reference", "test", frozenset({"int8_block128"}))


def test_weight_buffers_are_nonpersistent_reused_and_invalidated():
    from rlinf.quantization.layers.weights import QuantizedWeight

    backend = CpuInt8Backend()
    weight = QuantizedWeight(backend)
    master = nn.Parameter(torch.full((2, 3), 0.5))
    with pytest.raises(RuntimeError, match="not loaded"):
        weight.value
    weight.load_weights(master)
    previous = weight.value
    weight.data = weight.data.clone()
    assert weight.value.data is weight.data and weight.value.data is not previous.data
    pointer = weight.data.data_ptr()
    weight.load_weights(master * 2)
    assert weight.data.data_ptr() == pointer
    assert not tuple(weight.parameters()) and not weight.state_dict()
    weight.clear()
    assert not weight.is_loaded and weight.data.data_ptr() == pointer
    weight.load_weights(master)
    weight.to(dtype=torch.float64)
    assert not weight.is_loaded
    weight.load_weights(master)
    assert weight.value.scales.dtype == torch.float32
    weight.load_state_dict({})
    assert not weight.is_loaded


def test_linear_shares_master_parameters_and_uses_prepared_weights():
    from rlinf.quantization.layers.linear import FakeQuantLinear, QuantizedLinear

    backend = CpuInt8Backend()
    source = nn.Linear(3, 2).eval()
    source.weight.data.fill_(0.5)
    source.bias.data.zero_()
    rng = torch.get_rng_state()
    layer = QuantizedLinear.from_linear(source, backend=backend)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not layer.training and layer.bias is source.bias
    layer.load_state_dict(source.state_dict(), strict=True)
    values = torch.ones(1, 3)
    assert layer.weight is source.weight and tuple(layer.state_dict()) == tuple(
        source.state_dict()
    )
    with pytest.raises(RuntimeError, match="not loaded"):
        layer(values)
    layer.load_weights()
    first = layer(values)
    torch.testing.assert_close(first, source(values))
    layer(values * 2)
    assert backend.quantizations.count("weight") == 1
    source.weight.data.mul_(2)
    torch.testing.assert_close(layer(values), first)
    layer.load_weights()
    torch.testing.assert_close(layer(values), first * 2)
    fake = FakeQuantLinear.from_linear(source, backend=backend)
    assert fake.bias is source.bias and tuple(fake.state_dict()) == tuple(
        source.state_dict()
    )
    fake(values).sum().backward()
    assert fake.weight is source.weight and torch.count_nonzero(source.weight.grad)


def _training_config():
    tensor = {
        "format": "int8",
        "granularity": "per_block",
        "axis": "input",
        "block_size": 128,
        "scale": "dynamic",
    }
    return OmegaConf.create(
        {
            "qat": {
                "enabled": True,
                "weight": tensor,
                "activation": tensor,
                "rounding": "nearest_even",
                "targets": ["*"],
            },
            "runner": {"task_type": "embodied", "weight_sync_interval": 1},
            "algorithm": {
                "loss_type": "decoupled_actor_critic",
                "logprob_type": "chunk_level",
            },
            "actor": {
                "training_backend": "fsdp",
                "model": {
                    "model_type": "openpi",
                    "pi05": True,
                    "precision": "bf16",
                    "model_path": "/same/master",
                    "num_steps": 3,
                    "num_action_chunks": 2,
                    "action_dim": 3,
                    "openpi": {
                        "task": "rl",
                        "config_name": "pi05_test",
                        "model_action_dim": 4,
                        "action_horizon": 4,
                        "torch_compile": False,
                        "noise_method": "flow_sde",
                        "noise_level": 0.5,
                        "joint_logprob": False,
                    },
                },
            },
            "rollout": {
                "generation_backend": "huggingface",
                "pipeline_stage_num": 1,
                "model": {"precision": "bf16"},
            },
            "weight_syncer": {"type": "bucket", "bucket": {"bucket_dtype": None}},
        }
    )


def _nvfp4_training_config():
    cfg = _training_config()
    for tensor in (cfg.qat.weight, cfg.qat.activation):
        tensor.format, tensor.block_size = "nvfp4", 16
    return cfg


def test_runtime_disabled_does_not_import_optional_quantization_dependencies():
    script = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    assert not name.startswith(('triton', 'flashinfer', 'modelopt'))
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from torch import nn
from rlinf.quantization.runtime import configure_quantized_policy
assert configure_quantized_policy(nn.Linear(3, 2), {}, 'actor', 'cpu') is None
"""
    subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True, check=True
    )


def test_qat_presets_select_format_without_changing_training_settings(monkeypatch):
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        version_base="1.1", config_dir=str(root / "examples/embodiment/config")
    ):
        baseline = compose(config_name="calvin_d_d_ppo_openpi_pi05")
        default = compose(config_name="calvin_d_d_ppo_openpi_pi05_qat")
        configs = [
            compose(
                config_name="calvin_d_d_ppo_openpi_pi05_qat",
                overrides=[f"qat={preset}"],
            )
            for preset in ("off", "int8", "nvfp4")
        ]
    assert default == configs[-1]
    reference = OmegaConf.to_container(configs[0], resolve=True)
    reference.pop("qat")
    for cfg, preset in zip(configs, ("off", "int8", "nvfp4")):
        spec = validate_qat_config(cfg)
        assert spec.enabled == (preset != "off")
        if spec.enabled:
            assert spec.weight.format == spec.activation.format == preset
        actual = OmegaConf.to_container(cfg, resolve=True)
        actual.pop("qat")
        assert actual == reference
    cfg = configs[-1]
    assert cfg.env == baseline.env and cfg.actor.optim == baseline.actor.optim
    assert cfg.actor.global_batch_size == baseline.actor.global_batch_size
    assert cfg.actor.micro_batch_size == baseline.actor.micro_batch_size
    assert cfg.algorithm.update_epoch == baseline.algorithm.update_epoch


@pytest.mark.parametrize(
    "path,value,match",
    [
        ("runner.enable_async", True, "enable_async"),
        ("actor.training_backend", "megatron", "training_backend"),
        ("actor.fsdp_config.mixed_precision.param_dtype", "fp16", "param_dtype"),
        ("rollout.enable_cuda_graph", True, "enable_cuda_graph"),
        ("algorithm.recompute_logprobs", True, "recompute_logprobs"),
        ("actor.model.openpi.pytorch_numerics", True, "pytorch_numerics"),
        ("actor.model.openpi.short_memory", True, "short_memory"),
        ("rollout.model.openpi.pytorch_numerics", True, "pytorch_numerics"),
        ("rollout.model.openpi.short_memory", True, "short_memory"),
    ],
)
def test_runtime_config_rejects_unsupported_execution(path, value, match):
    cfg = _training_config()
    OmegaConf.update(cfg, path, value, merge=False)
    with pytest.raises(ValueError, match=match):
        validate_qat_config(cfg)


def test_runtime_config_requires_stochastic_correction_events():
    cfg = _nvfp4_training_config()
    cfg.qat.correction = {"mode": "importance_sampling"}
    cfg.actor.model.openpi.noise_level = 0.0
    with pytest.raises(ValueError, match="noise_level"):
        validate_qat_config(cfg)


def test_runtime_disabled_preserves_model_and_never_probes_backend():
    from rlinf.quantization.runtime import configure_quantized_policy

    model = nn.Linear(3, 2)
    weight = model.weight
    before = copy.deepcopy(model.state_dict())
    assert configure_quantized_policy(model, {}, "actor", "cpu") is None
    assert model.weight is weight
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value)


def test_runtime_enabled_rejects_cpu_before_model_mutation():
    from rlinf.quantization.runtime import configure_quantized_policy

    model = nn.Linear(3, 2)
    with pytest.raises(ValueError, match="CUDA"):
        configure_quantized_policy(model, _training_config(), "actor", "cpu")
    assert type(model) is nn.Linear


@pytest.mark.parametrize(
    "patch,match",
    [
        ({"delta_encoding": True, "init_sync": {}}, "init_sync.enabled"),
        (
            {
                "delta_encoding": False,
                "init_sync": {"enabled": True, "prefixes": ["llm"]},
            },
            "init_sync.prefixes",
        ),
    ],
)
def test_runtime_config_requires_complete_patch_initialization(patch, match):
    cfg = _training_config()
    cfg.weight_syncer = {"type": "patch", "patch": patch}
    with pytest.raises(ValueError, match=match):
        validate_qat_config(cfg)


@pytest.mark.parametrize("delta_encoding", [False, True])
def test_runtime_config_accepts_bit_exact_full_patch_weight_sync(delta_encoding):
    cfg = _training_config()
    cfg.weight_syncer = {
        "type": "patch",
        "patch": {
            "delta_encoding": delta_encoding,
            "init_sync": {"enabled": True, "prefixes": None},
        },
    }
    assert validate_qat_config(cfg).enabled


@pytest.mark.parametrize(
    "path,value",
    [
        ("rollout.model.num_steps", 5),
        ("rollout.model.openpi_data.norm_stats_path", "/different/stats"),
    ],
)
def test_runtime_config_rejects_different_sampling_or_preprocessing_contract(
    path, value
):
    cfg = _training_config()
    OmegaConf.update(cfg, path, value, merge=False)
    with pytest.raises(ValueError, match="sampling or preprocessing contract"):
        validate_qat_config(cfg)


def _weight_runtime():
    from rlinf.quantization.config import (
        parse_quantization_spec,
        resolve_quantization_profile,
    )
    from rlinf.quantization.layers.weights import QuantizedWeight
    from rlinf.quantization.runtime import QuantizedPolicyRuntime

    backend = CpuInt8Backend()
    spec = parse_quantization_spec(_training_config().qat)
    profile = resolve_quantization_profile(spec, ["projection"], backend.probe("cpu"))
    model = nn.Module()
    model.first = QuantizedWeight(backend)
    model.second = QuantizedWeight(CpuInt8Backend())
    fail = [None]

    def load(model):
        assert runtime.version is None
        with pytest.raises(RuntimeError, match="not ready"):
            runtime.assert_ready()
        for index, weight in enumerate((model.first, model.second)):
            if fail[0] == "incomplete" and index == 1:
                continue
            weight.load_weights(torch.full((2, 3), 0.5))
            assert runtime.version is None
            if fail[0] == "packing" and index == 0:
                raise RuntimeError("device packing failed")

    runtime = QuantizedPolicyRuntime(model, profile, "rollout", load)
    return model, runtime, fail


@pytest.mark.parametrize(
    "failure,error,match",
    [
        ("packing", RuntimeError, "device packing failed"),
        ("incomplete", RuntimeError, "not loaded"),
        ("recipe", RuntimeError, "block size"),
        ("version", ValueError, "nonnegative integer"),
    ],
)
def test_runtime_publishes_complete_versions_and_invalidates_on_failure(
    failure, error, match
):
    model, runtime, fail = _weight_runtime()
    with pytest.raises(RuntimeError, match="not ready"):
        runtime.assert_ready()
    runtime.load_weights(4)
    runtime.assert_ready()
    assert runtime.version == 4

    if failure in ("packing", "incomplete"):
        fail[0] = failure
    elif failure == "recipe":
        model.second.backend.block_size = 64
    with pytest.raises(error, match=match):
        runtime.load_weights(True if failure == "version" else 5)
    assert runtime.version is None
    assert not model.first.is_loaded and not model.second.is_loaded
    with pytest.raises(RuntimeError, match="not ready"):
        runtime.assert_ready()

    fail[0] = None
    model.second.backend.block_size = 128
    runtime.load_weights(5)
    runtime.assert_ready()
    assert runtime.version == 5
    runtime.invalidate()
    assert runtime.version is None
    assert not model.first.is_loaded and not model.second.is_loaded
    with pytest.raises(RuntimeError, match="not ready"):
        runtime.assert_ready()


@pytest.mark.parametrize("output_dtype", [None, torch.float32])
def test_actor_logprobs_preserve_microbatch_order_kwargs_and_dtype(output_dtype):
    from contextlib import nullcontext

    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []
            self.kwargs = []

        def forward(self, forward_inputs, **kwargs):
            values = forward_inputs["values"]
            self.batch_sizes.append(values.shape[0])
            self.kwargs.append(kwargs)
            return {"logprobs": values * kwargs["temperature"]}

    values = torch.arange(16, dtype=torch.bfloat16).reshape(8, 1, 2)
    worker = object.__new__(EmbodiedFSDPActor)
    worker.cfg = OmegaConf.create({"actor": {"micro_batch_size": 2}})
    worker.device = torch.device("cpu")
    worker.amp_context = nullcontext()
    worker.model = Policy()
    worker.rollout_batch = {
        "prev_logprobs": torch.zeros_like(values),
        "forward_inputs": {"values": values},
    }
    scores = worker.compute_actor_logprobs(
        4, output_dtype=output_dtype, temperature=0.5, top_k=7
    )

    expected_dtype = output_dtype or values.dtype
    torch.testing.assert_close(scores, (values * 0.5).to(expected_dtype))
    assert not scores.requires_grad
    assert worker.model.batch_sizes == [2] * 4
    assert all(
        kwargs
        == {
            "compute_logprobs": True,
            "compute_entropy": False,
            "compute_values": False,
            "use_cache": False,
            "temperature": 0.5,
            "top_k": 7,
        }
        for kwargs in worker.model.kwargs
    )


def test_qat_actor_freezes_plain_replay_and_preserves_rollout_scores():
    from contextlib import nullcontext

    from rlinf.quantization.layers.linear import FakeQuantLinear
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = FakeQuantLinear(3, 1, backend=CpuInt8Backend())

        def forward(self, forward_inputs, **kwargs):
            return {"logprobs": self.projection(forward_inputs["values"]).unsqueeze(1)}

    model = Policy()
    values = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    expected = model.projection(values).detach().unsqueeze(1)
    behavior = expected + 0.25
    cfg = OmegaConf.create({"actor": {"micro_batch_size": 1}})
    worker = object.__new__(EmbodiedFSDPActor)
    worker.model = model
    worker.cfg = cfg
    worker.device = torch.device("cpu")
    worker.amp_context = nullcontext()
    worker.version = 0
    worker.rollout_batch = {
        "forward_inputs": {"values": values},
        "prev_logprobs": behavior,
        "versions": torch.zeros_like(behavior, dtype=torch.long),
    }
    freeze = inspect.unwrap(EmbodiedFSDPActor.freeze_qat_logprobs)
    freeze(worker, batch_size_per_rank=2)
    anchor = worker.rollout_batch["proximal_logprobs"]
    torch.testing.assert_close(anchor, expected)
    assert not anchor.requires_grad and anchor.dtype == torch.float32
    assert worker.rollout_batch["prev_logprobs"] is behavior


@pytest.mark.parametrize("syncer_type", ["patch", "bucket"])
def test_qat_actor_sync_preserves_selected_parameters(monkeypatch, syncer_type):
    from rlinf.hybrid_engines.weight_syncer import WeightSyncer
    from rlinf.scheduler import Worker
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    model = nn.Module()
    model.frozen = nn.Parameter(torch.ones(2), requires_grad=False)
    model.action = nn.Parameter(torch.ones(2))
    syncer = WeightSyncer.create(
        OmegaConf.create(
            {
                "type": syncer_type,
                "patch": {
                    "snapshot_device": "cpu",
                    "transport_device": "cpu",
                    "compression": "none",
                    "init_sync": {"enabled": True},
                },
                "bucket": {
                    "bucket_size": 1024,
                    "bucket_device": "cpu",
                    "bucket_dtype": None,
                },
            }
        )
    )
    state_dict = model.state_dict()

    class Metadata:
        async def async_wait(self):
            return {
                "ordered_keys": list(state_dict),
                "original_shapes": {
                    key: value.shape for key, value in state_dict.items()
                },
                "receiver_dtypes": {
                    key: value.dtype for key, value in state_dict.items()
                },
            }

    worker = SimpleNamespace(
        quantization_runtime=object(),
        weight_syncer=syncer,
        enable_offload=False,
        get_rollout_state_dict=model.state_dict,
        param_names_need_sync=["action"],
        _is_weight_sender=False,
        _rollout_group_name="rollout",
        _sync_weight_comm_options=None,
        recv=lambda **kwargs: Metadata(),
        version=0,
    )
    sync = inspect.unwrap(EmbodiedFSDPActor.sync_model_to_rollout)
    asyncio.run(sync(worker))

    assert syncer.param_names_need_sync == ["action"]
    if syncer_type == "bucket":
        buckets = list(syncer.iter_buckets(state_dict, version=0))
        assert all("frozen" not in bucket for bucket in buckets)
        assert any("action" in bucket for bucket in buckets)


def test_hf_sync_blocks_sampling_during_transfer_and_publishes_only_after_pack():
    from rlinf.hybrid_engines.weight_syncer import BucketWeightSyncer
    from rlinf.hybrid_engines.weight_syncer.bucket_syncer import (
        iter_named_tensor_buckets,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    model, runtime, fail = _weight_runtime()
    model.master = nn.Parameter(torch.ones(1))
    runtime.load_weights(0)
    payloads = []
    worker = SimpleNamespace(
        quantization_runtime=runtime,
        hf_model=model,
        weight_syncer=BucketWeightSyncer(1024, None, "cpu"),
        actor_group_name="actor",
        actor_weight_src_rank=0,
        _group_name="rollout",
        _weight_sync_rollout_ranks=[0],
        _weight_sync_is_sender=False,
        _sync_weight_comm_options=None,
        version=0,
        finished_episodes=None,
        total_num_train_envs=1,
        rollout_epoch=1,
        torch_platform=SimpleNamespace(empty_cache=lambda: None),
    )

    class Transfer:
        async def async_wait(self):
            with pytest.raises(RuntimeError, match="not ready"):
                predict(worker, {})
            await asyncio.sleep(0)
            return payloads.pop(0)

    worker.broadcast = lambda *args, **kwargs: Transfer()
    sync = inspect.unwrap(MultiStepRolloutWorker.sync_model_from_actor)
    predict = inspect.unwrap(MultiStepRolloutWorker.predict)

    def apply(version, value):
        payloads.extend(
            iter_named_tensor_buckets(
                [("master", torch.full((1,), value))],
                version,
                bucket_size=1024,
                bucket_device="cpu",
            )
        )
        asyncio.run(sync(worker))

    apply(1, 2.0)
    assert worker.version == runtime.version == 1
    runtime.assert_ready()
    assert model.master.item() == 2.0
    fail[0] = "packing"
    with pytest.raises(RuntimeError, match="device packing failed"):
        apply(2, 3.0)
    assert model.master.item() == 3.0
    assert worker.version == 1
    assert runtime.version is None
    with pytest.raises(RuntimeError, match="not ready"):
        predict(worker, {})
    fail[0] = None
    apply(3, 4.0)
    assert worker.version == runtime.version == 3
    runtime.assert_ready()


@pytest.mark.parametrize("mode,expected", [("off", 1.0), ("importance_sampling", 0.9)])
def test_correction_multiplies_ppo_loss_and_gradient_and_detaches(mode, expected):
    rollout = torch.tensor([math.log(0.20)], requires_grad=True)
    frozen = torch.tensor([math.log(0.18)], requires_grad=True)
    current = torch.tensor([math.log(0.18)], requires_grad=True)
    batch = {"prev_logprobs": rollout, "proximal_logprobs": frozen}
    routed = correction_loss_kwargs(batch, parse_correction_spec({"mode": mode}))
    loss, _ = compute_decoupled_ppo_actor_loss(
        logprobs=current,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=torch.ones(1),
        **routed,
    )
    loss.backward()
    assert loss.item() == pytest.approx(-expected)
    assert current.grad.item() == pytest.approx(-expected)
    assert frozen.grad is None and rollout.grad is None
    assert batch["prev_logprobs"] is rollout


def test_correction_cap_mask_negative_advantage_and_overflow_rejection():
    spec = parse_correction_spec(
        {"mode": "importance_sampling", "max_weight": 2.0, "reject_weight_above": 10.0}
    )
    batch = {
        "prev_logprobs": torch.zeros(4),
        "proximal_logprobs": torch.tensor([math.log(4), math.log(0.5), 1000.0, 0.0]),
    }
    current = batch["proximal_logprobs"].clone().requires_grad_()
    loss, _ = compute_decoupled_ppo_actor_loss(
        logprobs=current,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=torch.tensor([1.0, -1.0, 1.0, 100.0]),
        loss_mask=torch.tensor([True, True, True, False]),
        **correction_loss_kwargs(batch, spec),
    )
    assert torch.isfinite(loss)
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([-1.0, 0.25, 0, 0]))


def test_correction_config_fails_closed():
    with pytest.raises(ValueError, match="finite and positive"):
        parse_correction_spec({"max_weight": float("nan")})


def test_masked_overflow_has_zero_gradient_and_finite_loss():
    current = torch.tensor([0.0, 1000.0], requires_grad=True)
    batch = {"prev_logprobs": torch.zeros(2), "proximal_logprobs": torch.zeros(2)}
    loss, _ = compute_decoupled_ppo_actor_loss(
        logprobs=current,
        advantages=torch.ones(2),
        loss_mask=torch.tensor([True, False]),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        **correction_loss_kwargs(batch, parse_correction_spec(None)),
    )
    assert torch.isfinite(loss)
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([-1.0, 0.0]))


@pytest.fixture
def calvin():
    pytest.importorskip("calvin_env")
    pytest.importorskip("calvin_agent.evaluation.utils")
    return importlib.import_module("rlinf.envs.sim.calvin.calvin_gym_env")


@pytest.mark.parametrize("num_workers", [None, 2])
def test_sequence_pool_preserves_vendor_sequences(calvin, monkeypatch, num_workers):
    from calvin_agent.evaluation.multistep_sequences import (
        get_sequences as vendor_get_sequences,
    )

    from rlinf.envs.sim.calvin import utils

    def bounded_pool(*, max_workers):
        assert max_workers == (4 if num_workers is None else num_workers)
        return ProcessPoolExecutor(max_workers=max_workers)

    baseline = vendor_get_sequences(32, num_workers=1)
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 768)
    monkeypatch.setattr(utils, "ProcessPoolExecutor", bounded_pool)
    utils.get_sequences.cache_clear()
    try:
        assert utils.get_sequences(32, num_workers=num_workers) == baseline
    finally:
        utils.get_sequences.cache_clear()


@pytest.mark.parametrize("scheduled_device", [None, "3"])
def test_env_factory_uses_local_render_device(calvin, monkeypatch, scheduled_device):
    import hydra

    env = calvin.CalvinEnv.__new__(calvin.CalvinEnv)
    env.cfg = SimpleNamespace(task_suite_name="calvin_d")
    env.num_envs = 2
    env.seed_offset = 8
    env.worker_info = SimpleNamespace(accelerator_rank=1)
    monkeypatch.setenv("EGL_VISIBLE_DEVICES", "0")
    if scheduled_device is None:
        monkeypatch.delenv("EGL_DEVICE_ID", raising=False)
    else:
        monkeypatch.setenv("EGL_DEVICE_ID", scheduled_device)
    monkeypatch.setattr(
        hydra.utils,
        "instantiate",
        lambda *args, **kwargs: os.environ["EGL_VISIBLE_DEVICES"],
    )

    expected_device = scheduled_device if scheduled_device is not None else "1"
    assert [fn() for fn in env.get_env_fns()] == [expected_device] * 2


@pytest.mark.parametrize("dtype,contiguous", [(np.uint8, True), (np.int64, False)])
def test_observation_preserves_pybullet_integer_rgb_and_state(
    calvin, dtype, contiguous
):
    env = calvin.CalvinEnv.__new__(calvin.CalvinEnv)
    image = np.array([[[0, 127, 255], [255, 1, 0]]], dtype=dtype)
    if not contiguous:
        image = image[:, ::-1]
    state = np.arange(15, dtype=np.float64)

    result = env._extract_image_and_state(
        {
            "rgb_obs": {"rgb_static": image, "rgb_gripper": image},
            "robot_obs": state,
        }
    )

    for key in ["full_image", "wrist_image"]:
        assert result[key].dtype == np.uint8
        assert result[key].flags.c_contiguous
        np.testing.assert_array_equal(result[key], image)
    np.testing.assert_array_equal(result["state"], state[:7])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="requires NVIDIA CUDA",
)
@pytest.mark.parametrize("quantization_format", ["int8", "nvfp4"])
def test_runtime_cuda_actor_rollout_share_manifest_and_pack_initial_version(
    quantization_format,
):
    from test_quantization import _TinyPi05

    from rlinf.quantization.layers.weights import QuantizedWeight
    from rlinf.quantization.runtime import configure_quantized_policy

    if quantization_format == "nvfp4":
        if torch.cuda.get_device_capability() not in {
            (10, 0),
            (10, 3),
            (12, 0),
            (12, 1),
        }:
            pytest.skip("requires SM100/SM120-family NVFP4 GEMM")
        flashinfer = pytest.importorskip("flashinfer")
        if flashinfer.__version__ != "0.7.1":
            pytest.skip("requires FlashInfer 0.7.1")
        cfg = _nvfp4_training_config()
    else:
        cfg = _training_config()
    actor = _TinyPi05().to(device="cuda", dtype=torch.bfloat16)
    rollout = copy.deepcopy(actor)
    masters = [dict(model.named_parameters()) for model in (actor, rollout)]
    keys = tuple(actor.state_dict())
    actor_runtime = configure_quantized_policy(actor, cfg, "actor", "cuda")
    rollout_runtime = configure_quantized_policy(rollout, cfg, "rollout", "cuda")
    assert actor_runtime.profile == rollout_runtime.profile
    assert actor_runtime.profile_hash == rollout_runtime.profile_hash
    assert actor_runtime.profile.target_manifest
    assert actor_runtime.profile.spec.weight.format == quantization_format
    for model, parameters in zip((actor, rollout), masters):
        assert tuple(model.state_dict()) == keys
        assert all(
            dict(model.named_parameters())[key] is value
            for key, value in parameters.items()
        )
        assert model.action_out_proj.backend.quantization_format == quantization_format
    rollout_runtime.assert_ready()
    weights = [
        module for module in rollout.modules() if isinstance(module, QuantizedWeight)
    ]
    assert all(weight.is_loaded for weight in weights)
    assert rollout_runtime.version == 0
    assert actor_runtime.version == rollout_runtime.version == 0
    before = rollout.action_out_proj(
        torch.ones((2, 4), device="cuda", dtype=torch.bfloat16)
    )
    with torch.no_grad():
        rollout.action_out_proj.weight.mul_(2)
    rollout_runtime.invalidate()
    with pytest.raises(RuntimeError, match="not ready"):
        rollout_runtime.assert_ready()
    rollout_runtime.load_weights(1)
    rollout_runtime.assert_ready()
    after = rollout.action_out_proj(
        torch.ones((2, 4), device="cuda", dtype=torch.bfloat16)
    )
    assert not torch.equal(before, after)
    assert all(weight.is_loaded for weight in weights)
    assert rollout_runtime.version == 1
