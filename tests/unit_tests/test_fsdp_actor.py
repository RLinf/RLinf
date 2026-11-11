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

"""FSDP actor inference, training, and checkpoint tests."""

import copy
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from rlinf.algorithms.registry import get_loss_scales
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.scheduler import Channel, Worker
from rlinf.scheduler.channel.channel_worker import LocalChannel
from rlinf.scheduler.hardware import AcceleratorType
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_seqlen_balanced_partitions,
)
from rlinf.utils.distributed import RolloutDataBalance
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.utils import get_loss_agg_func, retrieve_model_state_dict_in_cpu
from rlinf.workers.actor.ma_fsdp_actor_worker import MAFSDPActor


class _TokenOutput(NamedTuple):
    logits: torch.Tensor


class _TokenModel(torch.nn.Module):
    """Deterministic token model for actor tests."""

    _no_split_modules = ["Linear"]

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32)
        self.forward_batch_sizes = []

    def forward(self, input_ids, **kwargs):
        self.forward_batch_sizes.append(input_ids.shape[0])
        return _TokenOutput(logits=self.head(self.embedding(input_ids)))


@pytest.fixture
def actor_group(tmp_path, monkeypatch, request):
    device = getattr(request, "param", "cpu")
    monkeypatch.setattr(Worker, "torch_device_type", device)
    monkeypatch.setattr(
        Worker, "torch_platform", torch.cuda if device == "cuda" else _CPUPlatform
    )
    dist.init_process_group(
        "nccl" if device == "cuda" else "gloo",
        init_method=f"file://{tmp_path / 'group'}",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )
    try:
        yield device
    finally:
        dist.destroy_process_group()


def _rollout(second_group=False):
    if second_group:
        ids = [[12, 13, 14, 15], [12, 13, 16, 17]]
        traj, prompts, rewards, ends = [0, 1], [2, 2], [2.0, 4.0], [True, True]
    else:
        ids = [[1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 9, 10]]
        traj, prompts, rewards, ends = (
            [0, 0, 1],
            [2, 6, 2],
            [1.0, 1.0, 0.0],
            [False, True, True],
        )
    return DynamicRolloutResult(
        num_sequence=len(ids),
        group_size=2,
        idx_to_traj=traj,
        input_ids=ids,
        prompt_lengths=prompts,
        response_lengths=[len(row) - p for row, p in zip(ids, prompts)],
        is_end=ends,
        rewards=rewards,
        extra_fields_train={"idx_to_sub_traj": [0] * len(ids)},
    )


def _actor(dynamic=False, model_type="token"):
    actor = object.__new__(MAFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "data": {"rollout_batch_size": 2},
            "actor": {
                "fsdp_config": {
                    "strategy": "fsdp2",
                    "enable_gradient_accumulation": False,
                }
            },
            "algorithm": {
                "logprob_forward_micro_batch_size": 4,
                "sampling_params": {"temperature": 1.0},
                "adv_type": "grpo_dynamic",
                "advantage_mode": "trajectory",
                "group_size": 2,
                "loss_type": "actor",
                "loss_agg_func": "token-mean",
                "ratio_clip_eps": 0.2,
            },
        }
    )
    torch.manual_seed(7)
    if model_type == "qwen":
        from transformers import Qwen2Config, Qwen2ForCausalLM

        config = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        actor.model = Qwen2ForCausalLM(config).to(Worker.torch_device_type)
    else:
        actor.model = _TokenModel().to(Worker.torch_device_type)
    actor.amp_context = nullcontext()
    actor.tokenizer = SimpleNamespace(eos_token_id=0)
    actor.enable_dynamic_batch_size = dynamic
    actor.max_tokens_per_mbs = 10
    actor.entropy_op_type = "torch"
    actor.ref_policy_state_dict = None
    actor._strategy = FSDPStrategyBase.create(actor.cfg.actor, 1)
    actor._timer_metrics = {}
    actor.task_type = "reasoning"
    actor.reinpp_kl_beta = 0.0
    actor.kl_beta = 0.0
    actor.kl_penalty_type = "low_var_kl"
    actor.calculate_entropy = False
    actor.calculate_entropy_loss = False
    actor.loss_agg_func = get_loss_agg_func("token-mean")
    actor.gradient_accumulation = 1
    actor.grad_scaler = torch.amp.GradScaler("cpu", enabled=False)
    return actor


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "actor_group,reference",
    [
        ("cpu", False),
        pytest.param(
            "cuda",
            True,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="Reference weight swap requires a GPU",
            ),
        ),
    ],
    indirect=["actor_group"],
)
def test_dynamic_logprob_inference_restores_rows_and_weights(
    actor_group, dynamic, reference
):
    actor = _actor(dynamic)
    rollout = _rollout()
    batch = rollout.to_actor_batch(10, 0)
    original = copy.deepcopy(batch)
    with torch.no_grad():
        actor.enable_dynamic_batch_size = False
        expected = actor.forward_batch(batch).cpu()
        if reference:
            with torch.no_grad():
                actor.model.head.weight[0].add_(0.1)
            expected_ref = actor.forward_batch(batch).cpu()
            actor.ref_policy_state_dict = {
                k: v.cpu().clone() for k, v in actor.model.state_dict().items()
            }
            actor.model.head.weight[0].sub_(0.1)
        actor.enable_dynamic_batch_size = dynamic
        before = {k: v.clone() for k, v in actor.model.state_dict().items()}
        actual, ref = actor.inference_step(batch, rollout, reference)
        actor.enable_dynamic_batch_size = False
        after = actor.forward_batch(batch).cpu()
    mask = batch["response_mask"].cpu()
    torch.testing.assert_close(actual[mask], expected[mask])
    torch.testing.assert_close(after[mask], expected[mask])
    assert actual.shape == (3, 10)
    if reference:
        torch.testing.assert_close(ref[mask], expected_ref[mask])
    else:
        assert ref is None
    for key, value in before.items():
        torch.testing.assert_close(actor.model.state_dict()[key], value)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, original[key])
    if not dynamic:
        assert 4 in actor.model.forward_batch_sizes


def test_group_packing_preserves_mask_loss_gradient_and_update(actor_group):
    actor = _actor()
    groups = [_rollout(), _rollout(True)]
    batch = DynamicRolloutResult.merge_batches(
        [group.to_actor_batch(10, 0) for group in groups], group_size=2
    )
    assert batch["idx_to_traj"] == [0, 0, 1, 2, 3]
    assert not batch["response_mask"][1, 4:6].any()
    assert batch["attention_mask"][1, 4:6].all()
    with torch.no_grad():
        batch["recomputed_logprobs"] = actor.forward_batch(batch)
    batch = actor.compute_advantages_and_returns(batch)
    assert batch["advantages"][0, 2] > 0
    assert batch["advantages"][2, 2] < 0
    assert batch["advantages"][3, 2] < 0
    assert batch["advantages"][4, 2] > 0
    batch["loss_scales"] = batch["response_mask"].float()
    context = {
        "folding_scale": [],
        "group_scale_in_loss": True,
        "actor_global_batch_size": 4,
        "data_parallel_world_size": 1,
    }
    for scale in get_loss_scales(["group_level", "agent_level", "turn_level"]):
        batch = scale(context, batch)
    packed = DynamicRolloutResult.pack_traj_batch(context, batch)
    assert packed["input_ids"].shape[0] == 4
    assert packed["response_mask"].sum() == batch["response_mask"].sum() == 10
    assert not packed["response_mask"][0, 4:6].any()
    initial_state = copy.deepcopy(actor.model.state_dict())
    results = []
    for prepared in [batch, packed]:
        actor.model.load_state_dict(initial_state)
        optimizer = torch.optim.SGD(actor.model.parameters(), lr=0.01)
        optimizer.zero_grad()
        actor.gradient_accumulation = len(prepared["input_ids"])
        losses = []
        for idx, micro in enumerate(
            get_iterator_k_split(prepared, actor.gradient_accumulation)
        ):
            metrics = actor.forward_backward_batch(
                idx, micro, actor.gradient_accumulation
            )
            losses.append(metrics["actor/final_loss"])
        grads = [p.grad.clone() for p in actor.model.parameters()]
        assert any(grad.abs().sum() > 0 for grad in grads)
        optimizer.step()
        assert any(
            not torch.equal(actor.model.state_dict()[key], value)
            for key, value in initial_state.items()
        )
        results.append(
            (
                torch.stack(losses).mean(),
                grads,
                copy.deepcopy(actor.model.state_dict()),
            )
        )
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-6, rtol=1e-5)
    for left, right in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    for key in initial_state:
        torch.testing.assert_close(
            results[0][2][key], results[1][2][key], atol=1e-6, rtol=1e-5
        )


@pytest.mark.parametrize("batch_size", [0, 1, 3])
def test_placement_rejects_incomplete_rollout_groups(batch_size):
    from test_placement import create_fake_cluster

    config_dir = Path(__file__).resolve().parents[2] / "examples/agent/searchr1/config"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name="train_qwen2.5_fsdp_disaggregated")
    cfg.data.rollout_batch_size = batch_size
    placement = ModelParallelComponentPlacement(cfg, create_fake_cluster(1, 3))
    with pytest.raises(ValueError, match="positive multiple"):
        MAFSDPActor.validate_placement(cfg, placement)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NCCL requires a GPU")
@pytest.mark.parametrize("actor_group", ["cuda"], indirect=True)
def test_dynamic_training_splits_cpu_batches_with_nccl(actor_group):
    actor = _actor(dynamic=True)
    batch = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v
        for k, v in _rollout().to_actor_batch(10, 0).items()
    }
    iterator, count, indices = actor._split_to_micro_batch(
        batch, True, max_tokens_per_mbs=10, split_num=1
    )
    micro_batches = list(iterator)
    assert count == len(micro_batches) == 2
    assert sorted(sum(indices, [])) == [0, 1, 2]
    assert all(micro["input_ids"].device.type == "cpu" for micro in micro_batches)


def _check_two_rank_dynamic_actor(rank, init_file):
    Worker.torch_device_type = "cpu"
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        actor = _actor()
        assert actor._get_num_rollout_results_per_dp() == 1
        rollout = _rollout(second_group=bool(rank))
        batch = rollout.to_actor_batch(10, 0)
        with torch.no_grad():
            expected = actor.forward_batch(batch)
            for dynamic in (False, True):
                actor.enable_dynamic_batch_size = dynamic
                actor.model.forward_batch_sizes.clear()
                actual, ref = actor.inference_step(batch, rollout, False)
                torch.testing.assert_close(
                    actual[batch["response_mask"]], expected[batch["response_mask"]]
                )
                assert actual.shape[0] == rollout.num_sequence
                assert ref is None
                counts = [None, None]
                dist.all_gather_object(counts, len(actor.model.forward_batch_sizes))
                assert counts[0] == counts[1]
        batch["recomputed_logprobs"] = actual
        batch.pop("idx_to_traj")
        batch.pop("extra:idx_to_sub_traj")
        original = [None, None]
        dist.all_gather_object(original, batch)
        balanced = RolloutDataBalance.from_rollout_batches_dynamic(
            batch,
            2,
            rank,
            dist.group.WORLD,
            DynamicRolloutResult.get_batch_pad(10, list(batch)),
            split_fix_chunk=2,
            partitioning_tool=get_seqlen_balanced_partitions,
        )
        assert balanced["input_ids"].shape[0] == 4
        rebalanced = [None, None]
        dist.all_gather_object(rebalanced, balanced)
        for key in batch:
            before = torch.cat([part[key] for part in original])
            after = torch.cat(
                [part[key][part["response_mask"].any(-1)] for part in rebalanced]
            )
            # Match by the complete token row, since turns can share a prompt.
            before_ids = torch.cat([part["input_ids"] for part in original]).tolist()
            after_ids = torch.cat(
                [
                    part["input_ids"][part["response_mask"].any(-1)]
                    for part in rebalanced
                ]
            ).tolist()
            order = [after_ids.index(row) for row in before_ids]
            torch.testing.assert_close(before, after[order])
        assert sum(part["response_mask"].sum().item() for part in rebalanced) == 10
    finally:
        dist.destroy_process_group()


def test_two_rank_dynamic_inference_and_batch_distribution(tmp_path):
    torch.multiprocessing.spawn(
        _check_two_rank_dynamic_actor,
        args=(str(Path(tmp_path) / "two-rank"),),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize(
    "invalid", ["fsdp1", "no_recompute", "dedicated_inference", "uneven_batch"]
)
def test_disaggregated_actor_rejects_unsupported_modes_before_worker_init(invalid):
    from test_placement import create_fake_cluster

    config_dir = Path(__file__).resolve().parents[2] / "examples/agent/searchr1/config"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name="train_qwen2.5_fsdp_disaggregated")
    if invalid == "fsdp1":
        cfg.actor.fsdp_config.strategy = "fsdp"
        message = "requires FSDP2"
    elif invalid == "no_recompute":
        cfg.algorithm.recompute_logprobs = False
        message = "requires algorithm.recompute_logprobs=True"
    elif invalid == "uneven_batch":
        cfg.data.rollout_batch_size = 3
        message = "positive multiple"
    else:
        with open_dict(cfg.cluster.component_placement):
            cfg.cluster.component_placement.inference = 3
        message = "remove inference"
    placement = ModelParallelComponentPlacement(cfg, create_fake_cluster(1, 4))
    with pytest.raises(ValueError, match=message):
        MAFSDPActor(cfg, placement)


@pytest.mark.parametrize("all_padding", [False, True])
def test_padding_microbatch_keeps_valid_rows_and_backward_graph(
    actor_group, all_padding
):
    actor = _actor()
    batch = _rollout(True).to_actor_batch(10, 0)
    with torch.no_grad():
        batch["recomputed_logprobs"] = actor.forward_batch(batch)
    batch["advantages"] = torch.ones_like(batch["recomputed_logprobs"])
    batch["loss_scales"] = batch["response_mask"].float()
    batch["response_mask"][0] = False
    if all_padding:
        batch["response_mask"][1] = False
    metrics = actor.forward_backward_batch(0, batch, 1)
    grads = [p.grad for p in actor.model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    if all_padding:
        assert metrics["actor/final_loss"] == 0
        assert all(torch.count_nonzero(g) == 0 for g in grads)
    else:
        assert metrics["actor/final_loss"] < 0
        assert any(torch.count_nonzero(g) > 0 for g in grads)


class _CPUPlatform:
    """CPU implementations of the accelerator operations used by the actor."""

    @staticmethod
    def current_device():
        return "cpu"

    @staticmethod
    def is_available():
        return False

    @staticmethod
    def synchronize():
        pass

    ipc_collect = synchronize
    empty_cache = synchronize


def _local_channel(name):
    channel = Channel()
    channel._initialize(name, None, None, None, local_channel=LocalChannel())
    return channel


def _sharded_actor(dynamic, model_type="token", world_size=2):
    from torch.distributed.device_mesh import init_device_mesh

    actor = _actor(dynamic, model_type)
    actor.cfg = OmegaConf.merge(
        actor.cfg,
        {
            "data": {"max_prompt_length": 2},
            "actor": {
                "seed": 7,
                "micro_batch_size": 1,
                "model": {"encoder_seq_length": 10},
                "optim": {"clip_grad": 1.0},
                "fsdp_config": {
                    "enable_gradient_accumulation": True,
                    "mixed_precision": {"param_dtype": "fp32", "reduce_dtype": "fp32"},
                    "cpu_offload": False,
                    "reshard_after_forward": True,
                    "save_full_model_weights": True,
                },
            },
            "algorithm": {
                "n_minibatches": 1,
                "normalize_advantages": False,
                "shuffle_rollout": False,
            },
        },
    )
    actor._cfg = actor.cfg.actor
    actor._strategy = FSDPStrategyBase.create(
        actor.cfg.actor, world_size, dist.group.WORLD
    )
    actor.model = actor._strategy.wrap_model(
        actor.model, init_device_mesh(Worker.torch_device_type, (world_size,))
    )
    actor.optimizer = torch.optim.SGD(actor.model.parameters(), lr=0.01, momentum=0.9)
    actor.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        actor.optimizer, step_size=1, gamma=0.9
    )
    actor.optimizer_steps = 0
    actor.critic_warmup_steps = 0
    actor.lr_sched_sync_with_optim = True
    actor.loss_scale_fns = get_loss_scales(["group_level", "agent_level", "turn_level"])
    actor.pack_traj = True
    actor.enable_dp_load_balance = True
    actor.enable_offload = False
    actor.is_weight_offloaded = False
    actor.is_optimizer_offloaded = False
    actor.is_pipeline = False
    actor.micro_batch_size = 1
    actor.response_len = 8
    actor.device = Worker.torch_platform.current_device()
    actor._device_lock = nullcontext()
    actor._accelerator_type = AcceleratorType.NO_ACCEL
    return actor


@pytest.mark.parametrize("dynamic", [False, True])
def test_generated_padding_keeps_backward_graph(actor_group, dynamic):
    actor = _actor(dynamic)
    batch = DynamicRolloutResult.get_batch_pad(
        10,
        [
            "input_ids",
            "attention_mask",
            "position_ids",
            "prompt_lengths",
            "response_lengths",
            "response_mask",
            "recomputed_logprobs",
            "advantages",
            "loss_scales",
        ],
    )
    metrics = actor.forward_backward_batch(0, batch, 1)
    assert metrics["actor/final_loss"] == 0
    for parameter in actor.model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("dynamic", [False, True])
def test_sharded_training_with_generated_padding(actor_group, dynamic):
    from packaging.version import Version

    if Version(torch.__version__) < Version("2.12"):
        pytest.skip("CPU FSDP2 coverage currently targets PyTorch >= 2.12")
    actor = _sharded_actor(dynamic, world_size=1)
    actor.pack_traj = False
    actor.cfg.algorithm.n_minibatches = 4
    actor.n_mini_batches = 4
    rollout_metrics, metrics = _run_sharded_step(actor, 0)
    assert rollout_metrics["valid_response_tokens"] == 10
    assert len(metrics) == actor.optimizer_steps == 3
    assert actor.optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * 0.9**3)
    for parameter in actor.model.parameters():
        assert torch.isfinite(parameter.full_tensor()).all()
        assert torch.isfinite(parameter.grad.full_tensor()).all()


def _unsharded_update(temperature=1.0, model_type="token", kl_beta=0.0):
    actor = _actor(model_type=model_type)
    actor.kl_beta = kl_beta
    actor.cfg.algorithm.sampling_params.temperature = temperature
    batch = DynamicRolloutResult.merge_batches(
        [_rollout().to_actor_batch(10, 0), _rollout(True).to_actor_batch(10, 0)],
        group_size=2,
    )
    with torch.no_grad():
        batch["recomputed_logprobs"] = actor.forward_batch(batch)
    if kl_beta > 0:
        batch["ref_logprobs"] = batch["recomputed_logprobs"] + 0.15
    batch = actor.compute_advantages_and_returns(batch)
    mask = batch["response_mask"]
    trajectories = set(batch["idx_to_traj"])
    weights = torch.zeros_like(batch["advantages"])
    for traj in trajectories:
        rows = [i for i, t in enumerate(batch["idx_to_traj"]) if t == traj]
        weights[rows] = 1 / (len(trajectories) * mask[rows].sum())
    optimizer = torch.optim.SGD(actor.model.parameters(), lr=0.01, momentum=0.9)
    logprobs = actor.forward_batch(batch)
    ratio = torch.exp(logprobs - batch["recomputed_logprobs"])
    loss = -(ratio * batch["advantages"] * weights * mask).sum()
    if kl_beta > 0:
        log_ratio = batch["ref_logprobs"] - logprobs
        kld = log_ratio.exp() - log_ratio - 1
        loss = loss + kl_beta * (kld * weights * mask).sum()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(actor.model.parameters(), 1.0)
    grads = {name: p.grad.clone() for name, p in actor.model.named_parameters()}
    optimizer.step()
    return actor.model.state_dict(), grads, norm, loss.detach()


def _run_sharded_step(actor, rank):
    inputs, inference = _local_channel("input"), _local_channel("inference")
    for group_index in range(rank, 2, dist.get_world_size()):
        inputs.put(_rollout(bool(group_index)))
    actor.run_inference(inputs, inference, compute_ref_logprobs=False)
    if actor.kl_beta > 0:
        groups = [
            inference.get() for _ in range(actor._get_num_rollout_results_per_dp())
        ]
        for rollout in groups:
            rollout.ref_logprobs = rollout.recomputed_logprobs + 0.15
            inference.put(rollout)
    metrics = actor.run_training(inference)
    assert inputs.empty() and inference.empty()
    return metrics


def _check_sharded_training(
    rank,
    init_file,
    checkpoint_dir,
    dynamic,
    temperature,
    device="cpu",
    model_type="token",
    world_size=2,
    kl_beta=0.0,
    pack_traj=True,
    micro_batch_size=1,
):
    Worker.torch_device_type = device
    Worker.torch_platform = torch.cuda if device == "cuda" else _CPUPlatform
    if device == "cuda":
        torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "nccl" if device == "cuda" else "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )
    try:
        expected, grads, norm, loss = _unsharded_update(
            temperature, model_type, kl_beta
        )
        actor = _sharded_actor(dynamic, model_type, world_size)
        actor.kl_beta = kl_beta
        actor.pack_traj = pack_traj
        actor.micro_batch_size = micro_batch_size
        actor.cfg.actor.micro_batch_size = micro_batch_size
        actor.cfg.algorithm.sampling_params.temperature = temperature
        assert all(
            isinstance(p, torch.distributed.tensor.DTensor)
            for p in actor.model.parameters()
        )
        if world_size > 1:
            assert any(
                p.to_local().numel() < p.numel() for p in actor.model.parameters()
            )
        rollout_metrics, training_metrics = _run_sharded_step(actor, rank)
        assert rollout_metrics["total_num_sequence"] == 5
        assert rollout_metrics["valid_response_tokens"] == 10
        assert len(training_metrics) == 1
        torch.testing.assert_close(
            torch.tensor(training_metrics[0]["actor/grad_norm"]), norm.cpu()
        )
        torch.testing.assert_close(
            torch.tensor(training_metrics[0]["actor/final_loss"]), loss.cpu()
        )
        weight_error, gradient_error = 0.0, 0.0
        for name, p in actor.model.named_parameters():
            weight = p.full_tensor()
            gradient = p.grad.full_tensor()
            torch.testing.assert_close(weight, expected[name], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(gradient, grads[name], atol=1e-6, rtol=1e-5)
            weight_error = max(
                weight_error, (weight - expected[name]).abs().max().item()
            )
            gradient_error = max(
                gradient_error, (gradient - grads[name]).abs().max().item()
            )
        actor.save_checkpoint(checkpoint_dir)
        _run_sharded_step(actor, rank)
        uninterrupted = {
            k: v.full_tensor().clone() for k, v in actor.model.state_dict().items()
        }
        expected_lr = actor.lr_scheduler.state_dict()
        actor.load_checkpoint(checkpoint_dir)
        _run_sharded_step(actor, rank)
        restore_error = 0.0
        for name, p in actor.model.named_parameters():
            restored = p.full_tensor()
            torch.testing.assert_close(
                restored, uninterrupted[name], atol=1e-6, rtol=1e-5
            )
            restore_error = max(
                restore_error, (restored - uninterrupted[name]).abs().max().item()
            )
        assert actor.lr_scheduler.state_dict() == expected_lr
        if rank == 0:
            print(
                json.dumps(
                    {
                        "model": model_type,
                        "device": device,
                        "world_size": world_size,
                        "dynamic_batch": dynamic,
                        "temperature": temperature,
                        "kl_beta": kl_beta,
                        "pack_traj": pack_traj,
                        "micro_batch_size": micro_batch_size,
                        "loss": float(loss.detach()),
                        "grad_norm": float(norm),
                        "weight_max_abs_error": weight_error,
                        "gradient_max_abs_error": gradient_error,
                        "resume_max_abs_error": restore_error,
                        "resumed_lr": actor.optimizer.param_groups[0]["lr"],
                    }
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_sharded_actor_matches_unsharded_update_and_checkpoint_resume(
    tmp_path, dynamic, temperature
):
    from packaging.version import Version

    if Version(torch.__version__) < Version("2.12"):
        pytest.skip("CPU FSDP2 coverage currently targets PyTorch >= 2.12")
    torch.multiprocessing.spawn(
        _check_sharded_training,
        args=(
            str(tmp_path / "group"),
            str(tmp_path / "checkpoint"),
            dynamic,
            temperature,
        ),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize(
    "placement_mode", ["collocated", "disaggregated", "unsupported"]
)
def test_fsdp_entrypoint_import_and_validation_do_not_load_megatron(placement_mode):
    repo = Path(__file__).resolve().parents[2]
    script = """
import sys
import runpy
from pathlib import Path
from hydra import compose, initialize_config_dir
from rlinf.utils.placement import ModelParallelComponentPlacement
from test_placement import create_fake_cluster
from rlinf import config

get_ma_actor_worker = runpy.run_path('examples/agent/searchr1/train.py')['get_ma_actor_worker']
with initialize_config_dir(version_base='1.1', config_dir=str(Path.cwd() / 'examples/agent/searchr1/config')):
    cfg = compose(config_name='train_qwen2.5_fsdp_disaggregated')
mode = sys.argv[1]
if mode == 'collocated':
    cfg.cluster.component_placement.actor = '0-1'
    cfg.cluster.component_placement.rollout = '0-1'
if mode == 'unsupported':
    cfg.actor.fsdp_config.strategy = 'fsdp'
config.Cluster = lambda: create_fake_cluster(1, 3)
if mode == 'unsupported':
    try:
        config.validate_searchr1_cfg(cfg)
    except ValueError as exc:
        assert 'requires FSDP2' in str(exc), str(exc)
    else:
        raise AssertionError('Disaggregated FSDP1 passed config validation')
else:
    config.validate_searchr1_cfg(cfg)
actor = get_ma_actor_worker(cfg)
assert actor.__name__ == 'MAFSDPActor'
if mode != 'unsupported':
    actor.validate_placement(cfg, ModelParallelComponentPlacement(cfg, create_fake_cluster(1, 3)))
assert not any(name == 'megatron' or name.startswith('megatron.') for name in sys.modules)
print('FSDP entrypoint validated without Megatron')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, placement_mode],
        cwd=repo,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(repo), str(repo / "tests/unit_tests")]),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "validated without Megatron" in result.stdout


def test_tiny_qwen_sharded_update_and_checkpoint_resume(tmp_path):
    from packaging.version import Version

    if Version(torch.__version__) < Version("2.12"):
        pytest.skip("CPU FSDP2 coverage currently targets PyTorch >= 2.12")
    torch.multiprocessing.spawn(
        _check_sharded_training,
        args=(
            str(tmp_path / "group"),
            str(tmp_path / "checkpoint"),
            False,
            1.0,
            "cpu",
            "qwen",
        ),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize(
    "world_size,dynamic", [(1, False), (1, True), (2, False), (2, True)]
)
def test_nonzero_kl_sharded_update_and_checkpoint_resume(tmp_path, world_size, dynamic):
    from packaging.version import Version

    if Version(torch.__version__) < Version("2.12"):
        pytest.skip("CPU FSDP2 coverage currently targets PyTorch >= 2.12")
    args = (
        str(tmp_path / "group"),
        str(tmp_path / "checkpoint"),
        dynamic,
        0.7,
        "cpu",
        "token",
        world_size,
        0.1,
    )
    if world_size == 1:
        _check_sharded_training(0, *args)
    else:
        torch.multiprocessing.spawn(
            _check_sharded_training, args=args, nprocs=world_size, join=True
        )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("pack_traj", [False, True])
def test_padded_microbatches_match_trajectory_objective(tmp_path, dynamic, pack_traj):
    from packaging.version import Version

    if Version(torch.__version__) < Version("2.12"):
        pytest.skip("CPU FSDP2 coverage currently targets PyTorch >= 2.12")
    _check_sharded_training(
        0,
        str(tmp_path / "group"),
        str(tmp_path / "checkpoint"),
        dynamic,
        0.7,
        world_size=1,
        kl_beta=0.1,
        pack_traj=pack_traj,
        micro_batch_size=2,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("kl_beta", [0.0, 0.1])
def test_cuda_sharded_actor_update_and_checkpoint_resume(tmp_path, dynamic, kl_beta):
    torch.multiprocessing.spawn(
        _check_sharded_training,
        args=(
            str(tmp_path / "group"),
            str(tmp_path / "checkpoint"),
            dynamic,
            0.7,
            "cuda",
            "token" if dynamic else "qwen",
            2,
            kl_beta,
        ),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires one CUDA GPU")
def test_single_cuda_fsdp_actor_update_and_checkpoint_resume(tmp_path):
    torch.multiprocessing.spawn(
        _check_sharded_training,
        args=(
            str(tmp_path / "group"),
            str(tmp_path / "checkpoint"),
            False,
            0.7,
            "cuda",
            "qwen",
            1,
        ),
        nprocs=1,
        join=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires one CUDA GPU")
@pytest.mark.parametrize("actor_group", ["cuda"], indirect=True)
def test_single_cuda_fsdp_reference_swap_preserves_policy(actor_group):
    _check_fsdp_reference_swap()


def _check_fsdp_reference_swap():
    actor = _sharded_actor(False, "qwen", world_size=dist.get_world_size())
    reference = _actor(model_type="qwen")
    reference_initial = {
        name: parameter.detach().clone()
        for name, parameter in reference.model.named_parameters()
    }
    rollout = _rollout()
    batch = rollout.to_actor_batch(10, 0)
    actor.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(actor, is_fsdp=True)
    assert all(
        isinstance(value, torch.distributed.tensor.DTensor)
        and value.device.type == "cpu"
        for value in actor.ref_policy_state_dict.values()
    )
    with torch.no_grad():
        initial = actor.get_model_state_dict(cpu_offload=True, full_state_dict=True)
        expected = reference.forward_batch(batch).cpu()
        reference.model.lm_head.weight[0].add_(0.25)
        expected_ref = reference.forward_batch(batch).cpu()
        if dist.get_rank() == 0:
            actor.ref_policy_state_dict["lm_head.weight"].to_local()[0].add_(0.25)
        actual, ref = actor.inference_step(batch, rollout, True)
        after = actor.get_model_state_dict(cpu_offload=True, full_state_dict=True)
    mask = batch["response_mask"].cpu()
    torch.testing.assert_close(actual[mask], expected[mask])
    torch.testing.assert_close(ref[mask], expected_ref[mask])
    assert (actual[mask] - ref[mask]).abs().max() > 1e-5
    for name in initial:
        torch.testing.assert_close(after[name], initial[name], atol=0, rtol=0)
    with torch.no_grad():
        for name, parameter in actor.model.named_parameters():
            torch.testing.assert_close(
                parameter.full_tensor(),
                reference_initial[name],
                atol=0,
                rtol=0,
            )


def _check_cuda_sharded_reference_swap(rank, init_file):
    Worker.torch_device_type = "cuda"
    Worker.torch_platform = torch.cuda
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        _check_fsdp_reference_swap()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
def test_cuda_sharded_fsdp_reference_swap_preserves_policy(tmp_path):
    torch.multiprocessing.spawn(
        _check_cuda_sharded_reference_swap,
        args=(str(tmp_path / "group"),),
        nprocs=2,
        join=True,
    )
