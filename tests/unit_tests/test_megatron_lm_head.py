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


"""Compare chunked vocabulary projection against dense values and gradients."""

import pytest
import torch


@pytest.mark.parametrize(
    "chunk_size,temperature,dtype,entropy",
    [
        (1, 0.7, torch.float32, False),
        (4, 1.3, torch.float32, True),
        (3, 1.0, torch.bfloat16, False),
        (32, 1.0, torch.float16, False),
    ],
)
def test_chunked_lm_head_matches_dense_values_and_gradients(
    chunk_size, temperature, dtype, entropy
):
    from rlinf.hybrid_engines.megatron.lm_head import chunked_vocab_parallel_log_probs

    torch.manual_seed(81)
    hidden = torch.randn(2, 7, 5, dtype=dtype, requires_grad=True)
    weight = torch.randn(19, 5, dtype=dtype, requires_grad=True)
    bias = torch.randn(19, dtype=dtype, requires_grad=True)
    labels = torch.randint(19, (2, 7))
    mask = torch.rand(2, 7) > 0.3
    scale = torch.randn(2, 7)
    dense = torch.nn.functional.linear(hidden, weight, bias).float() / temperature
    dense_log_probs = dense.log_softmax(-1)
    expected = (
        dense_log_probs.gather(-1, labels.unsqueeze(-1))
        .squeeze(-1)
        .masked_fill(~mask, 0)
    )
    reference_grads = torch.autograd.grad(
        (expected * scale).sum(), (hidden, weight, bias)
    )
    actual = chunked_vocab_parallel_log_probs(
        hidden,
        weight,
        labels,
        mask,
        chunk_size=chunk_size,
        temperature=temperature,
        bias=bias,
        calculate_entropy=entropy,
    )
    torch.testing.assert_close(actual["log_probs"], expected, atol=2e-6, rtol=2e-6)
    actual_grads = torch.autograd.grad(
        (actual["log_probs"] * scale).sum(), (hidden, weight, bias)
    )
    tolerance = 0.02 if dtype in (torch.bfloat16, torch.float16) else 2e-6
    for got, want in zip(actual_grads, reference_grads):
        torch.testing.assert_close(got, want, atol=tolerance, rtol=tolerance)
    if entropy:
        assert not actual["entropy"].requires_grad
        torch.testing.assert_close(
            actual["entropy"], -(dense_log_probs.exp() * dense_log_probs).sum(-1)
        )


def test_chunked_lm_head_accumulates_main_grad_without_double_add():
    from rlinf.hybrid_engines.megatron.lm_head import chunked_vocab_parallel_log_probs

    torch.manual_seed(93)
    weight = torch.nn.Parameter(torch.randn(17, 5))
    weight.main_grad = torch.ones_like(weight)
    weight.grad_added_to_main_grad = False
    expected = weight.main_grad.clone()
    for _ in range(2):
        hidden = torch.randn(7, 5, requires_grad=True)
        labels = torch.randint(17, (7,))
        dense = torch.nn.functional.linear(hidden, weight).log_softmax(-1)
        expected += torch.autograd.grad(dense.gather(1, labels[:, None]).sum(), weight)[
            0
        ]
        result = chunked_vocab_parallel_log_probs(
            hidden,
            weight,
            labels,
            torch.ones(7, dtype=torch.bool),
            chunk_size=3,
            gradient_accumulation_fusion=True,
        )
        result["log_probs"].sum().backward()
    torch.testing.assert_close(weight.main_grad, expected, atol=3e-6, rtol=3e-6)
    assert weight.grad_added_to_main_grad
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight))


def _chunked_lm_head_tp_reference(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist

    from rlinf.hybrid_engines.megatron.lm_head import chunked_vocab_parallel_log_probs

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(17)
        hidden = torch.randn(9, 5, requires_grad=True)
        full_weight = torch.randn(22, 5, requires_grad=True)
        labels = torch.tensor([0, 10, 11, 21, 3, 16, 7, 18, 20])
        mask = torch.tensor([True, False, True, True, True, True, False, True, True])
        scales = torch.randn(9)
        reference = (hidden @ full_weight.t() / 0.7).log_softmax(-1)
        reference = (
            reference.gather(1, labels[:, None]).squeeze(1).masked_fill(~mask, 0)
        )
        dh, dw = torch.autograd.grad((reference * scales).sum(), (hidden, full_weight))
        shard = full_weight.detach().chunk(2)[rank].clone().requires_grad_()
        actual = chunked_vocab_parallel_log_probs(
            hidden,
            shard,
            labels,
            mask,
            chunk_size=3,
            temperature=0.7,
            tp_group=dist.group.WORLD,
        )
        (actual["log_probs"] * scales).sum().backward()
        dist.all_reduce(hidden.grad)
        torch.testing.assert_close(actual["log_probs"], reference, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(hidden.grad, dh, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(shard.grad, dw.chunk(2)[rank], atol=3e-6, rtol=3e-6)
    finally:
        dist.destroy_process_group()


def test_chunked_lm_head_two_rank_vocab_shards(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_gloo_available():
        pytest.skip("The TP reference test requires Gloo")
    mp.spawn(
        _chunked_lm_head_tp_reference, args=((tmp_path / "tp-init").as_uri(),), nprocs=2
    )


def test_pipeline_output_projection(tmp_path):
    pytest.importorskip("megatron.core")
    torch.multiprocessing.spawn(
        _pipeline_output_projection, args=((tmp_path / "pg").as_uri(),), nprocs=2
    )


def _pipeline_output_projection(rank, rendezvous):
    import torch.distributed as dist
    from megatron.core import ModelParallelConfig, parallel_state
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    from rlinf.hybrid_engines.megatron.output_projection import (
        chunked_scoring,
        install_chunked_projection,
    )
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
    from rlinf.workers.actor.megatron_actor_worker import MegatronActor

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        groups = [dist.new_group([i], backend="gloo") for i in range(2)]
        with pytest.MonkeyPatch.context() as mp:
            for name in ("_TENSOR_MODEL_PARALLEL_GROUP", "_DATA_PARALLEL_GROUP"):
                mp.setattr(parallel_state, name, groups[rank])
            model = torch.nn.Module()
            model.post_process = rank == 1
            if rank == 0:
                install_chunked_projection(model)
                hidden = torch.randn(5, 1, 4)
                dist.send(hidden, 1)
                grad = torch.empty_like(hidden)
                dist.recv(grad, 1)
                assert torch.isfinite(grad).all() and grad.abs().sum() > 0
                return
            model.output_layer = ColumnParallelLinear(
                4,
                8,
                config=ModelParallelConfig(use_cpu_initialization=True),
                init_method=torch.nn.init.xavier_uniform_,
                bias=False,
            )
            hidden = torch.empty(5, 1, 4)
            dist.recv(hidden, 0)
            hidden.requires_grad_()
            labels = torch.arange(5).unsqueeze(0)
            dense, _ = model.output_layer(hidden)
            keys = list(model.state_dict())
            params = {k: id(v) for k, v in model.named_parameters()}
            layer = model.output_layer
            hooks = (dict(layer._forward_pre_hooks), dict(layer._forward_hooks))
            actor = MAMegatronActor.__new__(MAMegatronActor)
            mp.setattr(MegatronActor, "model_provider_func", lambda *args: model)
            for size in (0, 2):
                actor.lm_head_chunk_size = size
                assert actor.model_provider_func(False, True) is model
                assert (type(layer) is ColumnParallelLinear) == (size == 0)
                assert hooks == (layer._forward_pre_hooks, layer._forward_hooks)
                assert list(model.state_dict()) == keys
                assert {k: id(v) for k, v in model.named_parameters()} == params

            with chunked_scoring(
                model,
                labels=labels,
                label_mask=torch.ones_like(labels, dtype=torch.bool),
                chunk_size=2,
            ):
                scores, _ = model.output_layer(hidden)
            expected = (
                dense.transpose(0, 1).log_softmax(-1).gather(-1, labels[..., None])
            )
            torch.testing.assert_close(scores.transpose(0, 1), expected)
            grads = torch.autograd.grad(
                scores.sum(), (hidden, model.output_layer.weight)
            )
            reference = torch.autograd.grad(
                expected.sum(), (hidden, model.output_layer.weight)
            )
            torch.testing.assert_close(grads, reference)
            dist.send(grads[0].contiguous(), 0)
            assert torch.equal(model.output_layer(hidden)[0], dense)
    finally:
        dist.destroy_process_group()
