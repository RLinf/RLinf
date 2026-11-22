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


@pytest.mark.parametrize("instance_override", [False, True])
def test_chunked_forward_restores_postprocess_after_failure(instance_override):
    from types import SimpleNamespace

    from rlinf.hybrid_engines.megatron.qwen_vl import (
        forward_qwen_vl_with_chunked_log_probs,
    )

    class LanguageModel(torch.nn.Module):
        def _postprocess(self, **kwargs):
            return kwargs

    language = LanguageModel()
    language.post_process = True
    language.mtp_process = False
    language.config = SimpleNamespace(
        mtp_num_layers=0, use_mup=False, defer_embedding_wgrad_compute=False, fp8=False
    )
    language.pg_collection = SimpleNamespace(tp=None)
    if instance_override:
        language._postprocess = lambda **kwargs: kwargs
    original = language._postprocess
    model = SimpleNamespace(language_model=language, post_process=True)

    def fail(**kwargs):
        assert language._postprocess is not original
        raise RuntimeError("decoder failed")

    with pytest.raises(RuntimeError, match="decoder failed"):
        forward_qwen_vl_with_chunked_log_probs(
            fail,
            model,
            label=torch.zeros(1, 2, dtype=torch.long),
            label_mask=torch.ones(1, 2, dtype=torch.bool),
            chunk_size=1,
            temperature=1.0,
        )
    assert ("_postprocess" in language.__dict__) is instance_override
    assert language._postprocess == original


@pytest.mark.parametrize("load_mtp", [None, True, False])
def test_chunked_qwen_scoring_requires_explicit_mtp_disable(load_mtp):
    from omegaconf import OmegaConf

    from rlinf.hybrid_engines.megatron.qwen_vl import validate_qwen_vl_chunked_cfg

    model = {
        "model_type": "qwen3_5",
        "lm_head_chunk_size": 256,
        "pipeline_model_parallel_size": 1,
    }
    if load_mtp is not None:
        model["load_mtp"] = load_mtp
    cfg = OmegaConf.create(
        {
            "actor": {
                "training_backend": "megatron",
                "model": model,
                "megatron": {"mbridge": True},
            },
            "agentloop": {"is_dynamic_rollout_batch": True},
            "algorithm": {
                "calculate_entropy": False,
                "entropy_bonus": 0,
                "sampling_params": {"temperature": 1.0},
            },
        }
    )
    if load_mtp is False:
        validate_qwen_vl_chunked_cfg(cfg)
        cfg.algorithm.entropy_bonus = 0.1
        with pytest.raises(ValueError, match="entropy_bonus must be zero"):
            validate_qwen_vl_chunked_cfg(cfg)
    else:
        with pytest.raises(ValueError, match="load_mtp=False"):
            validate_qwen_vl_chunked_cfg(cfg)


def test_qwen_chunked_adapter_bypasses_dense_projection(monkeypatch):
    from types import SimpleNamespace

    from rlinf.hybrid_engines.megatron.qwen_vl import (
        forward_qwen_vl_with_chunked_log_probs,
    )

    torch.manual_seed(12)
    hidden = torch.randn(7, 1, 4, requires_grad=True)
    head = torch.nn.Linear(4, 11)
    head.gradient_accumulation_fusion = False
    labels = torch.tensor([[2, 4, 1, 3, 5, 2, 8]])
    mask = labels.remainder(2).bool()
    temperature = 0.7
    logits = head(hidden).transpose(0, 1) / temperature
    expected = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    expected = expected.masked_fill(~mask, 0)
    parameters = (hidden, head.weight, head.bias)
    gradients = torch.autograd.grad(expected.sum(), parameters)

    # This is the Bridge/Core output-processor boundary. A regular head call
    # after the decoder would materialize the full vocabulary tensor.
    def postprocess(*, output_processor):
        return output_processor(
            hidden_states=hidden,
            output_layer=head,
            output_weight=None,
            labels=None,
            inference_context=None,
        )

    language = SimpleNamespace(
        post_process=True,
        mtp_process=False,
        pg_collection=SimpleNamespace(tp=None),
        config=SimpleNamespace(
            mtp_num_layers=0,
            use_mup=False,
            defer_embedding_wgrad_compute=False,
            fp8=False,
        ),
        _postprocess=postprocess,
    )
    monkeypatch.setattr(head, "forward", lambda *_: pytest.fail("dense projection"))
    result = forward_qwen_vl_with_chunked_log_probs(
        lambda: language._postprocess(),
        SimpleNamespace(post_process=True, language_model=language),
        label=labels,
        label_mask=mask,
        chunk_size=2,
        temperature=temperature,
    )["log_probs"]
    torch.testing.assert_close(result, expected)
    for actual, reference in zip(
        torch.autograd.grad(result.sum(), parameters), gradients
    ):
        torch.testing.assert_close(actual, reference)
