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

"""Dense and chunked response scoring: values, gradients, and saved tensors."""

import pytest
import torch


def test_response_only_logits_match_full_logits_at_scored_positions():
    pytest.importorskip("megatron.training")
    from rlinf.hybrid_engines.megatron.utils import output_layer_positions
    from rlinf.workers.actor.ma_megatron_actor_worker import _at_positions

    torch.manual_seed(0)
    module = torch.nn.Module()
    module.output_layer = torch.nn.Linear(4, 7)
    hidden = torch.randn(6, 1, 4)  # [tokens, batch, hidden]
    positions = torch.tensor([1, 2, 5])
    full = module.output_layer(hidden)
    with output_layer_positions(module, positions):
        partial = module.output_layer(hidden)
    torch.testing.assert_close(partial, full.index_select(0, positions))
    torch.testing.assert_close(module.output_layer(hidden), full)

    def logprobs(logits, label, label_mask):
        values = logits.log_softmax(-1).gather(-1, label.unsqueeze(-1)).squeeze(-1)
        return {"log_probs": values.masked_fill(~label_mask, 0.0)}

    label = torch.randint(0, 7, (1, 6))
    label_mask = torch.ones(1, 6, dtype=torch.bool)
    expected = logprobs(full.transpose(0, 1), label, label_mask)["log_probs"]
    scored = _at_positions(logprobs, positions)(
        partial.transpose(0, 1), label, label_mask
    )["log_probs"]
    torch.testing.assert_close(scored[:, positions], expected[:, positions])
    assert scored[:, [0, 3, 4]].count_nonzero() == 0


def _assert_chunked_sao_gradients(rank, dtype, temperature):
    from rlinf.algorithms.losses import compute_dis_actor_loss
    from rlinf.utils.distributed import vocab_parallel_log_probs_from_logits

    torch.manual_seed(3)
    full = torch.randn(1, 9, 22, dtype=dtype, requires_grad=True)
    labels = torch.tensor([[0, 21, 4, 15, 11, 8, 1, 20, 12]])
    advantages = torch.linspace(-1, 1, 9).unsqueeze(0)
    mask = torch.tensor([[True, False, True, True, False, True, True, False, True]])
    expected = (
        (full.float() / temperature)
        .log_softmax(-1)
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(-1)
    )
    old = expected.detach() + torch.linspace(-0.4, 0.4, 9).unsqueeze(0)
    delta = (expected - old).detach()
    keep = mask & (delta > torch.tensor(0.8).log()) & (delta < torch.tensor(1.2).log())
    ratio = delta.exp().masked_fill(~keep, 0.0)
    reference_loss = -(ratio * advantages * expected).sum() / mask.sum()
    reference_loss.backward()
    local = full.detach().chunk(2, dim=-1)[rank].clone().requires_grad_()
    original = local.detach().clone()
    saved = []

    def pack(tensor):
        saved.append((tensor.dtype, tensor.numel(), tensor.data_ptr()))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual = vocab_parallel_log_probs_from_logits(
            local, labels, chunk_size=4, temperature=temperature
        )
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    loss, metrics = compute_dis_actor_loss(
        logprobs=actual,
        old_logprobs=old,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=advantages,
        loss_mask=mask,
    )
    torch.testing.assert_close(loss, reference_loss, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(metrics["actor/ratio"], ratio[mask].mean().detach())
    loss.backward()
    torch.testing.assert_close(
        local.grad, full.grad.chunk(2, dim=-1)[rank], rtol=2e-3, atol=2e-6
    )
    assert torch.equal(local, original)
    assert not any(
        kind == torch.float32 and count == local.numel() and pointer != local.data_ptr()
        for kind, count, pointer in saved
    ), "Chunked logprobs retained a separate full-vocabulary FP32 tensor"


def _chunked_log_probs_rank(rank: int, store: str) -> None:
    from megatron.core import parallel_state
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=2
    )
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
        wrapper = Float16Module(
            TransformerConfig(
                num_layers=1, hidden_size=4, num_attention_heads=2, bf16=True
            ),
            torch.nn.Identity(),
        )
        sample = torch.ones(1, 9, 11, dtype=torch.bfloat16)
        assert wrapper(sample).dtype == torch.float32
        assert wrapper(sample, fp32_output=False).dtype == torch.bfloat16
        for dtype, temperature in (
            (torch.float32, 1.0),
            (torch.bfloat16, 1.0),
            (torch.bfloat16, 0.7),
        ):
            _assert_chunked_sao_gradients(rank, dtype, temperature)
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_chunked_vocab_log_probs_match_dense_tp_values_and_gradients(tmp_path):
    pytest.importorskip("megatron.core")
    torch.multiprocessing.spawn(
        _chunked_log_probs_rank, args=(str(tmp_path / "store"),), nprocs=2
    )


def _sequence_parallel_positions_rank(rank: int, store: str) -> None:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.transformer.transformer_config import TransformerConfig

    from rlinf.hybrid_engines.megatron.utils import output_layer_positions

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{store}", rank=rank, world_size=2
    )
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
        tensor_parallel.model_parallel_cuda_manual_seed(0)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        module = torch.nn.Module()
        module.output_layer = tensor_parallel.ColumnParallelLinear(
            8,
            12,
            config=config,
            init_method=torch.nn.init.normal_,
            bias=False,
            gather_output=False,
        )
        torch.manual_seed(0)
        weight = torch.randn(12, 8, device="cuda") * 0.1
        with torch.no_grad():
            module.output_layer.weight.copy_(weight.chunk(2)[rank])
        # Odd counts need padding before the scatter; one row leaves a rank
        # holding padding only.
        for positions in ([0, 3, 7], [6]):
            selected = torch.tensor(positions, device="cuda")
            full = torch.randn(8, 1, 8, device="cuda", requires_grad=True)
            full_weight = weight.clone().requires_grad_()
            expected = torch.nn.functional.linear(full, full_weight)[selected]
            expected.square().sum().backward()

            hidden = full.detach().chunk(2)[rank].clone().requires_grad_()
            module.zero_grad()
            with output_layer_positions(module, selected):
                logits, _ = module.output_layer(hidden)
            logits.square().sum().backward()

            close = {"rtol": 2e-5, "atol": 2e-6}
            torch.testing.assert_close(logits, expected.chunk(2, -1)[rank], **close)
            torch.testing.assert_close(hidden.grad, full.grad.chunk(2)[rank], **close)
            torch.testing.assert_close(
                module.output_layer.weight.grad,
                full_weight.grad.chunk(2)[rank],
                **close,
            )
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_sequence_parallel_response_logits_match_full_output_and_gradients(
    tmp_path,
):
    pytest.importorskip("megatron.training")
    if torch.cuda.device_count() < 2:
        pytest.skip("Tensor parallelism with sequence parallelism needs two GPUs.")
    torch.multiprocessing.spawn(
        _sequence_parallel_positions_rank, args=(str(tmp_path / "store"),), nprocs=2
    )


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


def test_chunked_actor_scores_variable_length_rows(monkeypatch):
    from types import SimpleNamespace

    pytest.importorskip("megatron.training")
    from megatron.core import parallel_state
    from omegaconf import OmegaConf

    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: 0)

    def score(model, module, *, label, label_mask, packed_seq_params, **inputs):
        assert packed_seq_params.cu_seqlens_q.tolist() == [0, 4, 6, 8]
        assert label.tolist() == [[2, 3, 0, 0, 0, 0, 6, 0]]
        assert label_mask.nonzero(as_tuple=True)[1].tolist() == [0, 6]
        return {"log_probs": label.float().masked_fill(~label_mask, 0)}

    actor = MAMegatronActor.__new__(MAMegatronActor)
    actor.cfg = OmegaConf.create(
        {"actor": {}, "algorithm": {"sampling_params": {"temperature": 1.0}}}
    )
    actor.transformer_config = SimpleNamespace(sequence_parallel=True)
    actor.torch_device_type = "cpu"
    actor.lm_head_chunk_size = 2
    actor.response_only_logits = False
    actor.calculate_entropy = False
    actor.return_loss = False
    actor.scoring_forward = score
    monkeypatch.setattr(actor, "_forward_multimodal_inputs", lambda batch: {})
    model = torch.nn.Module()
    model.pre_process = model.post_process = True
    rows = torch.tensor([[1, 2, 3], [4, 9, 9], [5, 6, 9]])
    batch = {
        "input_ids": rows,
        "attention_mask": rows.ne(9),
        "position_ids": torch.arange(3).expand_as(rows),
        "response_mask": torch.tensor([[0, 1, 0], [0, 0, 0], [0, 1, 0]]).bool(),
    }
    output, collect = actor.get_forward_step_func()(iter([batch]), model)
    assert collect(output).tolist() == [[0, 2, 0], [0, 0, 0], [0, 6, 0]]
