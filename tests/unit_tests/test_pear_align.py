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

"""Alignment and integration checks for unified PEAR VLM-as-Policy training."""

import asyncio
import base64
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.pear_objective import pear_reinforce_loss
from rlinf.envs.sim.libero.utils import _aggregated_libero_benchmark
from rlinf.models.embodiment.qwen_vlm.conversation import (
    IncrementalEpisodeTokens,
    assistant_positions,
    render_incremental_user_delta,
)
from rlinf.models.embodiment.qwen_vlm.protocol import (
    LIBERO_SCHEMA,
    ROBODOJO_SCHEMA,
    build_user_content,
    cameras_for_schema,
    chat_template_kwargs,
    mirror_instruction,
    native_step_budget,
    reset_wait_steps,
)


def _chat_template_path() -> Path:
    configured = os.environ.get("QWEN38_CHAT_TEMPLATE")
    candidates = [
        Path(configured) if configured else None,
        Path("/mnt/public/guozhen/models/Qwen/Qwen3.8-27B/chat_template.jinja"),
    ]
    for path in candidates:
        if path is not None and path.is_file():
            return path
    pytest.skip("Qwen3.8 chat_template.jinja is not available")


def _render_template(path: Path, **kwargs) -> str:
    from jinja2 import Environment

    source = path.read_text()
    source = source.replace("{%- generation %}", "").replace("{%- endgeneration %}", "")

    def raise_exception(message):
        raise RuntimeError(message)

    environment = Environment(autoescape=False)
    environment.globals["raise_exception"] = raise_exception
    template = environment.from_string(source)
    return template.render(
        messages=[{"role": "user", "content": "move the left mug"}],
        add_generation_prompt=True,
        **kwargs,
    )


def _tiny_png_b64() -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=(12, 34, 56)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_qwen38_template_defaults_to_xhigh_and_requests_pass_it():
    path = _chat_template_path()
    omitted = _render_template(path)
    explicit = _render_template(path, **chat_template_kwargs())
    low = _render_template(path, enable_thinking=True, reasoning_effort="low")
    disabled = _render_template(path, enable_thinking=False)
    assert "Reasoning effort is set to xhigh" in omitted
    assert "Reasoning effort is set to xhigh" in explicit
    assert "Reasoning effort is set to low" in low
    assert "Reasoning effort is set to xhigh" not in low
    assert "Reasoning effort" not in disabled
    assert chat_template_kwargs() == {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
    }
    with pytest.raises(ValueError):
        chat_template_kwargs("max")


def test_libero_instruction_budget_and_reset_match_pear():
    text = "pick up the left bowl and place it on the right plate"
    assert mirror_instruction(text) == (
        "pick up the right bowl and place it on the left plate"
    )
    assert mirror_instruction("leftover rice") == "leftover rice"
    assert native_step_budget("libero_spatial") == 220
    assert native_step_budget("libero_object") == 280
    assert native_step_budget("libero_goal") == 300
    assert native_step_budget("libero_10") == 520
    assert reset_wait_steps(True) == 10
    assert reset_wait_steps(False) == 15
    assert reset_wait_steps(True, 7) == 7


def test_gso_benchmark_keeps_each_suite_budget():
    class BaseBenchmark:
        def __init__(self, task_order_index=0):
            self.task_order_index = task_order_index

    fake_benchmark = SimpleNamespace(
        BENCHMARK_MAPPING={},
        task_maps={
            "libero_goal": {"g0": SimpleNamespace(language="goal task")},
            "libero_spatial": {"s0": SimpleNamespace(language="spatial task")},
            "libero_object": {"o0": SimpleNamespace(language="object task")},
        },
    )
    suite_cls = _aggregated_libero_benchmark(
        fake_benchmark,
        BaseBenchmark,
        "libero_gso",
        ("libero_goal", "libero_spatial", "libero_object"),
    )
    suite = suite_cls()
    assert suite.n_tasks == 3
    assert [native_step_budget(name) for name in suite.task_suite_names] == [
        300,
        220,
        280,
    ]


def test_assistant_positions_and_incremental_user_delta_preserve_newline():
    sequence = [1, 2, 10, 11, 3, 4, 20, 21]
    assert assistant_positions(sequence, [[10, 11], [20, 21]]) == [2, 3, 6, 7]

    class _ChatMLProcessor:
        eos_token = "<|im_end|>"

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=False, **kwargs
        ):
            del tokenize, kwargs
            parts = []
            for msg in messages:
                parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n<think>\n")
            return "".join(parts)

    processor = _ChatMLProcessor()
    user_msg = {"role": "user", "content": "OBS_TURN_1"}
    full_delta, no_eos_delta = render_incremental_user_delta(processor, user_msg, {})
    assert full_delta == (
        "<|im_end|>\n<|im_start|>user\nOBS_TURN_1<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    assert no_eos_delta == (
        "\n<|im_start|>user\nOBS_TURN_1<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    expected_full = processor.apply_chat_template(
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "U0"},
            {"role": "assistant", "content": "<think>\nA0"},
            user_msg,
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    turn0 = processor.apply_chat_template(
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "U0"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Sampler emitted <|im_end|> at the end of turn 0.
    assert turn0 + "A0<|im_end|>" + no_eos_delta == expected_full
    # Sampler stopped without emitting <|im_end|>.
    assert turn0 + "A0" + full_delta == expected_full


def test_chunked_head_matches_the_full_logits_and_spans_chunks():
    from rlinf.models.embodiment.qwen_vlm.fp32_head import (
        Fp32HeadLogProbs,
        response_logprobs_from_hidden,
    )

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden = torch.randn(10, 8, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(13, 8, device=device, dtype=torch.float32, requires_grad=True)
    ids = torch.randint(0, 13, (10, 1), device=device)
    Fp32HeadLogProbs.CHUNK_ROWS = 3
    try:
        log_probs, entropy = Fp32HeadLogProbs.apply(hidden, weight, ids, 1.0)
        reference_hidden = hidden.detach().requires_grad_(True)
        reference_weight = weight.detach().requires_grad_(True)
        full = reference_hidden @ reference_weight.t()
        normalizer = torch.logsumexp(full, dim=-1)
        expected = full.gather(1, ids) - normalizer[:, None]
        log_p = full - normalizer[:, None]
        expected_entropy = -(log_p.exp() * log_p).sum(-1)
        torch.testing.assert_close(log_probs, expected)
        torch.testing.assert_close(entropy, expected_entropy)
        (log_probs.sum() + entropy.sum()).backward()
        (expected.sum() + expected_entropy.sum()).backward()
        torch.testing.assert_close(hidden.grad, reference_hidden.grad)
        torch.testing.assert_close(weight.grad, reference_weight.grad)
    finally:
        Fp32HeadLogProbs.CHUNK_ROWS = 4096

    sequence = torch.randn(
        1, 6, 8, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    head = torch.randn(13, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
    tokens = torch.randint(0, 13, (1, 6), device=device)
    response_index = torch.tensor([[3, 4, 0]], device=device)
    got, got_entropy = response_logprobs_from_hidden(
        sequence, head, tokens, response_index
    )
    assert got[0, 2].item() == 0.0
    assert got_entropy[0, 2].item() == 0.0
    (got[0, 0] + got[0, 1]).backward()
    assert torch.count_nonzero(sequence.grad[:, :2]) == 0
    assert torch.count_nonzero(sequence.grad[:, 2:4]) > 0


def test_user_content_includes_camera_labels_for_libero_and_robodojo():
    state = {"arm": {"x": 1.0}, "gripper": {"width": 8.0}}
    libero_blocks = build_user_content(
        state, ["img_a", "img_w"], cameras_for_schema(LIBERO_SCHEMA)
    )
    assert [b.get("text") for b in libero_blocks if b["type"] == "text"][1:] == [
        "[agentview]",
        "[wrist]",
    ]
    robodojo_blocks = build_user_content(
        state, ["img_h", "img_l", "img_r"], cameras_for_schema(ROBODOJO_SCHEMA)
    )
    assert [b.get("text") for b in robodojo_blocks if b["type"] == "text"][1:] == [
        "[head]",
        "[left_wrist]",
        "[right_wrist]",
    ]


def test_pear_reinforce_global_token_mean_matches_full_batch():
    lp = torch.tensor(
        [[-0.5, -1.0, -1.5, -2.0], [-0.2, -0.4, 0.0, 0.0]], requires_grad=True
    )
    adv = torch.tensor([[1.0], [-1.0]]).expand_as(lp)
    mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False]], dtype=torch.bool
    )
    total_tokens = float(mask.sum().item())
    full_loss, _ = pear_reinforce_loss(lp, adv, mask)
    mb0, _ = pear_reinforce_loss(
        lp[:1], adv[:1], mask[:1], global_valid_tokens=total_tokens
    )
    mb1, _ = pear_reinforce_loss(
        lp[1:], adv[1:], mask[1:], global_valid_tokens=total_tokens
    )
    torch.testing.assert_close(mb0 + mb1, full_loss)


def test_score_centering_loss_adds_advantage_times_correction():
    torch.manual_seed(0)
    logprobs = torch.randn(2, 4, requires_grad=True)
    advantages = torch.tensor([[1.5], [-0.5]]).expand_as(logprobs)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    correction = torch.randn(2, 4, requires_grad=True)
    loss_sc, metrics = pear_reinforce_loss(
        logprobs, advantages, mask, sc_correction=correction
    )
    expected = ((-advantages * logprobs + advantages * correction)[mask]).mean()
    torch.testing.assert_close(loss_sc, expected)
    assert "actor/sc_correction" in metrics


def test_fp32_head_returns_target_and_topk_head_logprobs():
    from rlinf.models.embodiment.qwen_vlm.fp32_head import (
        Fp32HeadLogProbs,
        response_logprobs_and_heads_from_hidden,
    )

    torch.manual_seed(0)
    hidden = torch.randn(1, 5, 6, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(11, 6, dtype=torch.float32, requires_grad=True)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    response_index = torch.tensor([[2, 4, 0]], dtype=torch.long)
    head_ids = torch.tensor([[[3, 7, 9], [5, 0, 2], [0, 1, 2]]], dtype=torch.long)
    Fp32HeadLogProbs.CHUNK_ROWS = 1
    try:
        lp, _ent, head_lp = response_logprobs_and_heads_from_hidden(
            hidden, weight, input_ids, response_index, head_ids
        )
        ref_h = hidden.detach().requires_grad_(True)
        ref_w = weight.detach().requires_grad_(True)
        logits = ref_h[0] @ ref_w.t()
        log_probs_full = torch.log_softmax(logits, dim=-1)
        torch.testing.assert_close(lp[0, 0], log_probs_full[1, 3])
        torch.testing.assert_close(lp[0, 1], log_probs_full[3, 5])
        assert lp[0, 2].item() == 0.0
        torch.testing.assert_close(head_lp[0, 0], log_probs_full[1, head_ids[0, 0]])
        torch.testing.assert_close(head_lp[0, 1], log_probs_full[3, head_ids[0, 1]])
        assert torch.count_nonzero(head_lp[0, 2]) == 0

        (lp.sum() + 0.5 * head_lp.sum()).backward()
        ref_loss = (
            log_probs_full[1, 3]
            + log_probs_full[3, 5]
            + 0.5
            * (
                log_probs_full[1, head_ids[0, 0]].sum()
                + log_probs_full[3, head_ids[0, 1]].sum()
            )
        )
        ref_loss.backward()
        torch.testing.assert_close(hidden.grad, ref_h.grad)
        torch.testing.assert_close(weight.grad, ref_w.grad)
    finally:
        Fp32HeadLogProbs.CHUNK_ROWS = 4096


def test_incremental_episode_tokens_budget_and_topk_extraction():
    from rlinf.workers.rollout.session_server.payload import extract_topk_logprobs

    episode = IncrementalEpisodeTokens(response_budget=10)
    episode.start_episode(
        prompt_text="SYS+U0",
        inputs={
            "input_ids": torch.tensor([[10, 11, 12]]),
            "pixel_values": torch.ones(2, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
        },
    )
    assert episode.remaining_budget() == 10

    token_lps, topk_ids, topk_lps = extract_topk_logprobs(
        {
            "output_token_logprobs": [(-0.2, 20), (-0.3, 21)],
            "output_top_logprobs": [
                [(-0.2, 20), (-1.5, 22)],
                [(-0.3, 21), (-1.2, 23)],
            ],
        },
        [20, 21],
        topk=2,
    )
    episode.append_assistant_response(
        response_ids=[20, 21],
        response_text="A0",
        token_logprobs=token_lps,
        topk_ids=topk_ids,
        topk_logprobs=topk_lps,
    )
    assert episode.response_positions == [3, 4]
    assert episode.remaining_budget() == 8

    fits = episode.append_user_observation(
        delta_text="+U1",
        delta_inputs={
            "input_ids": torch.tensor([[30, 31, 32, 33]]),
            "pixel_values": torch.ones(2, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "mm_token_type_ids": torch.tensor([[0, 1, 1, 0]]),
        },
    )
    assert fits is True
    assert episode.remaining_budget() == 4
    episode.append_assistant_response(
        response_ids=[40, 41],
        response_text="A1",
        token_logprobs=[-0.1, -0.4],
        topk_ids=[[40, 42], [41, 43]],
        topk_logprobs=[[-0.1, -2.0], [-0.4, -1.8]],
    )
    assert episode.response_positions == [3, 4, 9, 10]
    assert episode.remaining_budget() == 2

    fits_next = episode.append_user_observation(
        delta_text="+U2",
        delta_inputs={
            "input_ids": torch.tensor([[50, 51, 52]]),
            "pixel_values": torch.ones(2, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
    )
    assert fits_next is False
    assert episode.budget_exhausted is True
    sample = episode.build_sample()
    assert sample["input_ids"].tolist() == [10, 11, 12, 20, 21, 30, 31, 32, 33, 40, 41]
    assert sample["response_index"].tolist() == [3, 4, 9, 10]
    assert sample["rollout_topk_ids"].shape == (4, 2)
    assert sample["sampler_head_log_probs"].shape == (4, 2)


def test_fsdp2_fp32_module_matching_and_ulysses_window():
    from rlinf.hybrid_engines.fsdp.ulysses import local_response_window
    from rlinf.hybrid_engines.fsdp.utils import _matches_fp32_module

    assert _matches_fp32_module("model.visual", ("model.visual",))
    assert _matches_fp32_module("model.model.visual.blocks.0", ("model.visual",))
    assert not _matches_fp32_module("model.language_model.layers.0", ("model.visual",))

    response_index = torch.tensor([[2, 4, 6, 8, 0]])
    local_pos_r0, owns_r0 = local_response_window(
        response_index, local_seq_len=4, sp_rank=0
    )
    local_pos_r1, owns_r1 = local_response_window(
        response_index, local_seq_len=4, sp_rank=1
    )
    assert owns_r0.tolist() == [[True, True, False, False, False]]
    assert local_pos_r0[0, :2].tolist() == [1, 3]
    assert owns_r1.tolist() == [[False, False, True, True, False]]
    assert local_pos_r1[0, 2:4].tolist() == [1, 3]


def test_embodied_task_dataset_encodes_specs_for_collate():
    from rlinf.agents.embodied.task_dataset import EmbodiedTaskDataset
    from rlinf.data.datasets.reasoning.collate_fn import collate_fn
    from rlinf.data.schema.reasoning_requests import build_rollout_requests_from_batch

    specs = [
        {
            "embodiment": "libero",
            "task_suite_name": "libero_gso",
            "source_suite": "libero_goal",
            "task_id": 0,
            "trial_id": 0,
            "reset_state_id": 0,
            "instruction": "open the drawer",
        }
    ]
    dataset = EmbodiedTaskDataset(specs=specs, min_size=4)
    assert len(dataset) == 4
    batch = collate_fn([dataset[i] for i in range(4)])
    requests = build_rollout_requests_from_batch(batch, group_size=8, split_size=2)
    assert len(requests) == 2
    assert requests[0].n == 8
    decoded = json.loads(requests[0].answers[0])
    assert decoded["task_suite_name"] == "libero_gso"
    assert decoded["instruction"] == "open the drawer"


def test_embodied_env_tool_worker_session_lifecycle():
    from rlinf.agents.embodied.env_tool_worker import EmbodiedEnvToolWorker
    from rlinf.agents.tool_call.schema import ToolChannelRequest

    events: list[str] = []

    class _FakeBackend:
        def __init__(self, spec):
            self.spec = spec

        def open(self):
            events.append("open")
            return {
                "state": {"arm": {"x": 0.0}, "gripper": {"width": 8.0}},
                "image_data": ["b64a", "b64b"],
                "instruction": self.spec["instruction"],
                "done": False,
                "reward": 0.0,
            }

        def step(self, action):
            events.append("step")
            assert action["gripper"]["width"] == -4.0
            return {
                "state": {"arm": {"x": 1.0}, "gripper": {"width": 4.0}},
                "image_data": None,
                "done": True,
                "reward": 1.0,
                "success": True,
                "outcome": "done",
                "native_steps": 5,
                "remaining_native_budget": 215,
            }

        def close(self):
            events.append("close")

    worker = EmbodiedEnvToolWorker.__new__(EmbodiedEnvToolWorker)
    worker._backend_factory = _FakeBackend
    worker._sessions = {}
    worker._semaphore = None
    worker._executor = None

    async def _run():
        r0 = await worker.handle_request(
            ToolChannelRequest(
                session_id="s1",
                request_type="session_start",
                tool_args={"instruction": "pick cup"},
            )
        )
        assert r0.success is True
        assert r0.result["instruction"] == "pick cup"
        r1 = await worker.handle_request(
            ToolChannelRequest(
                session_id="s1",
                request_type="execute",
                tool_name="embodied_env",
                tool_args={"action": {"gripper": {"width": -4.0}}},
            )
        )
        assert r1.success is True
        assert r1.result["done"] is True and r1.result["reward"] == 1.0
        r2 = await worker.handle_request(
            ToolChannelRequest(session_id="s1", request_type="session_end")
        )
        assert r2.success is True and r2.result == {"closed": True}
        # Idempotent second close.
        r3 = await worker.handle_request(
            ToolChannelRequest(session_id="s1", request_type="session_end")
        )
        assert r3.success is True and r3.result == {"closed": False}

    asyncio.run(_run())
    assert events == ["open", "step", "close"]


def test_embodied_vlm_agent_loop_multi_turn_and_cleanup_on_discard():
    from rlinf.agents.embodied.vlm_agent_loop import EmbodiedVLMAgentLoopWorker
    from rlinf.agents.tool_call.schema import ToolChannelResponse
    from rlinf.workers.agent.agent_loop import GroupDiscarded
    from rlinf.workers.agent.session_server_client import (
        SessionBinding,
        SessionDiscardedError,
    )

    valid_json = (
        "<think>move</think>\n```json\n"
        '{"arm": {"x": 1.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}, '
        '"gripper": {"width": 0.0}}\n```'
    )

    class _FakeProcessor:
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=False, **kwargs
        ):
            del tokenize, kwargs
            return "|".join(m["role"] for m in messages) + (
                "|gen" if add_generation_prompt else ""
            )

    class _FakeTokenizer:
        eos_token_id = 99
        eos_token = "<eos>"

        @staticmethod
        def encode(text, add_special_tokens=False):
            del add_special_tokens
            return [1] * min(len(text), 4)

        @staticmethod
        def decode(ids, skip_special_tokens=False):
            assert skip_special_tokens is False
            del ids
            return valid_json

    def _make_worker(gen_responses, *, max_turns=None):
        worker = EmbodiedVLMAgentLoopWorker.__new__(EmbodiedVLMAgentLoopWorker)
        worker.tokenizer = _FakeTokenizer()
        worker._processor = _FakeProcessor()
        worker._schema = LIBERO_SCHEMA
        worker._cameras = cameras_for_schema(LIBERO_SCHEMA)
        worker._template_kwargs = chat_template_kwargs("xhigh")
        worker._response_budget = 100
        worker._max_turns = max_turns
        worker._tokens_per_image_estimate = 2
        worker.log_warning = lambda msg: None

        env_calls: list[str] = []
        seen_sampling_params: list[dict] = []
        gen_iter = iter(gen_responses)

        async def fake_call_env(env_sid, req_type, tool_args=None):
            del env_sid, tool_args
            env_calls.append(req_type)
            if req_type == "session_start":
                return ToolChannelResponse(
                    success=True,
                    result={
                        "state": {
                            "arm": {
                                "x": 0.0,
                                "y": 0.0,
                                "z": 0.0,
                                "roll": 0.0,
                                "pitch": 0.0,
                                "yaw": 0.0,
                            },
                            "gripper": {"width": 8.0},
                        },
                        "image_data": ["img0a", "img0b"],
                        "instruction": "pick cup",
                        "done": False,
                        "reward": 0.0,
                    },
                )
            if req_type == "execute":
                step_n = env_calls.count("execute")
                done = step_n >= 2
                return ToolChannelResponse(
                    success=True,
                    result={
                        "state": {
                            "arm": {
                                "x": float(step_n),
                                "y": 0.0,
                                "z": 0.0,
                                "roll": 0.0,
                                "pitch": 0.0,
                                "yaw": 0.0,
                            },
                            "gripper": {"width": 8.0},
                        },
                        "image_data": ["img1a", "img1b"],
                        "done": done,
                        "reward": 1.0 if done else 0.0,
                    },
                )
            return ToolChannelResponse(success=True, result={"closed": True})

        class _FakeSessionClient:
            async def generate_multimodal(self, *args, **kwargs):
                del args
                seen_sampling_params.append(dict(kwargs.get("sampling_params") or {}))
                nxt = next(gen_iter)
                if isinstance(nxt, BaseException):
                    raise nxt
                return nxt

        worker._call_env_tool = fake_call_env
        worker.session_server_client = _FakeSessionClient()
        return worker, env_calls, seen_sampling_params

    binding = SessionBinding(group_id=1, traj_index=0, session_id="sid0")
    spec = json.dumps({"instruction": "pick cup"})

    # Normal 2-turn episode: even when engine text lacks </think>, decoding
    # output_ids with skip_special_tokens=False recovers the valid JSON action.
    worker, env_calls, seen_params = _make_worker(
        [
            {
                "output_ids": [10, 11, 99],
                "text": "stripped text without json",
                "finish_reason": {"type": "stop"},
            },
            {
                "output_ids": [12, 13, 99],
                "text": "stripped text without json",
                "finish_reason": {"type": "stop"},
            },
        ]
    )
    out, close_fields, reward = asyncio.run(
        worker.run_session_trajectory(binding, [0], spec)
    )
    assert out is None
    assert reward == 1.0
    assert close_fields == {"kept_lens": {0: 3, 1: 3}, "is_end": {0: False, 1: True}}
    assert env_calls == ["session_start", "execute", "execute", "session_end"]
    assert all(p.get("skip_special_tokens") is False for p in seen_params)

    # Discard mid-episode still runs session_end in finally.
    worker2, env_calls2, _ = _make_worker([SessionDiscardedError("group aborted")])
    with pytest.raises(GroupDiscarded, match="group aborted"):
        asyncio.run(worker2.run_session_trajectory(binding, [0], spec))
    assert env_calls2 == ["session_start", "session_end"]


def test_assemble_group_payload_merge_continuous_turns_multimodal_and_topk():
    from rlinf.data.schema.reasoning_results import DynamicRolloutResult
    from rlinf.workers.rollout.session_server.assembly import assemble_group_payload
    from rlinf.workers.rollout.session_server.payload import GroupPayload, PayloadPart

    img = _tiny_png_b64()

    class _FakeProcessor:
        def __call__(self, text, images=None, return_tensors="pt", **kwargs):
            del return_tensors, kwargs
            t = text[0]
            ids = [10, 11, 12] if t == "SYS+U0" else [30, 31]
            n_imgs = len(images) if images else 1
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "pixel_values": torch.ones((n_imgs, 4), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * n_imgs, dtype=torch.long),
                "mm_token_type_ids": torch.tensor([[0] * len(ids)], dtype=torch.long),
            }

    parts = [
        PayloadPart(
            group_id=5,
            traj_index=0,
            turn_index=0,
            input_ids=[],
            output_ids=[20, 21],
            logprobs=[-0.2, -0.3],
            version_start=1,
            weight_version=1,
            finish_reason={"type": "stop"},
            delta_text="SYS+U0",
            image_data=[img, img],
            output_text="A0",
            topk_ids=[[20, 22], [21, 23]],
            topk_logprobs=[[-0.2, -1.5], [-0.3, -1.4]],
        ),
        PayloadPart(
            group_id=5,
            traj_index=0,
            turn_index=1,
            input_ids=[],
            output_ids=[40, 41],
            logprobs=[-0.1, -0.5],
            version_start=1,
            weight_version=2,
            finish_reason={"type": "stop"},
            delta_text="+U1",
            image_data=[img, img],
            output_text="A1",
            topk_ids=[[40, 42], [41, 43]],
            topk_logprobs=[[-0.1, -2.0], [-0.5, -1.8]],
        ),
        PayloadPart(
            group_id=5,
            traj_index=-1,
            turn_index=-1,
            input_ids=[],
            output_ids=[],
            logprobs=None,
            version_start=1,
            weight_version=2,
            finish_reason={},
            last=True,
            rewards=[1.0],
            verdicts={
                0: {
                    "kept_lens": {0: 2, 1: 2},
                    "continues_prev": [False, True],
                    "is_end": {0: False, 1: True},
                    "not_training": {},
                    "extra_fields_train": {},
                }
            },
        ),
    ]
    result = assemble_group_payload(
        GroupPayload(group_id=5, parts=parts),
        return_logprobs=True,
        merge_continuous_turns=True,
        processor=_FakeProcessor(),
        pixel_dtype="bf16",
    )
    assert result.num_sequence == 1
    assert result.input_ids == [[10, 11, 12, 20, 21, 30, 31, 40, 41]]
    assert result.prompt_lengths == [3]
    assert result.response_lengths == [6]
    assert result.response_mask == [
        [False, False, False, True, True, False, False, True, True]
    ]
    assert result.rollout_logprobs == [[-0.2, -0.3, 0.0, 0.0, -0.1, -0.5]]
    assert result.multi_modal_inputs is not None
    assert result.multi_modal_inputs[0]["pixel_values"].shape == (4, 4)
    assert result.multi_modal_inputs[0]["pixel_values"].dtype == torch.bfloat16
    assert result.multi_modal_inputs[0]["image_grid_thw"].shape == (4, 3)
    assert result.rollout_topk_ids is not None and len(result.rollout_topk_ids[0]) == 4

    actor_batch = result.to_actor_batch()
    assert actor_batch["response_mask"][0].tolist() == result.response_mask[0]
    assert actor_batch["rollout_topk_ids"][0].shape == (4, 2)
    mat = DynamicRolloutResult.materialize_actor_batch(
        actor_batch, seq_length=9, pad_token=0
    )
    assert mat["rollout_topk_ids"][0].shape == (4, 2)

    # Inconsistent top-K presence across turns must fail loudly.
    broken_parts = [
        parts[0],
        PayloadPart(**{**parts[1].__dict__, "topk_ids": None, "topk_logprobs": None}),
        parts[2],
    ]
    with pytest.raises(ValueError, match="inconsistent top-K"):
        assemble_group_payload(
            GroupPayload(group_id=5, parts=broken_parts),
            return_logprobs=True,
            merge_continuous_turns=True,
            processor=_FakeProcessor(),
        )


def test_extract_topk_logprobs_fails_loudly_on_missing_or_short_heads():
    from rlinf.workers.rollout.session_server.payload import extract_topk_logprobs

    assert extract_topk_logprobs({}, [], topk=4) == (None, [], [])
    with pytest.raises(ValueError, match="output_top_logprobs"):
        extract_topk_logprobs({}, [10, 11], topk=4)
    with pytest.raises(ValueError, match="expected 4"):
        extract_topk_logprobs(
            {"output_top_logprobs": [[(-0.2, 10), (-1.5, 11)]]},
            [10],
            topk=4,
        )


def test_pad_actor_batch_rows_handles_multimodal_and_topk_columns(monkeypatch):
    from rlinf.data.schema.actor_batch import pad_actor_batch_rows

    def _fake_all_reduce(tensor, op=None, group=None):
        del op, group
        tensor.fill_(3)

    monkeypatch.setattr(torch.distributed, "all_reduce", _fake_all_reduce)
    batch = {
        "input_ids": [torch.tensor([1, 2, 3], dtype=torch.long)],
        "attention_mask": [torch.tensor([True, True, True], dtype=torch.bool)],
        "response_mask": [torch.tensor([False, True, True], dtype=torch.bool)],
        "multi_modal_inputs": [
            {"pixel_values": torch.ones(2, 4, dtype=torch.bfloat16)}
        ],
        "rollout_topk_ids": [torch.tensor([[1, 2], [3, 4]], dtype=torch.long)],
        "rollout_topk_logprobs": [torch.tensor([[-0.1, -1.0], [-0.2, -1.2]])],
        "rewards": torch.tensor([1.0]),
        "idx_to_traj": [0],
    }
    padded = pad_actor_batch_rows(batch, divisor=2, dp_group=None, device="cpu")
    assert len(padded["input_ids"]) == 4
    assert padded["multi_modal_inputs"][0] is not None
    assert padded["multi_modal_inputs"][1:] == [None, None, None]
    assert padded["rollout_topk_ids"][0].shape == (2, 2)
    assert [t.shape for t in padded["rollout_topk_ids"][1:]] == [(0, 2), (0, 2), (0, 2)]
    assert [t.shape for t in padded["rollout_topk_logprobs"][1:]] == [
        (0, 2),
        (0, 2),
        (0, 2),
    ]


def test_install_fp32_residual_stream_keeps_residual_fp32_and_norms_bf16():
    from rlinf.hybrid_engines.fsdp.utils import install_fp32_residual_stream

    class _Block(torch.nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.input_layernorm = torch.nn.RMSNorm(dim)
            self.attn = torch.nn.Linear(dim, dim, bias=False).to(torch.bfloat16)
            self.post_attention_layernorm = torch.nn.RMSNorm(dim)
            self.mlp = torch.nn.Linear(dim, dim, bias=False).to(torch.bfloat16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            normed = self.input_layernorm(x)
            assert normed.dtype == torch.bfloat16
            x = x + self.attn(normed)
            assert x.dtype == torch.float32
            normed2 = self.post_attention_layernorm(x)
            assert normed2.dtype == torch.bfloat16
            x = x + self.mlp(normed2)
            assert x.dtype == torch.float32
            return x

    class _MiniModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8).to(torch.bfloat16)
            self.layer = _Block(8)
            self.norm = torch.nn.RMSNorm(8)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            h = self.embed(ids)
            assert h.dtype == torch.float32
            return self.norm(self.layer(h))

    model = _MiniModel()
    install_fp32_residual_stream(model, compute_dtype=torch.bfloat16)
    out = model(torch.tensor([[1, 2, 3]]))
    assert out.dtype == torch.float32


def test_streamed_adamw_updates_bf16_and_fp32_params_in_fp32():
    from rlinf.hybrid_engines.fsdp.optim import StreamedAdamW

    p_bf16 = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    p_fp32 = torch.nn.Parameter(torch.ones(8, dtype=torch.float32))
    p_ref = torch.nn.Parameter(torch.ones(8, dtype=torch.float32))
    opt = torch.optim.AdamW(
        [p_bf16, p_fp32], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    ref_opt = torch.optim.AdamW(
        [p_ref], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    streamed = StreamedAdamW(opt, chunk_bytes=128)
    assert "fp32_master_param" in opt.state[p_bf16]
    assert "fp32_master_param" not in opt.state[p_fp32]

    for _ in range(3):
        p_bf16.grad = torch.full_like(p_bf16, 0.5)
        p_fp32.grad = torch.full_like(p_fp32, 0.5)
        p_ref.grad = torch.full_like(p_ref, 0.5)
        assert streamed.is_ready()
        streamed.step()
        ref_opt.step()

    torch.testing.assert_close(p_fp32.detach(), p_ref.detach())
    torch.testing.assert_close(
        opt.state[p_bf16]["fp32_master_param"].detach(), p_ref.detach()
    )
    torch.testing.assert_close(p_bf16.detach(), p_ref.detach().to(torch.bfloat16))


def test_validate_session_server_score_centering_guards():
    from rlinf.config import validate_session_server_cfg

    def _base_cfg(**overrides):
        cfg = OmegaConf.create(
            {
                "cluster": {"auto_scheduler": False},
                "runner": {"async_mode": True},
                "actor": {"training_backend": "fsdp"},
                "reward": {"reward_type": "embodied_vlm"},
                "algorithm": {
                    "loss_type": "pear_reinforce",
                    "pear": {"score_centering": True},
                    "sampling_params": {"top_p": 1.0, "top_k": -1},
                },
                "rollout": {
                    "pause_mode": "retract",
                    "sglang": {"backend_type": "server"},
                    "router": {
                        "type": "sglang",
                        "policy": "manual",
                        "disable_health_check": True,
                        "request_timeout_secs": 1800,
                    },
                },
                "session_server": {
                    "enable": True,
                    "engine_streaming_session": True,
                    "topk_log_probs": 32,
                },
            }
        )
        for key, value in overrides.items():
            OmegaConf.update(cfg, key, value, merge=False)
        return cfg

    with pytest.raises(ValueError, match="topk_log_probs > 0"):
        validate_session_server_cfg(_base_cfg(**{"session_server.topk_log_probs": 0}))
    with pytest.raises(ValueError, match="pear_reinforce"):
        validate_session_server_cfg(_base_cfg(**{"algorithm.loss_type": "actor"}))
    with pytest.raises(ValueError, match="full-support"):
        validate_session_server_cfg(
            _base_cfg(**{"algorithm.sampling_params.top_p": 0.9})
        )


def test_disaggregated_pear_configs_compose_and_match_session_server_contract():
    from hydra import compose, initialize_config_dir

    config_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"
    )
    shared = OmegaConf.load(config_dir / "hybrid_engines" / "sglang_router_server.yaml")
    assert shared.server.attention_backend == "triton"
    for name, expected_nodes, expected_batch, expected_turns in (
        ("libero_gso_grpo_pear_sglang", 2, 8, 300),
        ("libero_spatial_grpo_pear_sglang_a100_4node", 4, 32, 300),
        ("libero_gso_grpo_pear_sglang_paper", 4, 128, 300),
        ("libero_10_grpo_pear_stage2", 2, 8, 520),
        ("pear_stage3_libero_robodojo", 2, 8, 520),
    ):
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name=name)
        assert cfg.cluster.num_nodes == expected_nodes
        assert cfg.data.rollout_batch_size == expected_batch
        assert cfg.agentloop.max_turns == expected_turns
        assert cfg.runner.task_type == "reasoning"
        assert cfg.runner.async_mode is True
        assert cfg.session_server.enable is True
        assert cfg.session_server.merge_continuous_turns is True
        assert cfg.session_server.engine_streaming_session is True
        assert cfg.session_server.topk_log_probs == 32
        assert cfg.algorithm.pear.score_centering is True
        assert cfg.rollout.sglang.backend_type == "server"
        assert cfg.rollout.sglang.attention_backend is None
        assert cfg.rollout.sglang.max_mamba_cache_size == 256
        assert list(cfg.rollout.sglang.fp32_modules) == ["visual", "A_log"]
        assert cfg.rollout.router.policy == "manual"
        assert cfg.algorithm.recompute_logprobs is False
        assert cfg.algorithm.loss_type == "pear_reinforce"
        assert cfg.actor.training_backend == "fsdp"
        assert cfg.actor.fsdp_config.strategy == "fsdp2"
        assert cfg.reward.reward_type == "embodied_vlm"
        if name == "libero_spatial_grpo_pear_sglang_a100_4node":
            assert cfg.rollout.tensor_parallel_size == 4
            assert cfg.rollout.sglang.reasoning_effort == "low"
            assert cfg.runner.seq_length == 32768
            assert cfg.rollout.sglang.response_budget == 28672
            assert cfg.session_server.num_servers == 3
            assert cfg.rollout_buffer.max_inflight_groups == 18
            assert cfg.actor.optim.streamed_adamw is True
            assert cfg.actor.optim.streamed_adamw_chunk_bytes == 536870912
        if name == "libero_gso_grpo_pear_sglang_paper":
            assert cfg.rollout_buffer.max_inflight_groups == 16
            assert cfg.actor.optim.streamed_adamw is True
