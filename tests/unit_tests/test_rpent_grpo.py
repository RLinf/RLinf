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

"""Unit tests for RLinf RPent GRPO agent loop, runner, and Qwen3.5 convertor."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rlinf.agents.rpent import (
    SimulatorBatchSampler,
    episode_task,
    merge_prefix_turns,
    mixed_simulators,
)
from rlinf.runners.rpent_runner import _aggregate_metrics
from rlinf.utils.convertor.utils import Qwen35Convertor
from rlinf.workers.agent.agent_loop import AgentLoopOutput


def test_episode_task_parsing():
    raw = '{"environment": "libero", "suite": "libero_goal", "task": 0, "seed": 1}'
    assert episode_task(raw)["environment"] == "libero"
    assert episode_task([raw])["seed"] == 1
    with pytest.raises(ValueError):
        episode_task([raw, raw])


def test_merge_prefix_turns_respects_chain_and_budget():
    t0 = AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        response_text="a",
        response_logprobs=[-0.1, -0.2],
        response_weight_versions=[0, 0],
        response_images={"k": 0},
        version_start=0,
        version_end=0,
        is_end=False,
        reward_score=1.0,
        extra_fields={"turn_id": 0, "chain_id": 0, "tool_names": ["view_env"]},
    )
    t1 = AgentLoopOutput(
        prompt_ids=[1, 2, 3, 4, 5, 99, 98],
        response_ids=[6, 7],
        response_text="b",
        response_logprobs=[-0.3, -0.4],
        response_weight_versions=[0, 0],
        response_images={"k": 1},
        version_start=0,
        version_end=0,
        is_end=False,
        reward_score=1.0,
        extra_fields={"turn_id": 1, "chain_id": 0, "tool_names": ["act"]},
    )
    t2 = AgentLoopOutput(
        prompt_ids=[1, 2, 77],
        response_ids=[8],
        response_text="c",
        response_logprobs=[-0.5],
        response_weight_versions=[0],
        response_images={"k": 2},
        version_start=0,
        version_end=0,
        is_end=True,
        reward_score=1.0,
        extra_fields={"turn_id": 2, "chain_id": 1, "tool_names": []},
    )
    merged = merge_prefix_turns([t0, t1, t2], max_scored_tokens=16)
    assert len(merged) == 2
    assert merged[0].prompt_ids == [1, 2, 3]
    assert merged[0].response_ids == [4, 5, 99, 98, 6, 7]
    assert merged[0].response_mask == [1, 1, 0, 0, 1, 1]
    assert merged[0].response_logprobs == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]
    assert merged[0].response_images == {"k": 1}
    assert merged[0].extra_fields["turn_ids"] == [0, 1]
    assert merged[1].extra_fields["turn_ids"] == [2]

    # Budget split: max_scored_tokens=3 forces t0 and t1 into separate rows.
    split = merge_prefix_turns([t0, t1], max_scored_tokens=3)
    assert len(split) == 2


@pytest.mark.parametrize("second", ["robotwin", "metaworld"])
def test_mixed_simulators_and_sampler_quotas(second):
    cfg = OmegaConf.create(
        {
            "algorithm": {"group_size": 8},
            "data": {"rollout_batch_size": 2, "sampling_batch_size": 4},
            "env": {
                "rpent": {
                    "libero": {
                        "endpoint": "http://127.0.0.1:28561",
                        "train_data_paths": "/tmp/a.jsonl",
                    },
                    second: {
                        "endpoint": "http://127.0.0.1:28562",
                        "train_data_paths": "/tmp/b.jsonl",
                    },
                }
            },
        }
    )
    sims = mixed_simulators(cfg)
    assert set(sims) == {"libero", second}
    assert sims["libero"].rollout_groups == 1
    assert sims["libero"].sampling_groups == 2

    sampler = SimulatorBatchSampler([10, 10], [2, 2], shuffle=True, seed=42)
    batches = list(sampler)
    assert len(batches) == 5
    for batch in batches:
        assert len(batch) == 4
        assert sum(idx < 10 for idx in batch) == 2
        assert sum(idx >= 10 for idx in batch) == 2


def test_aggregate_metrics_and_qwen35_bf16_norm():
    agg = _aggregate_metrics(
        [
            {"__mean__/rpent/success_rate": (1, 1), "__sum__/rpent/episodes": 1},
            {"__mean__/rpent/success_rate": (0, 1), "__sum__/rpent/episodes": 1},
        ]
    )
    assert agg["rpent/success_rate"] == pytest.approx(0.5)
    assert agg["rpent/episodes"] == pytest.approx(2.0)

    import torch

    # Verify Qwen3.5 GDN out_norm weight adds +1.0 in fp32 before casting back to bf16
    compact_src = " ".join(inspect.getsource(Qwen35Convertor).split())
    assert "tensor.float() + 1.0" in compact_src
    assert ").to(tensor.dtype)" in compact_src
    conv = object.__new__(Qwen35Convertor)
    conv.convert_rules = []
    t = torch.tensor([0.001953125, -0.5, 0.25], dtype=torch.bfloat16)
    out = Qwen35Convertor.convert(
        conv, {"decoder.layers.0.self_attention.out_norm.weight": t}
    )
    got = out["model.language_model.layers.0.linear_attn.norm.weight"]
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, (t.float() + 1.0).to(torch.bfloat16))


@pytest.mark.parametrize("second", ["robotwin", "metaworld"])
def test_validate_sim_pool_capacity(second):
    from rlinf.agents.rpent.simulators import SimulatorSpec
    from rlinf.runners.rpent_runner import validate_sim_pool_capacity

    sims = {
        "libero": SimulatorSpec(
            "libero", "http://127.0.0.1:28561", "/tmp/a.jsonl", 1, 2
        ),
        second: SimulatorSpec(second, "http://127.0.0.1:28562", "/tmp/b.jsonl", 1, 2),
    }

    # Exact capacity succeeds with uncapped client
    statuses = validate_sim_pool_capacity(
        simulators=sims,
        default_endpoint=None,
        sampling_groups=4,
        group_size=8,
        num_agent_workers=4,
        episodes_per_worker=None,
        fetch_health=lambda ep: {"ok": True, "workers_total": 16, "workers_idle": 16},
    )
    assert statuses["libero"]["workers_total"] == 16
    assert statuses[second]["required"] == 16

    # Undersized simulator pool fails fast
    with pytest.raises(RuntimeError, match="workers_total=10"):
        validate_sim_pool_capacity(
            simulators=sims,
            default_endpoint=None,
            sampling_groups=4,
            group_size=8,
            num_agent_workers=4,
            episodes_per_worker=None,
            fetch_health=lambda ep: {
                "ok": True,
                "workers_total": 10,
                "workers_idle": 10,
            },
        )

    # Throttling client episodes_per_worker fails fast
    with pytest.raises(RuntimeError, match="episodes_per_worker=4"):
        validate_sim_pool_capacity(
            simulators=sims,
            default_endpoint=None,
            sampling_groups=4,
            group_size=8,
            num_agent_workers=4,
            episodes_per_worker=4,
            fetch_health=lambda ep: {
                "ok": True,
                "workers_total": 16,
                "workers_idle": 16,
            },
        )


def test_zero_base64_conversation_history_and_local_staging():
    from pathlib import Path

    from rlinf.agents.rpent.agent_loop import (
        _BINARY_EXECUTE_MEDIA_TYPE,
        _unpack_execute_response,
    )
    from rlinf.agents.rpent.history import AppendOnlyHistory, EpisodeConversation
    from rlinf.hybrid_engines.sglang.multimodal.replay import stage_local_images

    class _FakeInnerTok:
        def convert_tokens_to_ids(self, tok: str) -> int:
            return 999 if tok == "<|image_pad|>" else 1

        def __call__(self, text: str, add_special_tokens: bool = False):
            # Map each <|image_pad|> marker to token 999, words to dummy tokens
            ids = []
            for part in text.split():
                ids.append(
                    999 if "<|image_pad|>" in part else (abs(hash(part)) % 100 + 10)
                )
            return {"input_ids": ids}

    class _FakeAgenticTok:
        def __init__(self):
            self.tokenizer = _FakeInnerTok()

        def apply_chat_template(
            self, messages, tools=None, tokenize=False, add_generation_prompt=True
        ):
            chunks = []
            for m in messages:
                c = m.get("content", "")
                if isinstance(c, list):
                    for part in c:
                        if part.get("type") == "image_url":
                            chunks.append("<|image_pad|>")
                        else:
                            chunks.append(part.get("text", ""))
                else:
                    chunks.append(str(c))
            return " ".join(chunks)

        def render_incremental(self, delta_msgs):
            text = self.apply_chat_template(delta_msgs)
            return self.tokenizer(text)["input_ids"]

        def join_increment(self, prefix_ids, increment):
            return [*prefix_ids, *increment]

    png1 = b"\x89PNG\r\n\x1a\nfake1"
    png2 = b"\x89PNG\r\n\x1a\nfake2"
    conv = EpisodeConversation(
        system_prompt="SYS",
        user_message="USR",
        initial_observation_text="OBS0",
        tools=[],
        environment="libero",
    )
    hist = AppendOnlyHistory(_FakeAgenticTok())

    # Turn 0 (no images)
    m0 = conv.wire_messages()
    r0 = hist.render_turn(m0, conv.tools, conv.images())
    assert r0.images == []
    a0 = {
        "role": "assistant",
        "content": "call",
        "tool_calls": [
            {
                "id": "c0",
                "type": "function",
                "function": {"name": "view_env", "arguments": "{}"},
            }
        ],
    }
    hist.record(m0, conv.tools, r0.input_ids, [42, 43], a0)
    conv.append_assistant(a0)
    conv.append_tool_result(
        tool_name="view_env", tool_call_id="c0", text="ok", images=[png1]
    )

    # Turn 1 (continues prefix, zero base64, raw bytes preserved by identity)
    m1 = conv.wire_messages()
    assert all("base64" not in str(m) for m in m1)
    r1 = hist.render_turn(m1, conv.tools, conv.images())
    assert r1.images == [png1]
    assert r1.images[0] is png1
    assert len(r1.image_spans) == 1
    assert hist.chain_id == 1

    # Turn 2 (adds second image, reuses first raw bytes without copy or decode)
    a1 = {
        "role": "assistant",
        "content": "call2",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "view_env", "arguments": "{}"},
            }
        ],
    }
    hist.record(m1, conv.tools, r1.input_ids, [44], a1)
    conv.append_assistant(a1)
    conv.append_tool_result(
        tool_name="view_env", tool_call_id="c1", text="ok2", images=[png2]
    )
    r2 = hist.render_turn(conv.wire_messages(), conv.tools, conv.images())
    assert r2.images == [png1, png2]
    assert r2.images[0] is png1 and r2.images[1] is png2
    assert hist.chain_id == 1

    # Local staging writes once and returns stable .png file paths
    staged_a = stage_local_images([png1, png2])
    staged_b = stage_local_images([png1, png2])
    assert staged_a == staged_b
    assert all(
        isinstance(p, str) and p.endswith(".png") and Path(p).read_bytes() == raw
        for p, raw in zip(staged_a, [png1, png2])
    )

    # Binary execute unpacker sanity
    import json
    import struct

    meta = {"text": "hi", "done": False, "success": False, "truncated": False}
    meta_b = json.dumps(meta).encode()
    buf = (
        struct.pack("!I", len(meta_b))
        + meta_b
        + struct.pack("!I", 1)
        + struct.pack("!I", len(png1))
        + png1
    )
    got_meta, got_imgs = _unpack_execute_response(buf)
    assert got_meta == meta and got_imgs == [png1]
    assert _BINARY_EXECUTE_MEDIA_TYPE.startswith("application/")


def test_validate_sim_pool_capacity_48gpu_scale() -> None:
    """Verify 48-GPU 384-episode (192 LIBERO + 192 RoboTwin) capacity validation."""
    from rlinf.agents.rpent.simulators import SimulatorSpec
    from rlinf.runners.rpent_runner import validate_sim_pool_capacity

    sims = {
        "libero": SimulatorSpec("libero", "http://127.0.0.1:28561", "a.jsonl", 16, 24),
        "robotwin": SimulatorSpec(
            "robotwin", "http://127.0.0.1:28562", "b.jsonl", 16, 24
        ),
    }
    res = validate_sim_pool_capacity(
        simulators=sims,
        default_endpoint=None,
        sampling_groups=48,
        group_size=8,
        num_agent_workers=32,
        episodes_per_worker=None,
        fetch_health=lambda _ep: {"workers_total": 192, "workers_idle": 192},
    )
    assert res["libero"]["required"] == 192
    assert res["robotwin"]["required"] == 192


def test_rpent_runner_pool_sleep_wake_lifecycle(monkeypatch) -> None:
    """Verify RPentGRPORunner._pool_lifecycle posts /pool/{sleep,wake} to all endpoints."""
    import httpx
    from omegaconf import OmegaConf

    from rlinf.agents.rpent.simulators import SimulatorSpec
    from rlinf.runners.rpent_runner import RPentGRPORunner

    calls: list[str] = []

    class _FakeResp:
        def __init__(self, url: str):
            self.url = url

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            sleeping = self.url.endswith("/sleep")
            return {
                "ok": True,
                "environment": "libero" if "28561" in self.url else "robotwin",
                "workers_target": 192,
                "workers_total": 32 if sleeping else 192,
                "sleeping": sleeping,
            }

    def _fake_post(url, **kwargs):
        calls.append(url)
        return _FakeResp(url)

    monkeypatch.setattr(httpx, "post", _fake_post)

    runner = object.__new__(RPentGRPORunner)
    runner.simulators = {
        "libero": SimulatorSpec("libero", "http://127.0.0.1:28561", "a.jsonl", 16, 24),
        "robotwin": SimulatorSpec(
            "robotwin", "http://127.0.0.1:28562", "b.jsonl", 16, 24
        ),
    }
    runner.cfg = OmegaConf.create(
        {"tools": {"rpent": {"sleep_sim_pools_during_training": True}}}
    )

    assert runner._sleep_pools_enabled() is True
    runner._pool_lifecycle("sleep")
    runner._pool_lifecycle("wake")
    assert set(calls) == {
        "http://127.0.0.1:28561/pool/sleep",
        "http://127.0.0.1:28562/pool/sleep",
        "http://127.0.0.1:28561/pool/wake",
        "http://127.0.0.1:28562/pool/wake",
    }


def test_metaworld_task_pools(tmp_path, monkeypatch):
    from examples.agent.rpent.prepare_tasks import write_metaworld_tasks

    def initial_scene(name, index, benchmark):
        return SimpleNamespace(close=lambda: None), {
            "task_index": index,
            "rand_vec_sha256": name,
        }

    monkeypatch.setitem(
        sys.modules, "metaworld", SimpleNamespace(MT1=lambda name, seed: name)
    )
    monkeypatch.setitem(
        sys.modules,
        "robots.rl.envs.metaworld",
        SimpleNamespace(initial_scene=initial_scene),
    )
    args = SimpleNamespace(seeds="49", output_dir=tmp_path)
    write_metaworld_tasks(args)
    pools = {}
    for split, count in [("mt50", 50), ("ml45-train", 45), ("ml45-ood", 5)]:
        payload = (tmp_path / f"{split}.jsonl").read_bytes()
        rows = [json.loads(line) for line in payload.splitlines()]
        manifest = json.loads((tmp_path / f"{split}-manifest.json").read_bytes())
        assert len(rows) == count
        assert manifest["files"][split] == {
            "rows": count,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        pools[split] = {r["query_id"] for r in rows}
        for row in rows:
            task = episode_task(row["answer"])
            assert task["environment"] == "metaworld"
            assert (
                task["task_index"]
                == task["init_state_id"]
                == task["seed"]
                == row["initial_state"]["task_index"]
                == 49
            )
    assert pools["ml45-train"].isdisjoint(pools["ml45-ood"])
    assert pools["ml45-train"] | pools["ml45-ood"] == pools["mt50"]
    args.seeds = "50"
    with pytest.raises(ValueError, match="0, 49"):
        write_metaworld_tasks(args)


def test_metaworld_validation_retry():
    from rlinf.agents.rpent.history import EpisodeConversation

    move = {
        "type": "object",
        "properties": {"action": {"const": "move_to"}, "xyz": {"type": "array"}},
        "required": ["action", "xyz"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {"actions": {"type": "array", "items": move}},
    }
    conv = EpisodeConversation(
        system_prompt="SYS",
        user_message="TASK",
        initial_observation_text="{}",
        environment="metaworld",
        tools=[
            {
                "type": "function",
                "function": {"name": "execute_actions", "parameters": schema},
            }
        ],
    )
    action = {"action": "move_to", "xyz": [1, 2, 3], "orientation_rpy": [0, 0, 0]}
    kind, message, parsed, retry = conv.validate_or_retry(
        [{"name": "execute_actions", "arguments": {"actions": [action]}}]
    )
    assert kind == "tool_arguments" and parsed is None and retry
    assert "xyz only" in message
