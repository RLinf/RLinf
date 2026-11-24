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

"""Alignment checks for synchronous PEAR GRPO with SGLang."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rlinf.algorithms.pear_objective import (
    dapo_group_mask,
    fill_dapo_groups,
    keep_last_valid_step,
    pear_reinforce_loss,
)
from rlinf.envs.sim.libero.utils import _aggregated_libero_benchmark
from rlinf.models.embodiment.qwen_vlm.conversation import assistant_positions
from rlinf.models.embodiment.qwen_vlm.protocol import (
    chat_template_kwargs,
    mirror_instruction,
    native_step_budget,
    reset_wait_steps,
)
from rlinf.models.embodiment.qwen_vlm.sglang_client import QwenVLMSGLangClient
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.utils.distributed import ScopedTimer
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


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
    assert native_step_budget("LIBERO_OBJECT") == 280
    assert native_step_budget("libero_goal") == 300
    assert native_step_budget("libero_10") == 520
    with pytest.raises(KeyError):
        native_step_budget("libero_gso")
    assert reset_wait_steps(True) == 10
    assert reset_wait_steps(False) == 15
    assert reset_wait_steps(True, 30) == 30


def test_gso_benchmark_keeps_each_suite_budget():
    class Benchmark:
        def __init__(self, task_order_index=0):
            self.task_order_index = task_order_index

    module = SimpleNamespace(
        BENCHMARK_MAPPING={},
        task_maps={
            "libero_goal": {"g": "goal"},
            "libero_spatial": {"s": "spatial"},
            "libero_object": {"o": "object"},
            "libero_10": {"l": "long"},
        },
    )
    suite_cls = _aggregated_libero_benchmark(
        module,
        Benchmark,
        "libero_gso",
        ("libero_goal", "libero_spatial", "libero_object"),
    )
    suite = suite_cls()
    assert suite.tasks == ["goal", "spatial", "object"]
    assert suite.task_suite_names == [
        "libero_goal",
        "libero_spatial",
        "libero_object",
    ]
    assert [native_step_budget(name) for name in suite.task_suite_names] == [
        300,
        220,
        280,
    ]


def test_one_episode_then_two_optimizer_steps():
    sequence = [10, 1, 2, 11, 3, 4]
    positions = assistant_positions(sequence, [[1, 2], [3, 4]])
    assert positions == [1, 2, 4, 5]
    mask = torch.tensor([[[1.0], [1.0]], [[1.0], [0.0]]])
    kept = keep_last_valid_step(mask)
    assert kept[:, 0, 0].tolist() == [0.0, 1.0]
    assert kept[:, 1, 0].tolist() == [1.0, 0.0]

    logprobs = torch.nn.Parameter(torch.zeros(4))
    advantages = torch.ones(4)
    token_mask = torch.ones(4, dtype=torch.bool)
    optimizer = torch.optim.SGD([logprobs], lr=0.25)
    before = logprobs.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        loss, metrics = pear_reinforce_loss(
            logprobs, advantages, token_mask, old_logprobs=before - 5
        )
        assert float(metrics["actor/clip_fraction"]) == 0.0
        assert float(metrics["actor/ratio"]) == 1.0
        loss.backward()
        optimizer.step()
    assert torch.all(logprobs.detach() > before)


def test_dapo_drops_constant_groups_and_fills_to_target():
    rewards = torch.zeros(1, 4, 1)
    rewards[0, 2, 0] = 1.0
    keep, stats = dapo_group_mask(rewards, group_size=2)
    assert keep.tolist() == [False, False, True, True]
    assert stats["dapo/kept_groups"] == 1
    assert stats["dapo/dropped_groups"] == 1

    batches = iter(
        [
            torch.zeros(2, 2),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        ]
    )
    filled = fill_dapo_groups(2, 2, lambda: next(batches), max_batches=4)
    assert filled.shape == (2, 2)
    assert torch.all(filled.std(dim=-1, unbiased=False) > 0)


def _rollout_geometry(env_world_size: int, actor_world_size: int):
    from omegaconf import OmegaConf

    from rlinf.data.schema.embodied_trajectory import RolloutGeometry
    from rlinf.envs.layout import EnvLayout

    env_cfg = OmegaConf.create(
        {
            "rollout_epoch": 1,
            "max_steps_per_rollout_epoch": 1,
            "max_episode_steps": 1,
            "auto_reset": False,
        }
    )
    return RolloutGeometry(
        layout=EnvLayout(env_cfg, env_world_size, "env.train"),
        stage_num=1,
        num_action_chunks=1,
        actor_world_size=actor_world_size,
    )


def test_trajectory_pool_moves_groups_across_sources():
    from rlinf.algorithms.pear_objective import DapoPending
    from rlinf.data.schema.embodied_trajectory import RolloutOutput
    from rlinf.data.schema.embodied_types import Trajectory

    def trajectory(rewards: list[float], width: int) -> Trajectory:
        batch = len(rewards)
        return Trajectory(
            rewards=torch.tensor(rewards, dtype=torch.float32).view(1, batch, 1),
            prev_logprobs=torch.arange(batch * width, dtype=torch.float32).view(
                1, batch, width
            ),
        )

    class Accumulator:
        def __init__(self, item: Trajectory):
            self.item = item

        def to_trajectory(self) -> Trajectory:
            return self.item

    output = RolloutOutput(
        _Map(
            rollout=_Map(),
            algorithm=_Map(
                group_size=2,
                loss_type="pear_reinforce",
                pear=_Map(dapo_compact=True, dapo_target_groups=2),
            ),
            env=_Map(train=_Map(rollout_epoch=1)),
        ),
        _rollout_geometry(env_world_size=2, actor_world_size=2),
    )
    assert output._expected_accumulator_count == 2
    assert output._scope(SimpleNamespace(step_id=3, env_rank=1, stage_id=0)) == (3,)

    short = list(
        output._flush(
            {
                (0, 0): Accumulator(trajectory([0.0, 0.0, 0.0, 1.0], 1)),
                (1, 0): Accumulator(trajectory([1.0, 1.0], 1)),
            }
        )
    )
    assert len(short) == 2
    assert all(
        isinstance(item, DapoPending) and item.kept_groups == 1 for _, item in short
    )
    assert output._dapo_buffer.rewards[0, :, 0].tolist() == [0.0, 1.0]

    dealt = [
        item
        for _, item in output._flush(
            {
                (0, 0): Accumulator(trajectory([0.0, 1.0], 3)),
                (1, 0): Accumulator(trajectory([1.0, 0.0, 0.0, 0.0], 3)),
            }
        )
    ]
    assert [item.rewards[0, :, 0].tolist() for item in dealt] == [
        [0.0, 1.0],
        [0.0, 1.0],
    ]
    assert dealt[0].prev_logprobs[0, 0].tolist() == [2.0, 0.0, 0.0]
    assert output._dapo_buffer.rewards[0, :, 0].tolist() == [1.0, 0.0]


def test_stream_sends_a_wave_before_the_slow_source_finishes():
    from rlinf.algorithms.pear_objective import TrajectoryDrain
    from rlinf.data.schema.embodied_trajectory import RolloutOutput
    from rlinf.data.schema.embodied_types import Trajectory

    def trajectory(rewards: list[float]) -> Trajectory:
        batch = len(rewards)
        return Trajectory(
            rewards=torch.tensor(rewards, dtype=torch.float32).view(1, batch, 1),
        )

    class Accumulator:
        def __init__(self, item: Trajectory):
            self.item = item

        def to_trajectory(self) -> Trajectory:
            return self.item

    output = RolloutOutput(
        _Map(
            rollout=_Map(),
            algorithm=_Map(
                group_size=2,
                loss_type="pear_reinforce",
                pear=_Map(
                    dapo_compact=True,
                    stream_to_actors=True,
                    dapo_target_groups=4,
                ),
            ),
            env=_Map(train=_Map(rollout_epoch=1)),
        ),
        _rollout_geometry(env_world_size=2, actor_world_size=2),
    )
    early = list(
        output._flush({(0, 0): Accumulator(trajectory([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]))})
    )
    assert [item.rewards[0, :, 0].tolist() for _, item in early] == [
        [0.0, 1.0],
        [1.0, 0.0],
    ]
    assert output._ready_groups == []

    late = list(output._flush({(1, 0): Accumulator(trajectory([0.0, 1.0, 1.0, 1.0]))}))
    assert [type(item) for _, item in late] == [TrajectoryDrain, TrajectoryDrain]
    assert len(output._ready_groups) == 1
    assert output._ready_groups[0].rewards[0, :, 0].tolist() == [0.0, 1.0]


class _Map(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def test_actor_keeps_one_sequence_and_masks_zero_variance_groups():
    actor = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    actor.cfg = SimpleNamespace(
        algorithm=_Map(
            group_size=2,
            pear=_Map(episode_sequence=True, dapo_filter=True),
        )
    )
    actor._dapo_stats = {}
    batch = {
        "loss_mask": torch.ones(2, 4, 1),
        "rewards": torch.tensor(
            [
                [[0.0], [0.0], [0.0], [1.0]],
                [[0.0], [0.0], [0.0], [0.0]],
            ]
        ),
    }
    actor._apply_pear_objective(batch)
    assert batch["loss_mask"][:, 0, 0].tolist() == [0.0, 0.0]
    assert batch["loss_mask"][:, 1, 0].tolist() == [0.0, 0.0]
    assert batch["loss_mask"][:, 2, 0].tolist() == [0.0, 1.0]
    assert batch["loss_mask"][:, 3, 0].tolist() == [0.0, 1.0]
    assert actor._dapo_stats["dapo/kept_groups"] == 1
    assert float(batch["loss_mask_sum"][0, 2, 0]) == 1.0


class _Handle:
    def __init__(self, result=None):
        self.result = result
        self.waits = 0

    def wait(self):
        self.waits += 1
        return self.result


def test_scoped_timer_sums_a_repeated_section(monkeypatch):
    clock = {"t": 100.0}

    def fake_time():
        clock["t"] += 2.0
        return clock["t"]

    monkeypatch.setattr("rlinf.utils.timers.time.time", fake_time)
    timer = ScopedTimer(reduction="max", sync_cuda=False, trace=False)
    with timer("generate_rollouts"):
        pass
    first = timer.take_duration("generate_rollouts")
    with timer("generate_rollouts"):
        pass
    timer.add_duration("generate_rollouts", first)
    assert timer.consume_durations()["generate_rollouts"] == pytest.approx(4.0)
    with timer("generate_rollouts"):
        pass
    with pytest.raises(ValueError, match="generate_rollouts"):
        with timer("generate_rollouts"):
            pass


def test_runner_resamples_until_a_variable_group():
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = SimpleNamespace(
        algorithm=_Map(
            pear=_Map(
                dapo_resample=True,
                dapo_target_groups=1,
                dapo_max_batches=3,
            )
        )
    )
    runner.reward = None
    runner._timers = []
    runner.timer = ScopedTimer(reduction="max", sync_cuda=False, trace=False)
    env_handles = []
    metrics = iter([[{"dapo/kept_groups": 0.0}], [{"dapo/kept_groups": 1.0}]])

    class Env:
        def interact(self, **kwargs):
            handle = _Handle([])
            env_handles.append(handle)
            return handle

    class Rollout:
        def generate(self, **kwargs):
            return _Handle()

    class Actor:
        def recv_rollout_trajectories(self, **kwargs):
            return _Handle()

        def compute_advantages_and_returns(self):
            return _Handle(next(metrics))

    runner.env = Env()
    runner.rollout = Rollout()
    runner.actor = Actor()
    runner.env_channel = None
    runner.rollout_channel = None
    runner.reward_channel = None
    runner.actor_channel = None
    _, _, _, result = runner._rollout_until_variable_groups()
    assert len(env_handles) == 2
    assert env_handles[0].waits == 1
    assert env_handles[1].waits == 0
    assert result[0]["dapo/sampled_batches"] == 2
    assert result[0]["dapo/kept_groups"] == 1.0
    durations = runner.timer.consume_durations()
    assert set(durations) == {"generate_rollouts", "cal_adv_and_returns"}


def test_runner_treats_zero_batch_cap_as_unlimited():
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = SimpleNamespace(
        algorithm=_Map(
            pear=_Map(
                dapo_resample=True,
                dapo_target_groups=1,
                dapo_max_batches=0,
            )
        )
    )
    runner.reward = None
    runner._timers = []
    runner.timer = ScopedTimer(reduction="max", sync_cuda=False, trace=False)
    calls = {"n": 0}
    metrics = iter([[{"dapo/kept_groups": 0.0}], [{"dapo/kept_groups": 1.0}]])

    class Env:
        def interact(self, **kwargs):
            calls["n"] += 1
            return _Handle([])

    class Rollout:
        def generate(self, **kwargs):
            return _Handle()

    class Actor:
        def recv_rollout_trajectories(self, **kwargs):
            return _Handle()

        def compute_advantages_and_returns(self):
            return _Handle(next(metrics))

    runner.env = Env()
    runner.rollout = Rollout()
    runner.actor = Actor()
    runner.env_channel = None
    runner.rollout_channel = None
    runner.reward_channel = None
    runner.actor_channel = None
    runner._rollout_until_variable_groups()
    assert calls["n"] == 2


class _FakeSGLangSessionServer:
    """In-process fake matching SGLang's SessionController / Session.create_req contract."""

    def __init__(self, reply_text: str = "ANS", reply_ids: list[int] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self._sessions: dict[str, dict] = {}
        self._next_id = 0
        self._reply_text = reply_text
        self._reply_ids = list(reply_ids) if reply_ids is not None else [7, 8]

    def post(self, path: str, payload: dict, **kwargs) -> dict | str:
        del kwargs
        self.calls.append((path, payload))
        if path == "/open_session":
            self._next_id += 1
            sid = f"s{self._next_id}"
            self._sessions[sid] = {
                "streaming": bool(payload.get("streaming", False)),
                "req_nodes": {},
            }
            return {"session_id": sid}
        if path == "/close_session":
            self._sessions.pop(payload["session_id"], None)
            return {}
        if path == "/generate":
            session_params = payload.get("session_params") or {}
            sid = session_params.get("id")
            session = self._sessions[sid]
            last_req = None
            if session["streaming"]:
                if session["req_nodes"]:
                    last_req = next(iter(session["req_nodes"].values()))
            elif session_params.get("rid") is not None:
                last_req = session["req_nodes"].get(session_params["rid"])

            rid = payload["rid"]
            if not session["streaming"] and rid in session["req_nodes"]:
                raise AssertionError(f"Duplicate non-streaming rid: {rid}")

            delta_tokens = len(str(payload["text"])) + 4 * len(
                payload.get("image_data") or []
            )
            cached_tokens = (
                last_req["prompt_tokens"] + len(last_req["output_ids"])
                if last_req is not None
                else 0
            )
            prompt_tokens = cached_tokens + delta_tokens
            record = {
                "rid": rid,
                "prompt_tokens": prompt_tokens,
                "output_ids": list(self._reply_ids),
            }
            if session["streaming"]:
                session["req_nodes"] = {rid: record}
            else:
                session["req_nodes"][rid] = record
            return {
                "text": self._reply_text,
                "output_ids": list(self._reply_ids),
                "meta_info": {
                    "prompt_tokens": prompt_tokens,
                    "cached_tokens": cached_tokens,
                    "completion_tokens": len(self._reply_ids),
                },
            }
        return {}


def test_sglang_reuses_prefix_until_weights_change():
    server = _FakeSGLangSessionServer(reply_text="ANS", reply_ids=[7, 8])
    client = QwenVLMSGLangClient(reasoning_effort="xhigh", transport=server.post)
    first = client.generate(0, full_text="PROMPT", max_new_tokens=8, images=["first"])
    assert server.calls[0][0] == "/open_session"
    assert server.calls[0][1] == {"capacity_of_str_len": 1_000_000, "streaming": True}
    second = client.generate(
        0, full_text="PROMPTANS+USER", max_new_tokens=8, images=["second"]
    )
    generated = [payload for path, payload in server.calls if path == "/generate"]
    assert generated[0]["text"] == "PROMPT"
    assert generated[1]["text"] == "+USER"
    assert generated[0]["session_params"]["id"] == "s1"
    assert generated[1]["session_params"]["id"] == "s1"
    assert generated[0]["rid"] == "qwen-env-0-v0-t1"
    assert generated[1]["rid"] == "qwen-env-0-v0-t2"
    assert generated[1]["image_data"] == ["second"]
    assert generated[1]["sampling_params"] == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": 8,
    }
    assert first["meta_info"]["cached_tokens"] == 0
    assert second["meta_info"]["cached_tokens"] == first["meta_info"][
        "prompt_tokens"
    ] + len(first["output_ids"])
    assert second["meta_info"]["prompt_tokens"] > first["meta_info"]["prompt_tokens"]

    class Engine:
        def __init__(self):
            self.parameters = None

        def update_weights_from_tensor(self, parameters):
            self.parameters = parameters

    engine = Engine()
    client.engine = engine
    client.sync_weights(2, [("weight", torch.ones(1))])
    assert any(path == "/close_session" for path, _ in server.calls)
    assert any(path == "/flush_cache" for path, _ in server.calls)
    assert engine.parameters[0][0] == "weight"
    after_sync = client.generate(0, full_text="PROMPTANS+USER", max_new_tokens=8)
    follow = [payload for path, payload in server.calls if path == "/generate"][-1]
    assert follow["text"] == "PROMPTANS+USER"
    assert follow["session_params"]["id"] == "s2"
    assert follow["rid"] == "qwen-env-0-v2-t3"
    assert after_sync["meta_info"]["cached_tokens"] == 0
    client.release_gpu_memory()
    assert any(path == "/release_memory_occupation" for path, _ in server.calls)
    with pytest.raises(RuntimeError, match="memory is released"):
        client.generate(0, full_text="AFTER", max_new_tokens=4)
    client.resume_gpu_memory()
    assert any(path == "/resume_memory_occupation" for path, _ in server.calls)
    client.generate(0, full_text="AFTER", max_new_tokens=4)
    restored = [payload for path, payload in server.calls if path == "/generate"][-1]
    assert restored["text"] == "AFTER"
    assert restored["rid"] == "qwen-env-0-v2-t4"


def test_qwen_vlm_adapter_streaming_session_accumulates_prompt_and_cached_tokens():
    from rlinf.models.embodiment.qwen_vlm.sglang_adapter import QwenVLMSGLangAdapter

    action_reply = (
        "reason</think>\n```json\n"
        '{"arm": {"x": 0.1, "y": 0.0, "z": 0.0, '
        '"roll": 0.0, "pitch": 0.0, "yaw": 0.0}, '
        '"gripper": {"width": 1.0}}\n```'
    )
    server = _FakeSGLangSessionServer(reply_text=action_reply, reply_ids=[101, 102])

    class _FakeProcessor:
        eos_token = "<|im_end|>"
        eos_token_id = 999

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=False, **kwargs
        ):
            del tokenize, kwargs
            parts = []
            for msg in messages:
                content = msg["content"]
                if isinstance(content, list):
                    rendered = "".join(
                        b["text"] if b["type"] == "text" else "<image>" for b in content
                    )
                else:
                    rendered = str(content)
                parts.append(f"<|im_start|>{msg['role']}\n{rendered}<|im_end|>\n")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            return "".join(parts)

        def __call__(self, text, images, return_tensors="pt", padding=True, **kwargs):
            del return_tensors, padding, kwargs
            n_tokens = max(len(text[0]) // 8, 2)
            return {
                "input_ids": torch.arange(1, n_tokens + 1, dtype=torch.long).unsqueeze(
                    0
                ),
                "pixel_values": torch.ones((len(images), 4), dtype=torch.float32),
                "image_grid_thw": torch.ones((len(images), 3), dtype=torch.long),
            }

    cfg = _Map(
        rollout=_Map(
            model=_Map(model_path="dummy", embodiment="libero"),
            sampling_params=_Map(max_new_tokens=512),
            sglang=_Map(response_budget=512, reasoning_effort="xhigh"),
        ),
        algorithm=_Map(pear=_Map(score_centering=False)),
        env=_Map(train=_Map(vlm_action_mode=True)),
    )
    adapter = QwenVLMSGLangAdapter(cfg, rank=0)
    adapter._processor = _FakeProcessor()
    adapter.bind_http_client(server)

    def _obs(step: int, *, reset: bool = False) -> dict:
        return {
            "_env_index": 0,
            "reset_mask": torch.tensor([reset]),
            "task_descriptions": ["pick up the mug"],
            "main_images": torch.full((1, 4, 4, 3), step, dtype=torch.uint8),
            "wrist_images": torch.full((1, 4, 4, 3), step + 10, dtype=torch.uint8),
            "vlm_state": torch.tensor([[float(step), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        }

    meta_by_turn = []
    payloads = []
    for step in range(3):
        payload, state = adapter.build_request(_obs(step))
        response = server.post("/generate", payload)
        adapter.parse_response(response, state)
        payloads.append(payload)
        meta_by_turn.append(response["meta_info"])

    open_calls = [body for path, body in server.calls if path == "/open_session"]
    assert open_calls == [{"capacity_of_str_len": 1_000_000, "streaming": True}]
    assert [p["rid"] for p in payloads] == [
        "qwen-r0-env-0-v0-t1",
        "qwen-r0-env-0-v0-t2",
        "qwen-r0-env-0-v0-t3",
    ]
    assert {p["session_params"]["id"] for p in payloads} == {"s1"}
    assert [len(p["image_data"]) for p in payloads] == [2, 2, 2]
    assert meta_by_turn[0]["cached_tokens"] == 0
    assert meta_by_turn[1]["cached_tokens"] > 0
    assert meta_by_turn[2]["cached_tokens"] > meta_by_turn[1]["cached_tokens"]
    assert (
        meta_by_turn[0]["prompt_tokens"]
        < meta_by_turn[1]["prompt_tokens"]
        < meta_by_turn[2]["prompt_tokens"]
    )

    # Resetting the episode closes the old session and opens a fresh streaming one.
    reset_payload, reset_state = adapter.build_request(_obs(0, reset=True))
    reset_resp = server.post("/generate", reset_payload)
    adapter.parse_response(reset_resp, reset_state)
    assert reset_payload["session_params"]["id"] == "s2"
    assert reset_payload["rid"] == "qwen-r0-env-0-v0-t4"
    assert reset_resp["meta_info"]["cached_tokens"] == 0


def test_session_turn_sends_suffix_until_the_prefix_changes():
    from rlinf.models.embodiment.qwen_vlm.sglang_adapter import session_turn

    suffix, continuation = session_turn("PROMPT", "PROMPT+TURN", True)
    assert continuation is True
    assert suffix == "+TURN"
    full, continuation = session_turn("PROMPT", "REWRITTEN", True)
    assert continuation is False
    assert full == "REWRITTEN"
    full, continuation = session_turn("PROMPT", "PROMPT+TURN", False)
    assert continuation is False
    assert full == "PROMPT+TURN"


def test_qwen_vlm_adapter_is_registered_and_merges_turns():
    from rlinf.models.embodiment.qwen_vlm.sglang_adapter import QwenVLMSGLangAdapter
    from rlinf.models.embodiment.sglang_adapter import get_sglang_adapter_cls

    assert get_sglang_adapter_cls("qwen_vlm") is QwenVLMSGLangAdapter
    adapter = QwenVLMSGLangAdapter.__new__(QwenVLMSGLangAdapter)
    adapter.max_new_tokens = 4

    def sample(length: int, pixels: int) -> dict:
        return {
            "input_ids": torch.ones(length, dtype=torch.long),
            "attention_mask": torch.ones(length, dtype=torch.long),
            "pixel_values": torch.ones(pixels, 2),
            "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
            "response_logprobs": torch.zeros(2),
            "response_index": torch.tensor([length - 2, length - 1]),
        }

    action = torch.zeros(1, 1, 8)
    actions, info = adapter.merge_env_results(
        [
            (action, {"sample": sample(5, 2)}),
            (action, {"sample": sample(7, 3)}),
        ]
    )
    assert actions.shape == (2, 1, 8)
    assert info["forward_inputs"]["input_ids"].shape == (2, 7)
    assert info["forward_inputs"]["response_mask"].shape[0] == 2
    assert info["prev_values"] is None


def test_chunked_head_matches_the_full_logits_and_spans_chunks():
    from rlinf.models.embodiment.qwen_vlm.fp32_head import (
        Fp32HeadLogProbs,
        response_logprobs_from_hidden,
    )

    torch.manual_seed(0)
    device = "cuda"
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


def test_turn_token_budget_is_what_the_episode_has_left():
    from rlinf.models.embodiment.qwen_vlm.sglang_adapter import QwenVLMSGLangAdapter

    adapter = QwenVLMSGLangAdapter.__new__(QwenVLMSGLangAdapter)
    adapter._response_budget = 10
    adapter._response_ids = {0: [[1, 2, 3], [4, 5]]}
    assert adapter._remaining_response_tokens(0) == 5
    adapter._response_ids[0].append([9, 9, 9, 9, 9])
    assert adapter._remaining_response_tokens(0) == 0
    assert adapter._remaining_response_tokens(1) == 10


def test_finished_episode_does_not_call_sglang():
    from rlinf.workers.rollout.sglang.sglang_embodied_worker import SGLangEmbodiedWorker

    posted: list[int] = []

    class Adapter:
        action_path = "/generate"
        use_msgpack = False

        @staticmethod
        def env_is_finished(obs):
            return bool(obs["vlm_finished"][0].item())

        @staticmethod
        def finished_result(obs):
            index = int(obs["_env_index"])
            return torch.zeros(1, 1, 8), {"sample": {"env": index, "idle": True}}

        @staticmethod
        def build_request(obs, mode="train"):
            del mode
            index = int(obs["_env_index"])
            return {"i": index}, {"i": index}

        @staticmethod
        def parse_response(response, state):
            del response
            return torch.ones(1, 1, 8), {"sample": {"env": state["i"], "idle": False}}

    class Client:
        @staticmethod
        def post(path, payload, **kwargs):
            del path, kwargs
            posted.append(int(payload["i"]))
            return payload

    worker = SGLangEmbodiedWorker.__new__(SGLangEmbodiedWorker)
    worker.sglang_adapter = Adapter()
    worker.http_client = Client()
    worker._http_timeout_s = 1.0
    worker._http_max_retries = 0
    worker._http_retry_backoff_s = 0.0
    worker._infer_env_batch_size = lambda obs: int(obs["vlm_finished"].shape[0])
    outputs = worker._request_per_env(
        {"vlm_finished": torch.tensor([True, False, True])},
        "train",
    )
    assert posted == [1]
    assert [item[1]["sample"]["idle"] for item in outputs] == [True, False, True]
    assert [item[1]["sample"]["env"] for item in outputs] == [0, 1, 2]


def test_collocated_ipc_rank_path_and_two_step_resume():
    from rlinf.workers.rollout.sglang.ipc_placement import CollocatedIpcPlacement
    from rlinf.workers.rollout.sglang.sglang_embodied_worker import next_sglang_resume
    from rlinf.workers.rollout.utils import (
        RankMapper,
        instance_ids_to_entry_process_ranks,
    )

    placement = CollocatedIpcPlacement(8, 8)
    mapped = RankMapper.get_actor_rank_to_rollout_rank_map(placement)
    assert list(instance_ids_to_entry_process_ranks(mapped[3], placement)) == [0, 3]
    inverse = RankMapper.get_rollout_rank_to_actor_rank_map(placement)
    assert inverse[(0, 3)] == 3

    # 32 actor GPUs, TP 8: four engines. Rank 8 is engine 1, TP rank 0.
    wide = CollocatedIpcPlacement(32, 8)
    wide_map = RankMapper.get_actor_rank_to_rollout_rank_map(wide)
    assert list(instance_ids_to_entry_process_ranks(wide_map[8], wide)) == [1, 0]
    assert list(instance_ids_to_entry_process_ranks(wide_map[31], wide)) == [3, 7]
    assert RankMapper.get_rollout_rank_to_actor_rank_map(wide)[(1, 0)] == 8

    # TP 16 across two 8-GPU nodes: the entry of engine 1 is process 2.
    spanned = CollocatedIpcPlacement(32, 16, nnodes_per_model_instance=2)
    spanned_map = RankMapper.get_actor_rank_to_rollout_rank_map(spanned)
    assert list(instance_ids_to_entry_process_ranks(spanned_map[8], spanned)) == [0, 8]
    assert list(instance_ids_to_entry_process_ranks(spanned_map[16], spanned)) == [2, 0]
    assert spanned.rollout_model_instance_id(3) == 1
    assert spanned.rollout_node_rank_in_model_instance(3) == 1

    with pytest.raises(ValueError):
        CollocatedIpcPlacement(32, 12)

    from rlinf.utils.placement import _derive_rollout_node_layout

    four_nodes = [list(range(node * 8, node * 8 + 8)) for node in range(4)]
    on_node = _derive_rollout_node_layout(four_nodes, 8)
    assert on_node.nnodes_per_model_instance == 1
    assert on_node.gpus_per_process == 8
    across = _derive_rollout_node_layout(four_nodes, 16)
    assert across.nnodes_per_model_instance == 2
    assert across.gpus_per_process == 8
    assert next_sglang_resume(None) == (None, None)
    assert next_sglang_resume("all") == (["weights"], "kv")
    assert next_sglang_resume("kv") == (["kv_cache", "cuda_graph"], None)


def test_env_output_preserves_vlm_state_and_rounds_fp32_drift():
    from rlinf.data.schema.embodied_types import EnvOutput
    from rlinf.models.embodiment.qwen_vlm.policy import _state_from_observation

    raw_state = torch.tensor(
        [[35.1, -2.3, 24.6, 3.14, -0.0, 0.25, 7.9]], dtype=torch.float32
    )
    env_output = EnvOutput(
        obs={
            "main_images": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
            "wrist_images": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
            "states": torch.full((1, 8), -99.0),
            "vlm_state": raw_state,
            "vlm_finished": torch.tensor([False]),
            "reset_mask": torch.tensor([True]),
            "task_descriptions": ["pick up the bowl"],
        }
    )
    prepared = env_output.prepare_observations(env_output.obs)
    assert "vlm_state" in prepared
    parsed = _state_from_observation(prepared, 0, require_vlm_state=True)
    assert parsed == {
        "arm": {
            "x": 35.1,
            "y": -2.3,
            "z": 24.6,
            "roll": 3.14,
            "pitch": 0.0,
            "yaw": 0.25,
        },
        "gripper": {"width": 7.9},
    }
    prepared_missing = {k: v for k, v in prepared.items() if k != "vlm_state"}
    with pytest.raises(KeyError, match="vlm_state"):
        _state_from_observation(prepared_missing, 0, require_vlm_state=True)


def test_user_content_includes_camera_labels_for_libero_and_robodojo():
    from rlinf.models.embodiment.qwen_vlm.protocol import (
        LIBERO_SCHEMA,
        ROBODOJO_SCHEMA,
        build_user_content,
        cameras_for_schema,
    )

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


def test_compact_episode_sequence_batch_and_global_token_mean():
    from rlinf.algorithms.pear_objective import (
        compact_episode_sequence_batch,
        count_valid_sequence_tokens,
    )

    # T=3 turns, B=2 envs: env 0 ends at t=2 (reward at t=2), env 1 ends at t=0 (reward at t=0).
    batch = {
        "loss_mask": torch.tensor([[[0.0], [1.0]], [[0.0], [0.0]], [[1.0], [0.0]]]),
        "loss_mask_sum": torch.ones(3, 2, 1),
        "rewards": torch.tensor([[[0.0], [1.0]], [[0.0], [0.0]], [[1.0], [0.0]]]),
        "dones": torch.tensor(
            [[[0.0], [0.0]], [[0.0], [1.0]], [[0.0], [1.0]], [[1.0], [1.0]]]
        ),
        "prev_logprobs": torch.arange(3 * 2 * 4, dtype=torch.float32).view(3, 2, 4),
        "forward_inputs": {
            "input_ids": torch.arange(3 * 2 * 5).view(3, 2, 5),
            "response_mask": torch.tensor(
                [
                    [[True, False, False, False], [True, True, False, False]],
                    [[True, True, False, False], [False, False, False, False]],
                    [[True, True, True, True], [False, False, False, False]],
                ]
            ),
        },
    }
    compacted = compact_episode_sequence_batch(batch)
    assert compacted["prev_logprobs"].shape == (1, 2, 4)
    assert compacted["dones"].shape == (2, 2, 1)
    assert compacted["rewards"][0, :, 0].tolist() == [1.0, 1.0]
    torch.testing.assert_close(
        compacted["forward_inputs"]["input_ids"][0, 0],
        batch["forward_inputs"]["input_ids"][2, 0],
    )
    torch.testing.assert_close(
        compacted["forward_inputs"]["input_ids"][0, 1],
        batch["forward_inputs"]["input_ids"][0, 1],
    )

    # Global token-mean equivalence across variable-length micro-batches (4 tokens vs 2 tokens).
    flat = {
        "loss_mask": compacted["loss_mask"].reshape(-1, 1),
        "forward_inputs": {
            "response_mask": compacted["forward_inputs"]["response_mask"].reshape(-1, 4)
        },
    }
    total_tokens = count_valid_sequence_tokens(flat)
    assert total_tokens == 6.0

    lp = torch.tensor(
        [[-0.5, -1.0, -1.5, -2.0], [-0.2, -0.4, 0.0, 0.0]], requires_grad=True
    )
    adv = torch.tensor([[1.0], [-1.0]]).expand_as(lp)
    mask = flat["forward_inputs"]["response_mask"]
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
        lp, ent, head_lp = response_logprobs_and_heads_from_hidden(
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
    from rlinf.models.embodiment.qwen_vlm.conversation import IncrementalEpisodeTokens
    from rlinf.models.embodiment.qwen_vlm.sglang_adapter import (
        extract_sglang_topk_logprobs,
    )

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

    token_lps, topk_ids, topk_lps = extract_sglang_topk_logprobs(
        {
            "meta_info": {
                "output_token_logprobs": [(-0.2, 20), (-0.3, 21)],
                "output_top_logprobs": [
                    [(-0.2, 20), (-1.5, 22)],
                    [(-0.3, 21), (-1.2, 23)],
                ],
            }
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

    # Observation turn (4 tokens) consumes budget with mask=0.
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

    # Next observation (3 tokens) exceeds remaining budget (2): rejected cleanly.
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


def test_dapo_staleness_and_clear_surplus():
    from rlinf.algorithms.pear_objective import is_trajectory_stale
    from rlinf.data.schema.embodied_trajectory import RolloutOutput
    from rlinf.data.schema.embodied_types import Trajectory

    stale_traj = Trajectory(
        rewards=torch.tensor([[[0.0], [1.0]]]),
        versions=torch.tensor([[[1.0], [1.0]]]),
    )
    fresh_traj = Trajectory(
        rewards=torch.tensor([[[0.0], [1.0]]]),
        versions=torch.tensor([[[6.0], [6.0]]]),
    )
    assert is_trajectory_stale(
        stale_traj, current_version=6, max_off_policy_threshold=5
    )
    assert not is_trajectory_stale(
        fresh_traj, current_version=6, max_off_policy_threshold=5
    )

    class Accumulator:
        def __init__(self, item: Trajectory):
            self.item = item

        def to_trajectory(self) -> Trajectory:
            return self.item

    output = RolloutOutput(
        _Map(
            rollout=_Map(),
            algorithm=_Map(
                group_size=2,
                loss_type="pear_reinforce",
                pear=_Map(
                    dapo_compact=True,
                    dapo_target_groups=1,
                    clear_surplus_after_step=True,
                ),
            ),
            env=_Map(train=_Map(rollout_epoch=1)),
        ),
        _rollout_geometry(env_world_size=1, actor_world_size=1),
    )
    two_var_groups = Trajectory(
        rewards=torch.tensor([[[0.0], [1.0], [1.0], [0.0]]]),
        versions=torch.tensor([[[2.0], [2.0], [2.0], [2.0]]]),
    )
    emitted = list(output._flush({(0, 0): Accumulator(two_var_groups)}))
    assert len(emitted) == 1
    assert output._dapo_buffer is None


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


def test_gso_sglang_configs_sample_all_tasks_and_skip_recompute():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"
    )
    shared = OmegaConf.load(config_dir / "hybrid_engines" / "sglang_router_server.yaml")
    assert shared.server.attention_backend == "triton"
    for name in (
        "libero_gso_grpo_pear_sglang",
        "libero_gso_grpo_pear_sglang_paper",
    ):
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name=name)
        assert cfg.env.train.use_fixed_reset_state_ids is False
        assert cfg.env.train.specific_reset_id is None
        assert cfg.algorithm.recompute_logprobs is False
        assert cfg.rollout.sglang.server.attention_backend is None
        assert cfg.rollout.sglang.server.max_mamba_cache_size == 256
