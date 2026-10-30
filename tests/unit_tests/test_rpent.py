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

"""RPent: append-only prompts, the episode-server protocol, retries and config."""

import asyncio
import json
import logging
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from _qwen35_chat import Qwen35ToyTokenizer, qwen35_agentic_tokenizer
from omegaconf import OmegaConf

from rlinf.agents.rpent.episode import RPentEpisodeError, merge_prefix_turns
from rlinf.agents.rpent.history import (
    AppendOnlyHistory,
    render_prompt,
    template_messages,
)
from rlinf.agents.rpent.rpent_agent_loop import RPentAgentLoopWorker
from rlinf.workers.agent.agent_loop import AgentLoopOutput

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "look",
            "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
    }
]


def test_extract_original_checkpoint_excludes_vision_and_mtp(tmp_path):
    import importlib.util

    import torch
    from safetensors.torch import load_file, save_file

    # Load by path: Megatron-LM ships its own ``examples`` package.
    path = Path(__file__).parents[2] / "examples/agent/rpent/checkpoint.py"
    spec = importlib.util.spec_from_file_location("rpent_checkpoint", path)
    checkpoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checkpoint)

    source = tmp_path / "vlm"
    source.mkdir()
    language = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    save_file(
        {
            "model.language_model.embed_tokens.weight": language,
            "model.visual.patch_embed.weight": torch.ones(2, 3),
            "mtp.layers.0.mlp.down_proj.weight": torch.zeros(2, 3),
        },
        source / "model.safetensors",
    )
    (source / "config.json").write_text(
        json.dumps({"text_config": {"model_type": "qwen3_5_text"}})
    )

    output = tmp_path / "text"
    checkpoint.extract(source, output)

    tensors = load_file(output / "model.safetensors")
    assert set(tensors) == {"model.embed_tokens.weight"}
    assert torch.equal(tensors["model.embed_tokens.weight"], language)
    assert json.loads((output / "config.json").read_text())["architectures"] == [
        "Qwen3_5ForCausalLM"
    ]


@pytest.fixture
def qwen35(monkeypatch):
    return qwen35_agentic_tokenizer(monkeypatch)


def tool_reply(turn: int) -> dict:
    call = {"id": "c", "type": "function", "function": {"name": "look"}}
    call["function"]["arguments"] = '{"x": 1}'
    return {"role": "assistant", "content": f"step {turn}", "tool_calls": [call]}


def sample_turns(history, tokenizer, turns=3):
    """Record tool-call turns; return each turn's prompt, response and request."""
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
    ]
    sampled = []
    for turn in range(turns):
        request = {"messages": [dict(m) for m in messages], "tools": TOOLS}
        prompt = history.render(request)
        # One newline before the call, where the template writes two.
        response = tokenizer(
            f"step {turn}\n<tool_call>\n<function=look>\n<parameter=x>\n1\n"
            "</parameter>\n</function>\n</tool_call><|im_end|>"
        )["input_ids"]
        history.record(request, prompt, response, tool_reply(turn))
        sampled.append((prompt, response, request))
        messages += [
            tool_reply(turn),
            {"role": "tool", "tool_call_id": "c", "content": "seen"},
            {"role": "user", "content": f"observation {turn}"},
        ]
    return sampled, messages


def test_prompts_append_new_messages_to_the_sampled_tokens(qwen35):
    sampled, _ = sample_turns(AppendOnlyHistory(qwen35), qwen35.tokenizer)
    assert sampled[0][0] == render_prompt(qwen35, sampled[0][2])
    for (prompt, response, _), (following, _, request) in zip(sampled, sampled[1:]):
        assert following[: len(prompt) + len(response)] == prompt + response
        # The new messages are what the template writes after the last reply.
        full = qwen35.tokenizer.decode(render_prompt(qwen35, request))
        appended = full.rsplit("</tool_call><|im_end|>", 1)[1]
        start = len(prompt) + len(response)
        assert qwen35.tokenizer.decode(following[start:]) == appended
        # A full render rewrites the sampled replies, which training must not see.
        assert following != render_prompt(qwen35, request)


def test_rewritten_histories_are_rendered_in_full(qwen35):
    history = AppendOnlyHistory(qwen35)
    _, messages = sample_turns(history, qwen35.tokenizer)
    # The next request must contain the assistant reply actually returned.
    changed = {"messages": [dict(m) for m in messages], "tools": TOOLS}
    changed["messages"][-3] = {"role": "assistant", "content": "different reply"}
    assert history.render(changed) == render_prompt(qwen35, changed)
    # A compacted history starts a new prefix chain.
    compacted = {"messages": messages[:2] + messages[-1:], "tools": TOOLS}
    assert history.render(compacted) == render_prompt(qwen35, compacted)
    other_tools = {"messages": messages, "tools": []}
    assert history.render(other_tools) == render_prompt(qwen35, other_tools)


def test_tool_calls_with_non_object_arguments_stay_visible_as_text():
    valid, invalid = tool_reply(0), tool_reply(1)
    invalid["tool_calls"][0]["function"]["arguments"] = "{not json"
    rendered = template_messages([valid, invalid])
    assert rendered[0]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
    assert rendered[1]["tool_calls"] == []
    assert "{not json" in rendered[1]["content"]
    with pytest.raises(ValueError, match="string message content"):
        template_messages([{"role": "user", "content": [{"type": "image"}]}])


def test_merge_prefix_turns_masks_tool_tokens_and_splits_at_rewrites():
    def turn(turn_id, prompt, response, reward=1.0):
        return AgentLoopOutput(
            prompt_ids=prompt,
            response_ids=response,
            response_logprobs=[-float(token) for token in response],
            is_end=True,
            reward_score=reward,
            extra_fields={"turn_id": turn_id, "tool_names": [f"tool{turn_id}"]},
        )

    turns = [
        turn(0, [1, 2], [3, 4]),
        # Appends a tool result (9, 8) to the previous turn's tokens.
        turn(1, [1, 2, 3, 4, 9, 8], [5, 6]),
        # Compaction rewrote the history: a new row starts here.
        turn(2, [1, 2, 7], [8]),
        turn(3, [1, 2, 7, 8, 10], [11], reward=0.0),
    ]
    first, second = merge_prefix_turns(turns, max_scored_tokens=8)
    assert first.prompt_ids == [1, 2]
    assert first.response_ids == [3, 4, 9, 8, 5, 6]
    assert first.response_mask == [1, 1, 0, 0, 1, 1]
    assert first.response_logprobs == [-3.0, -4.0, 0.0, 0.0, -5.0, -6.0]
    assert first.extra_fields["turn_ids"] == [0, 1]
    assert first.extra_fields["tool_names"] == ["tool1"]
    assert second.response_ids == [8, 10, 11] and second.response_mask == [1, 0, 1]
    assert second.reward_score == 0.0 and second.extra_fields["turn_ids"] == [2, 3]
    # Inputs are not modified, and the scored-token budget splits a row.
    assert turns[0].response_mask is None and turns[0].response_ids == [3, 4]
    budgeted = merge_prefix_turns(turns, max_scored_tokens=3)
    assert [row.extra_fields["turn_ids"] for row in budgeted] == [[0], [1], [2, 3]]
    assert budgeted[1].prompt_ids == [1, 2, 3, 4, 9, 8]


# Stands in for RPent's ``robots.libero.rl_episode``: it plans against the
# worker's policy endpoint and writes the outcome, as the real process does.
FAKE_EPISODE = """
import argparse, json, os, signal, sys, time, urllib.error, urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--request", type=Path)
args, _ = parser.parse_known_args()
script = json.loads(os.environ["FAKE_RPENT"])
state = Path(script["state_dir"])
request = json.loads(args.request.read_text())
call = {"argv": sys.argv[1:], "request": request, "pid": os.getpid()}
with (state / "calls.jsonl").open("a") as calls:
    calls.write(json.dumps(call) + "\\n")
failure = script["failure"] if len((state / "calls.jsonl").read_text().splitlines()) == 1 else None
if failure == "env_setup":
    sys.exit(75)
if failure == "crash":
    sys.exit(1)
if failure == "killed":
    os.kill(os.getpid(), signal.SIGKILL)
if failure == "hang":
    time.sleep(600)
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
planner = request["planner"]
messages, status = [{"role": "user", "content": "task"}], 200
for _ in range(script["turns"]):
    body = {"model": planner["model_name"], "messages": messages, "tools": script["tools"]}
    post = urllib.request.Request(
        planner["base_url"] + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(post, timeout=30) as response:
            message = json.loads(response.read())["choices"][0]["message"]
    except urllib.error.HTTPError as error:
        status = error.code
        with (state / "rejections.jsonl").open("a") as rejections:
            rejections.write(json.loads(error.read())["error"]["code"] + "\\n")
        break
    tool_call = message["tool_calls"][0]["id"]
    messages += [message, {"role": "tool", "tool_call_id": tool_call, "content": "seen"}]
error_kind = {200: None, 400: "context_limit"}.get(status, "backend")
if error_kind is None and failure in ("model_transport", "backend"):
    error_kind = failure
outcome = {
    "success": script["success"] and error_kind is None,
    "truncated": False,
    "terminal": error_kind in (None, "context_limit"),
    "termination_reason": error_kind or "planner_completed",
    "error": "planner failed" if error_kind else None,
    "error_kind": error_kind,
    "init_state_id": 3,
    "memory_entries": [],
}
(args.output_dir / "outcome.json").write_text(json.dumps(outcome))
"""


class FakeRPent:
    """An episode process whose planner calls the policy endpoint ``turns`` times.

    ``failure`` hits the first episode only; a retry runs a fresh process.
    """

    def __init__(self, *, turns=2, success=True, failure=None):
        self.turns, self.success, self.failure = turns, success, failure
        self.state_dir: Path | None = None

    def lines(self, name: str) -> list[str]:
        path = self.state_dir / name
        return path.read_text().splitlines() if path.exists() else []

    @property
    def calls(self) -> list[dict]:
        return [json.loads(line) for line in self.lines("calls.jsonl")]

    @property
    def rejections(self) -> list[str]:
        return self.lines("rejections.jsonl")


@pytest.fixture
def make_agent(tmp_path, monkeypatch):
    """Build an agent whose episode process and sampler are fakes."""
    from rlinf.scheduler import Cluster, Worker

    def init(worker):
        worker._rank, worker._world_size = 0, 1
        worker._group_name = "test-agents"
        worker._logger = logging.getLogger(__name__)
        worker._stacklevel = 1

    monkeypatch.setattr(Worker, "__init__", init)
    monkeypatch.setattr(
        Worker, "acquire_free_port", lambda self: Cluster.find_free_port()
    )
    monkeypatch.setattr(
        "rlinf.workers.agent.agent_loop.hf_tokenizer",
        lambda *a, **k: Qwen35ToyTokenizer(),
    )
    # Builds the patched agentic tokenizer class that skips sglang's markers.
    qwen35_agentic_tokenizer(monkeypatch)
    package = tmp_path / "fake-rpent" / "robots" / "libero"
    package.mkdir(parents=True)
    for init_file in (package.parent / "__init__.py", package / "__init__.py"):
        init_file.touch()
    (package / "rl_episode.py").write_text(FAKE_EPISODE)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join([str(package.parents[1]), os.environ.get("PYTHONPATH", "")]),
    )

    def make(
        fake,
        *,
        seq_length=4096,
        logprob=-0.5,
        retries=1,
        process_timeout_s=30,
        keep_episodes="all",
        env_gpus=(0, 1),
    ):
        fake.state_dir = tmp_path / f"fake-state-{len(list(tmp_path.iterdir()))}"
        fake.state_dir.mkdir()
        monkeypatch.setenv(
            "FAKE_RPENT",
            json.dumps(
                {
                    "turns": fake.turns,
                    "success": fake.success,
                    "failure": fake.failure,
                    "tools": TOOLS,
                    "state_dir": str(fake.state_dir),
                }
            ),
        )
        cfg = OmegaConf.create(
            {
                "runner": {
                    "task_type": "reasoning",
                    "seq_length": seq_length,
                    "logger": {"log_path": str(tmp_path)},
                },
                "algorithm": {
                    "recompute_logprobs": True,
                    "sampling_params": {"max_new_tokens": 256},
                },
                "rollout": {
                    "return_logprobs": True,
                    "model": {"model_path": "toy", "trust_remote_code": False},
                },
                "agentloop": {
                    "toolcall_parser": "qwen3.5",
                    "agentic_tokenizer": "qwen3_5",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "print_outputs": False,
                    "max_turns": 4,
                    "model_request_timeout_s": 5,
                    "episode_timeout_s": 5,
                    "minimum_output_tokens": 1,
                    "max_infra_retries": retries,
                    "max_infra_retry_rate": 0.05,
                    "merge_prefix_turns": True,
                },
                "tools": {
                    "rpent": {
                        "python": sys.executable,
                        "vla_endpoint": "http://vla",
                        "sam3_endpoint": "http://sam3",
                        "libero_type": "pro",
                        "max_episode_steps": 10,
                        "env_gpus": list(env_gpus),
                        "episodes_per_worker": 2,
                        "output_dir": str(fake.state_dir / "episodes"),
                        "keep_episodes": keep_episodes,
                        "process_timeout_s": process_timeout_s,
                    }
                },
            }
        )
        worker = RPentAgentLoopWorker(cfg, SimpleNamespace(rollout_dp_size=1))
        tokenizer = worker.tokenizer
        worker.prompts = []

        async def generate(prompt_ids, sampling_params):
            turn = len(worker.prompts)
            response = tokenizer(
                f"step {turn}\n\n<tool_call>\n<function=look>\n<parameter=x>\n1\n"
                "</parameter>\n</function>\n</tool_call><|im_end|>"
            )["input_ids"]
            worker.prompts.append((prompt_ids, sampling_params, response))
            return {
                "output_ids": response,
                "prompt_tokens": len(prompt_ids),
                "logprobs": [logprob] * len(response),
                "finish_reason": "stop",
            }

        worker.generate = generate
        return worker

    return make


def run_episodes(worker, count=1):
    """Run ``count`` concurrent episodes while the worker's policy endpoint serves."""
    task = [json.dumps({"suite": "goal"})]

    async def main():
        await worker._model_server.start()
        try:
            return await asyncio.gather(
                *(worker.run_one_query([], task) for _ in range(count)),
                return_exceptions=count > 1,
            )
        finally:
            await worker._model_server.stop()

    return asyncio.run(asyncio.wait_for(main(), timeout=60))


def run_query(worker):
    return run_episodes(worker)[0]


def test_episode_turns_merge_into_one_masked_row_scored_by_the_outcome(make_agent):
    fake = FakeRPent(turns=2, success=True)
    worker = make_agent(fake)
    result = run_query(worker)
    (row,) = result.single_turn_outputs
    (first, _, first_response), (second, _, second_response) = worker.prompts
    gap = len(second) - len(first) - len(first_response)
    # The planner's echoed reply continues the sampled tokens.
    assert second[: len(first) + len(first_response)] == first + first_response
    assert row.prompt_ids == first
    assert row.prompt_ids + row.response_ids == second + second_response
    # Only sampled tokens are scored; the tool result between turns is not.
    assert row.response_mask == (
        [1] * len(first_response) + [0] * gap + [1] * len(second_response)
    )
    assert row.response_logprobs == [-0.5 if m else 0.0 for m in row.response_mask]
    assert row.reward_score == 1.0 and row.extra_fields["turn_ids"] == [0, 1]
    assert (
        result.extra_fields["turns"] == 2 and result.extra_fields["init_state_id"] == 3
    )
    assert result.extra_fields["success"] and result.extra_fields["infra_retries"] == 0
    assert len(fake.calls) == 1


def test_output_budget_is_capped_by_the_remaining_sequence(make_agent):
    worker = make_agent(FakeRPent(turns=1))
    run_query(worker)
    (prompt, sampling_params, _) = worker.prompts[0]
    assert sampling_params == {
        "max_new_tokens": 256,
        "stop_token_ids": [Qwen35ToyTokenizer.eos_token_id],
    }
    # Only 40 tokens remain after the same prompt.
    worker = make_agent(FakeRPent(turns=1), seq_length=len(prompt) + 40)
    run_query(worker)
    assert worker.prompts[0][1]["max_new_tokens"] == 40


def test_request_without_output_space_ends_the_episode_as_a_failure(make_agent):
    fake = FakeRPent(turns=1)
    worker = make_agent(fake, seq_length=100)
    worker.cfg.agentloop.minimum_output_tokens = 64
    result = run_query(worker)
    assert worker.prompts == [] and fake.rejections == ["context_length_exceeded"]
    assert not result.extra_fields["success"]
    assert result.extra_fields["termination_reason"] == "context_limit"
    assert result.single_turn_outputs == []


@pytest.mark.parametrize("logprob", [math.nan, math.inf])
def test_non_finite_behavior_logprobs_fail_without_retry(make_agent, logprob):
    fake = FakeRPent(turns=1)
    with pytest.raises(ValueError, match="finite behavior logprob"):
        run_query(make_agent(fake, logprob=logprob))
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "failure,kind",
    [
        ("env_setup", "env_setup"),
        ("killed", "process"),
        ("model_transport", "model_transport"),
    ],
)
def test_service_failures_retry_with_a_fresh_episode(make_agent, failure, kind):
    fake = FakeRPent(turns=1, failure=failure)
    result = run_query(make_agent(fake))
    first, second = fake.calls
    assert first["argv"] != second["argv"]
    assert result.extra_fields["infra_retries"] == 1
    (error,) = result.extra_fields["infra_retry_errors"]
    assert error["kind"] == kind
    # Without retries the failure reaches the caller.
    with pytest.raises(RPentEpisodeError):
        run_query(make_agent(FakeRPent(turns=1, failure=failure), retries=0))


@pytest.mark.parametrize(
    "failure,message",
    [("crash", "exit status 1"), ("backend", "stopped without an outcome")],
)
def test_crashes_and_planner_failures_are_not_retried(make_agent, failure, message):
    fake = FakeRPent(turns=1, failure=failure)
    with pytest.raises(RuntimeError, match=message):
        run_query(make_agent(fake))
    assert len(fake.calls) == 1


def test_an_episode_past_its_time_limit_is_stopped(make_agent):
    fake = FakeRPent(turns=1, failure="hang")
    with pytest.raises(RPentEpisodeError, match="process_timeout_s") as error:
        run_query(make_agent(fake, retries=0, process_timeout_s=1))
    assert error.value.kind == "process"
    (call,) = fake.calls
    with pytest.raises(ProcessLookupError):
        os.kill(call["pid"], 0)


def test_episodes_render_on_slot_gpus_and_successes_drop_artifacts(make_agent):
    fake = FakeRPent(turns=1)
    run_episodes(make_agent(fake, keep_episodes="failures", env_gpus=(4, 5)), 3)
    devices = {
        call["argv"][call["argv"].index("--cuda-device") + 1] for call in fake.calls
    }
    assert devices == {"4", "5"}
    assert list((fake.state_dir / "episodes").iterdir()) == []


@pytest.mark.parametrize("retried,stops", [(2, False), (3, True)])
def test_retry_rate_above_the_limit_stops_the_update(make_agent, retried, stops):
    worker = make_agent(FakeRPent())
    error = {"type": "RPentEpisodeError", "kind": "process", "message": "lost"}

    def episode(retries):
        return worker.get_rollout_metrics(
            SimpleNamespace(
                extra_fields_traj={
                    "success": [False],
                    "infra_retry_errors": [retries],
                    "turns": [3],
                },
                num_sequence=1,
            )
        )

    metrics = [episode([error]) for _ in range(retried)]
    metrics += [episode([]) for _ in range(10 - retried)]
    if stops:
        with pytest.raises(RuntimeError, match="retry rate 30.00%"):
            worker.post_process_metric(metrics)
    else:
        # Too few retried episodes to stop on, even above the rate limit.
        result = worker.post_process_metric(metrics)
        assert result["rpent/infra_retry_rate"] == 0.2
        assert result["rpent/infra_retries/process"] == 0.2


def compose(overrides=(), config_name="train_qwen3.5_4b_ppo"):
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).parents[2] / "examples/agent/rpent/config"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=list(overrides))


@pytest.mark.parametrize("config_name", ["train_qwen3.5_4b_ppo", "eval_qwen3.5_4b"])
def test_example_config_satisfies_the_rpent_contract(config_name):
    from rlinf.config import validate_rpent_cfg

    validate_rpent_cfg(compose(config_name=config_name))


# Shared synchronous agent PPO settings are covered by the SearchR1 tests.
@pytest.mark.parametrize(
    "override,message",
    [
        ("tools.rpent.episodes_per_worker=0", "one episode per worker"),
        ("tools.rpent.keep_episodes=some", "keep_episodes"),
        ("agentloop.toolcall_parser=qwen2.5", "Qwen3.5 chats"),
        ("agentloop.minimum_output_tokens=4096", "minimum_output_tokens"),
        ("critic.use_critic_model=false", "critic.use_critic_model=True"),
    ],
)
def test_invalid_settings_fail_before_launch(override, message):
    from rlinf.config import validate_rpent_cfg

    with pytest.raises(ValueError, match=message):
        validate_rpent_cfg(compose([override]))


def test_evaluation_reports_success_overall_and_per_suite():
    from rlinf.agents.rpent.eval_runner import summarize_episodes

    episodes = [
        {"task": {"suite": "libero_goal"}, "success": True},
        {"task": {"suite": "libero_goal"}, "success": False},
        {"task": {"suite": "libero_spatial"}, "success": True},
    ]
    assert summarize_episodes(episodes) == {
        "episodes": 3,
        "success_rate": 2 / 3,
        "suites": {
            "libero_goal": {"episodes": 2, "success_rate": 0.5},
            "libero_spatial": {"episodes": 1, "success_rate": 1.0},
        },
    }
