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
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from _qwen35_chat import Qwen35ToyTokenizer, qwen35_agentic_tokenizer
from omegaconf import OmegaConf, open_dict

from rlinf.agents.rpent.episode import RPentEpisodeError, merge_prefix_turns
from rlinf.agents.rpent.history import (
    AppendOnlyHistory,
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


@pytest.fixture
def qwen35(monkeypatch):
    return qwen35_agentic_tokenizer(monkeypatch)


def test_episode_failure_survives_ray_exception_serialization():
    error = RPentEpisodeError("backend.open rejected the task", kind="startup")
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is RPentEpisodeError
    assert str(restored) == str(error)
    assert restored.kind == "startup"


def test_merge_prefix_turns_folds_image_rows_and_keeps_the_last_image_set():
    """Image turns merge on the token prefix; the row takes the last image set.

    Each turn's prompt expands every image of the chain in order, so a merged
    row carries the last merged turn's images, which match its merged tokens. A
    rewrite breaks the prefix and starts a new row with its own images.
    """
    import torch

    def images(count):
        return {
            "image_grid_thw": torch.ones(count, 3, dtype=torch.long),
            "num_images": count,
        }

    def turn(turn_id, prompt, response, count):
        return AgentLoopOutput(
            prompt_ids=prompt,
            response_ids=response,
            response_logprobs=[-float(token) for token in response],
            response_weight_versions=[0] * len(response),
            response_images=images(count),
            is_end=True,
            reward_score=1.0,
            extra_fields={"turn_id": turn_id, "tool_names": [f"tool{turn_id}"]},
        )

    turns = [
        turn(0, [1, 2], [3, 4], 1),  # image A
        # Adds a tool observation with image B; its prompt extends turn 0.
        turn(1, [1, 2, 3, 4, 9, 8], [5, 6], 2),
        # Compaction rewrote the history: a new row with its own image set.
        turn(2, [1, 2, 7], [8], 1),
    ]
    first, second = merge_prefix_turns(turns, max_scored_tokens=8)
    assert first.prompt_ids == [1, 2] and first.response_ids == [3, 4, 9, 8, 5, 6]
    assert first.response_mask == [1, 1, 0, 0, 1, 1]
    assert first.extra_fields["turn_ids"] == [0, 1]
    # The merged row holds every image of the chain, once each, in order.
    assert first.response_images["num_images"] == 2
    assert torch.equal(
        first.response_images["image_grid_thw"], torch.ones(2, 3, dtype=torch.long)
    )
    assert second.prompt_ids == [1, 2, 7] and second.response_images["num_images"] == 1


# Stands in for RPent's ``robots.rl.rl_episode``: it plans against the
# worker's policy endpoint and writes the outcome, as the real process does.
FAKE_EPISODE = """
import argparse, json, os, urllib.error, urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--request", type=Path)
args, _ = parser.parse_known_args()
script = json.loads(os.environ["FAKE_RPENT"])
state = Path(script["state_dir"])
request = json.loads(args.request.read_text())
with (state / "calls.log").open("a") as calls:
    calls.write("started\\n")
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
        break
    tool_call = message["tool_calls"][0]["id"]
    messages += [message, {"role": "tool", "tool_call_id": tool_call, "content": "seen"}]
error_kind = {200: None, 400: "context_limit"}.get(status, "backend")
outcome = {
    "success": error_kind is None,
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
    """An episode process whose planner calls the policy endpoint ``turns`` times."""

    def __init__(self, *, turns=2):
        self.turns = turns
        self.state_dir: Path | None = None

    def lines(self, name: str) -> list[str]:
        path = self.state_dir / name
        return path.read_text().splitlines() if path.exists() else []

    @property
    def calls(self) -> list[str]:
        return self.lines("calls.log")


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
    package = tmp_path / "fake-rpent" / "robots" / "rl"
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
        logprob=-0.5,
    ):
        fake.state_dir = tmp_path / f"fake-state-{len(list(tmp_path.iterdir()))}"
        fake.state_dir.mkdir()
        monkeypatch.setenv(
            "FAKE_RPENT",
            json.dumps(
                {
                    "turns": fake.turns,
                    "tools": TOOLS,
                    "state_dir": str(fake.state_dir),
                }
            ),
        )
        cfg = OmegaConf.create(
            {
                "runner": {
                    "task_type": "reasoning",
                    "seq_length": 4096,
                    "logger": {"log_path": str(tmp_path)},
                },
                "algorithm": {
                    "recompute_logprobs": True,
                    "sampling_params": {
                        "max_new_tokens": 256,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "min_p": 0.0,
                        "presence_penalty": 0.0,
                        "repetition_penalty": 1.0,
                    },
                },
                "rollout": {
                    "return_logprobs": True,
                    "replica_affinity": True,
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
                    "max_infra_retries": 1,
                    "max_infra_retry_rate": 0.05,
                    "merge_prefix_turns": True,
                },
                "tools": {
                    "rpent": {
                        "python": sys.executable,
                        "repo_path": str(package.parents[1]),
                        "remote_toolkit_endpoint": "http://simulator",
                        "image_size": 1024,
                        "max_actions_per_call": 3,
                        "reasoning_history": -1,
                        "image_history": -1,
                        "libero_type": "pro",
                        "max_episode_steps": 10,
                        "env_gpus": [0, 1],
                        "episodes_per_worker": 2,
                        "output_dir": str(fake.state_dir / "episodes"),
                        "keep_episodes": "all",
                        "process_timeout_s": 30,
                    }
                },
            }
        )
        worker = RPentAgentLoopWorker(
            cfg, SimpleNamespace(rollout_num_model_instances=1)
        )
        tokenizer = worker.tokenizer
        worker.prompts = []

        async def generate(prompt_ids, sampling_params):
            turn = len(worker.prompts)
            response = tokenizer(
                f"step {turn}\n\n<tool_call>\n<function=look>\n"
                "<parameter=x>\n1\n</parameter>\n</function>\n</tool_call><|im_end|>"
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


def run_query(worker):
    async def run():
        await worker._model_server.start()
        try:
            return await worker.run_one_query([], [json.dumps({"suite": "goal"})])
        finally:
            await worker._model_server.stop()

    return asyncio.run(asyncio.wait_for(run(), timeout=60))


def test_episode_turns_merge_into_one_masked_row_scored_by_the_outcome(make_agent):
    fake = FakeRPent()
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


@pytest.mark.parametrize("logprob", [math.nan, math.inf])
def test_non_finite_behavior_logprobs_fail_without_retry(make_agent, logprob):
    fake = FakeRPent(turns=1)
    with pytest.raises(ValueError, match="finite behavior logprob"):
        run_query(make_agent(fake, logprob=logprob))
    assert len(fake.calls) == 1


# Shared synchronous agent PPO settings are covered by the SearchR1 tests.


def test_token_space_history_preserves_sampled_ids_and_expands_only_new_images(qwen35):
    import base64
    import io

    from PIL import Image

    from rlinf.agents.rpent.history import TemplateMessageError

    def image_part(color):
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
        return {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,"
                + base64.b64encode(buffer.getvalue()).decode()
            },
        }

    history = AppendOnlyHistory(qwen35)
    first_request = {"messages": [{"role": "user", "content": [image_part("red")]}]}
    compact = history.render_token_space(first_request)
    start, end = compact.image_spans[0]
    expanded = (
        compact.input_ids[:start]
        + compact.input_ids[start:end] * 4
        + compact.input_ids[end:]
    )
    history.accept_token_space(
        {
            "prompt_token_ids": expanded,
            "image_spans": [(start, start + 4)],
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }
    )
    response = qwen35.tokenizer("look again<|im_end|>")["input_ids"]
    assistant = {"role": "assistant", "content": "look again"}
    history.record(first_request, expanded, response, assistant)
    next_request = {
        "messages": [
            *first_request["messages"],
            assistant,
            {"role": "user", "content": [image_part("blue")]},
        ]
    }
    following = history.render_token_space(next_request)
    assert following.input_ids[: len(expanded + response)] == expanded + response
    assert following.prefix_length == len(expanded + response)
    assert following.image_spans[0] == (start, start + 4)
    assert following.image_spans[1][1] - following.image_spans[1][0] == 1
    assert following.pixel_values is None
    # Compaction starts a new chain; previous image occurrences cannot leak in.
    reset = history.render_token_space(
        {"messages": [{"role": "user", "content": [image_part("blue")]}]}
    )
    assert reset.prefix_length == 0 and len(reset.images) == 1
    with pytest.raises(TemplateMessageError, match="Literal vision"):
        history.render_token_space(
            {"messages": [{"role": "user", "content": "print <|image_pad|>"}]}
        )


@pytest.mark.parametrize(
    "key", ["enable_shared_multi_modal_processor", "enable_token_space_processor"]
)
def test_token_space_checks_exact_served_prompt_ids(make_agent, key):
    worker = make_agent(FakeRPent(turns=1))
    worker.cfg.rollout.sglang = {key: True}
    generated = {
        "output_ids": [9],
        "prompt_tokens": 2,
        "prompt_token_ids": [1, 3],
        "logprobs": [-0.2],
        "weight_versions": [0],
    }
    with pytest.raises(ValueError, match="different prompt IDs"):
        worker._check_generation(generated, [1, 2])
    generated["prompt_token_ids"] = [1, 2]
    worker._check_generation(generated, [1, 2])
    generated["output_ids"] = [worker.tokenizer.convert_tokens_to_ids("<|image_pad|>")]
    with pytest.raises(ValueError, match="Generated vision markers"):
        worker._check_generation(generated, [1, 2])
    worker.cfg.rollout.env_vars = {
        "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": "rlinf.hybrid_engines.sglang.multimodal"
    }
    worker._check_generation(generated, [1, 2])
    del generated["prompt_token_ids"]
    with pytest.raises(ValueError, match="different prompt IDs"):
        worker._check_generation(generated, [1, 2])


def compose(overrides=(), config_name="train_qwen3.5_4b_episode_sync_ppo"):
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).parents[2] / "examples/agent/rpent/config"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    # Templates require deployment paths; CPU tests supply explicit stand-ins.
    for name, value in {
        "runner.output_dir": "/test/training",
        "data.train_data_paths": ["/test/tasks.jsonl"],
        "data.val_data_paths": ["/test/eval.jsonl"],
        "tools.rpent.python": "/test/env/bin/python",
        "tools.rpent.remote_toolkit_endpoint": "http://127.0.0.1:28490",
        "tools.rpent.output_dir": "/test/episodes",
        "rollout.model.model_path": "/test/full-model",
    }.items():
        parent, key = name.rsplit(".", 1)
        node = OmegaConf.select(cfg, parent)
        if node is not None and OmegaConf.is_missing(node, key):
            OmegaConf.update(cfg, name, value)
    return cfg


@pytest.mark.parametrize(
    "recipe,requires_sync",
    [
        ("eval_qwen3.5_4b_episode_sync", False),
        ("eval_qwen3.5_4b_episode_token_space", False),
        ("train_qwen3.5_4b_episode_sync_grpo", True),
    ],
)
@pytest.mark.parametrize("version", [None, 0])
def test_episode_recipes_distinguish_static_weights_from_actor_sync(
    make_agent, recipe, requires_sync, version
):
    cfg = compose(config_name=recipe)
    worker = make_agent(FakeRPent(turns=1))
    worker.cfg.runner.synchronous = cfg.runner.synchronous
    worker.cfg.tools.rpent.update(
        {
            key: cfg.tools.rpent[key]
            for key in (
                "remote_toolkit_endpoint",
                "image_size",
                "max_actions_per_call",
                "reasoning_history",
                "image_history",
            )
        }
    )
    worker.cfg.algorithm.sampling_params.update(cfg.algorithm.sampling_params)
    generate = worker.generate

    async def versioned(*args, **kwargs):
        result = await generate(*args, **kwargs)
        if version is not None:
            result.update(version_start=version, version_end=version)
        return result

    worker.generate = versioned
    if requires_sync and version is None:
        with pytest.raises(ValueError, match="installed weight version"):
            run_query(worker)
        return
    result = run_query(worker)
    assert result.extra_fields["turns"] == 1
    (turn,) = result.single_turn_outputs
    assert turn.version_start == turn.version_end == version
    assert len(turn.response_ids) == len(turn.response_logprobs)
    if requires_sync:
        assert set(turn.response_weight_versions) == {version}


@pytest.mark.parametrize(
    "evaluation,turns,valid",
    [
        (True, 1, True),
        (True, 0, False),
        (True, True, False),
        (False, 25, True),
        (False, 24, False),
    ],
)
def test_request_budget_preserves_evaluation_and_training_limits(
    evaluation, turns, valid
):
    from rlinf.agents.rpent.config_validation import validate_rpent_cfg

    cfg = compose(
        config_name=(
            "eval_qwen3.5_4b_episode_sync"
            if evaluation
            else "train_qwen3.5_4b_episode_sync_ppo"
        )
    )
    cfg.agentloop.max_turns = turns
    if valid:
        validate_rpent_cfg(cfg)
    else:
        with pytest.raises(ValueError, match="max_turns must be at least"):
            validate_rpent_cfg(cfg)


@pytest.mark.parametrize(
    "config_name",
    [
        "train_qwen3.5_4b_episode_sync_ppo",
        "eval_qwen3.5_4b_episode_sync",
    ],
)
def test_example_configs_preserve_planner_and_training_contracts(config_name):
    """Training ratios use full support; evaluation keeps planner sampling."""
    cfg = compose(config_name=config_name)
    sampling = cfg.algorithm.sampling_params
    if config_name.startswith("train_"):
        assert (sampling.temperature, sampling.top_p, sampling.top_k) == (1.0, 1.0, -1)
        assert sampling.presence_penalty == 0.0
        assert sampling.repetition_penalty == 1.0
        assert cfg.agentloop.thinking_token_budget is None
        assert cfg.rollout.sglang.backend_type == "server"
        assert sampling.max_new_tokens == 32768
    assert cfg.agentloop.max_turns == 80
    assert (
        cfg.rollout.env_vars.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE
        == "rlinf.hybrid_engines.sglang.multimodal"
    )
    assert (sampling.temperature, sampling.top_p, sampling.top_k) == (1.0, 1.0, -1)
    if cfg.runner.task_type == "reasoning":
        for component in (cfg.actor, cfg.critic):
            assert not component.megatron.ckpt_convertor.hf_model_path.endswith("-text")
            assert component.model.freeze_vision_model
            assert component.model.freeze_vision_projection
    assert cfg.agentloop.chat_template_kwargs.enable_thinking is True
    assert cfg.runner.seq_length == 262144
    assert cfg.tools.rpent.process_timeout_s > cfg.agentloop.episode_timeout_s
    compaction = cfg.agentloop.history_compaction
    if "target_tokens" in compaction:
        assert compaction.target_tokens <= 0.7 * cfg.runner.seq_length
        # A compacted prompt still gets the whole response plus schema headroom.
        assert cfg.runner.seq_length - compaction.target_tokens >= 32768 + 16384
    else:
        assert compaction.context_window == cfg.runner.seq_length
        assert compaction.max_fraction == 0.7


@pytest.mark.parametrize(
    "mode,invalid,value,error",
    [
        ("eval", None, None, None),
        (
            "ppo",
            "rollout.env_vars.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE",
            "rlinf.hybrid_engines.sglang.multimodal",
            None,
        ),
        ("ppo", None, None, None),
        ("eval", "tools.rpent.store_images_on_disk", False, "store_images_on_disk"),
        (
            "eval",
            "rollout.env_vars.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE",
            "custom",
            "Remove",
        ),
        ("ppo", "actor.model.pipeline_model_parallel_size", 2, "PP=1"),
        ("ppo", "critic.megatron.mbridge", False, "Megatron-Bridge"),
    ],
)
def test_token_space_validation_scopes_learner_checks(mode, invalid, value, error):
    from rlinf.config import validate_rpent_cfg

    name = (
        "eval_qwen3.5_4b_episode_sync"
        if mode == "eval"
        else "train_qwen3.5_4b_episode_sync_ppo"
    )
    cfg = compose(
        [
            "++rollout.sglang.enable_shared_multi_modal_processor=true",
            "rollout.env_vars.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=null",
            "++tools.rpent.store_images_on_disk=true",
            "tools.rpent.keep_episodes=all",
        ],
        config_name=name,
    )
    if invalid:
        OmegaConf.update(cfg, invalid, value)
    if error:
        with pytest.raises(ValueError, match=error):
            validate_rpent_cfg(cfg)
    else:
        validate_rpent_cfg(cfg)
        if mode == "eval":
            assert cfg.runner.task_type == "reasoning_eval"


@pytest.mark.parametrize("field", ["cache_bytes", "prefetch_batches", "prefetch_bytes"])
def test_legacy_image_budgets_normalize_and_reject_conflicts(field):
    from rlinf.config import validate_multimodal_learner_cfg, validate_rpent_cfg

    cfg = compose(
        [
            "rollout.sglang.enable_shared_multi_modal_processor=true",
            "rollout.env_vars.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=null",
            "tools.rpent.store_images_on_disk=true",
        ]
    )
    budgets = {"cache_bytes": 1024, "prefetch_batches": 2, "prefetch_bytes": 4096}
    for key, value in budgets.items():
        OmegaConf.update(
            cfg, f"tools.rpent.training_image_{key}", value, force_add=True
        )
    OmegaConf.set_struct(cfg, True)
    validate_rpent_cfg(cfg)
    validate_multimodal_learner_cfg(cfg)
    assert dict(cfg.actor.multimodal) == dict(cfg.critic.multimodal) == budgets
    snapshot = OmegaConf.to_container(cfg)
    validate_rpent_cfg(cfg)
    assert OmegaConf.to_container(cfg) == snapshot
    OmegaConf.update(cfg, f"critic.multimodal.{field}", budgets[field] + 1)
    with pytest.raises(ValueError, match=f"critic.multimodal.{field} conflicts"):
        validate_rpent_cfg(cfg)
    del cfg.tools.rpent
    # Generic consumers can choose independent actor and critic budgets.
    validate_multimodal_learner_cfg(cfg)
    assert cfg.actor.multimodal[field] == budgets[field]
    assert cfg.critic.multimodal[field] == budgets[field] + 1


@pytest.fixture(params=[False, True], ids=["uncached", "disk-cache"])
def native_token_space(tmp_path, request):
    """Use real SGLang preprocessing when local processor assets are supplied."""
    model_path = os.environ.get("RLINF_TOKEN_SPACE_TEST_MODEL")
    if not model_path:
        pytest.skip(
            "Set RLINF_TOKEN_SPACE_TEST_MODEL to local Qwen3.5 processor assets"
        )

    from rlinf.hybrid_engines.sglang.multimodal.token_space import SGLangTokenSpace

    settings = {
        "model_path": model_path,
        "tokenizer_path": model_path,
        "tokenizer_mode": "auto",
        "tokenizer_backend": "huggingface",
        "trust_remote_code": False,
        "revision": None,
        "model_config_parser": "auto",
        "json_model_override_args": "{}",
        "image_processor_backend": "pil",
        "disable_fast_image_processor": True,
        "mm_process_config": {},
    }
    if request.param:
        settings.update(
            enable_token_space_preprocess_cache=True,
            token_space_preprocess_cache_dir=str(tmp_path / "cache"),
            token_space_preprocess_cache_trust_url=True,
        )
    processor = SGLangTokenSpace(settings)
    try:
        yield processor
    finally:
        processor.shutdown()


@pytest.fixture
def native_multiturn(native_token_space, tmp_path):
    import io

    from PIL import Image
    from transformers import AutoConfig

    from rlinf.agents.rpent.history import ImagePrompt

    processor = native_token_space
    config = AutoConfig.from_pretrained(
        processor.settings["model_path"], local_files_only=True
    )
    images = []
    for mode, size, color in [
        ("RGB", (64, 96), "red"),
        ("RGBA", (96, 64), (0, 0, 255, 80)),
    ]:
        data = io.BytesIO()
        Image.new(mode, size, color).save(data, format="PNG")
        images.append(data.getvalue())
    media_directory = tmp_path / "images"
    sources = images
    if processor.cache_directory is not None:
        stored = ImagePrompt([], images).store_images(media_directory)
        sources = [ref["path"] for ref in stored["image_refs"]]
    marker = [
        config.vision_start_token_id,
        config.image_token_id,
        config.vision_end_token_id,
    ]
    first = asyncio.run(
        processor.prepare(
            [42, *marker, 43],
            sources[:1],
            {"prefix_length": 0, "image_spans": [(2, 3)]},
        )
    )
    sampled = first["prompt_token_ids"] + [44, 45]
    compact = sampled + marker + [46]
    second = asyncio.run(
        processor.prepare(
            compact,
            sources,
            {
                "prefix_length": len(sampled),
                "image_spans": [
                    *first["image_spans"],
                    (len(sampled) + 1, len(sampled) + 2),
                ],
            },
        )
    )
    assert second["prompt_token_ids"][: len(sampled)] == sampled
    assert second["image_spans"][0] == first["image_spans"][0]
    assert len(second["image_spans"]) == 2
    refs = ImagePrompt(
        second["prompt_token_ids"], images, second["image_grid_thw"]
    ).store_images(media_directory)
    entry = {**refs, **second}
    ids = torch.tensor([second["prompt_token_ids"] + [47, 48, 0, 0]])
    mask = ids.ne(0)

    return SimpleNamespace(
        processor=processor,
        images=images,
        entry=entry,
        batch={"input_ids": ids, "attention_mask": mask, "multi_modal_inputs": [entry]},
    )


def test_native_token_space_multiturn_replay_and_cache(native_multiturn, monkeypatch):
    from sglang.srt.multimodal.cache import DiskPreprocessCache

    from rlinf.hybrid_engines.sglang.multimodal.token_space import SGLangTokenSpace
    from rlinf.utils.multimodal import MultimodalBatchPreparer

    case = native_multiturn
    processor = case.processor
    preparer = MultimodalBatchPreparer(
        processor.settings["model_path"], token_space=True, cache_bytes=8 * 1024 * 1024
    )
    cache_hits = []
    original_get = DiskPreprocessCache.get

    def counted_get(cache, key):
        result = original_get(cache, key)
        if result is not None:
            cache_hits.append(str(cache.directory))
        return result

    monkeypatch.setattr(DiskPreprocessCache, "get", counted_get)
    first_inputs = preparer.for_forward(case.batch, "cpu")
    if processor.cache_directory is not None:
        assert cache_hits == [processor.cache_directory] * 2
        uncached = SGLangTokenSpace(
            {**processor.settings, "enable_token_space_preprocess_cache": False}
        )
        try:
            assert uncached.fingerprint == processor.fingerprint
            pixels, grid = asyncio.run(uncached.replay(case.entry, case.images))
            assert torch.equal(pixels, first_inputs["pixel_values"])
            assert torch.equal(grid, first_inputs["image_grid_thw"])
        finally:
            uncached.shutdown()
    again = preparer.for_forward(case.batch, "cpu")
    for key in first_inputs:
        assert torch.equal(first_inputs[key], again[key])
    assert first_inputs["multimodal_position_ids"].shape == (
        3,
        *case.batch["input_ids"].shape,
    )


def test_native_token_space_prefetch_matches_synchronous_replay(native_multiturn):
    from rlinf.utils.multimodal import MultimodalBatchPreparer

    case = native_multiturn
    baseline = MultimodalBatchPreparer(
        case.processor.settings["model_path"], token_space=True
    )
    expected = baseline.for_forward(case.batch, "cpu")
    prefetch = MultimodalBatchPreparer(
        case.processor.settings["model_path"],
        token_space=True,
        prefetch_batches=2,
        prefetch_bytes=2 * sum(t.nbytes for t in expected.values()),
    )
    with prefetch.iterator_context(iter([case.batch] * 3)) as iterator:
        for batch in iterator:
            actual = prefetch.for_forward(batch, "cpu")
            assert actual.keys() == expected.keys()
            for key in expected:
                assert actual[key].device.type == "cpu"
                assert torch.equal(actual[key], expected[key])


@pytest.mark.parametrize(
    "overrides, error",
    [
        (["rollout.sglang.token_space_preprocess_cache_trust_url=false"], "trust_url"),
        (["rollout.sglang.token_space_preprocess_cache_dir=relative"], "absolute"),
        (
            ["rollout.sglang.token_space_preprocess_cache_check_rate=1.1"],
            "between 0 and 1",
        ),
        (
            ["rollout.sglang.enable_shared_multi_modal_processor=false"],
            "enable_shared_multi_modal_processor",
        ),
    ],
)
def test_shared_preprocess_cache_rejects_invalid_settings(tmp_path, overrides, error):
    from rlinf.config import validate_rollout_cfg

    cfg = compose(
        [
            "rollout.model.model_path=/unused/model",
            "rollout.sglang.enable_shared_multi_modal_processor=true",
            "rollout.sglang.enable_token_space_preprocess_cache=true",
            "rollout.sglang.token_space_preprocess_cache_trust_url=true",
            f"rollout.sglang.token_space_preprocess_cache_dir={tmp_path}",
        ]
    )
    validate_rollout_cfg(cfg.rollout, cfg.algorithm, cfg.actor)
    for override in overrides:
        cfg.merge_with_dotlist([override])
    with pytest.raises(ValueError, match=error):
        validate_rollout_cfg(cfg.rollout, cfg.algorithm, cfg.actor)


@pytest.mark.parametrize("prefetch_batches", [0, 2, 8])
def test_image_prefetch_preserves_order_and_gradients(prefetch_batches, monkeypatch):
    from rlinf.utils.multimodal import MultimodalBatchPreparer

    consumed = []
    source_ids = []
    # Simulate distinct host copies without depending on a GPU for this test.
    monkeypatch.setattr(torch.Tensor, "cpu", lambda tensor: tensor.clone())

    def batches():
        for index in range(5):
            consumed.append(index)
            ids = torch.tensor([[index + 1, 9]])
            source_ids.append(ids)
            yield {
                "input_ids": ids,
                "attention_mask": torch.ones(1, 2),
                "multi_modal_inputs": [
                    {
                        "pixel_values": torch.full((2, 3), float(index)),
                        "image_grid_thw": torch.tensor([[1, 1, 2]]),
                    }
                ],
            }

    # Each result is 48 bytes; the byte limit admits at most two batches.
    preparer = MultimodalBatchPreparer(
        "unused", prefetch_batches=prefetch_batches, prefetch_bytes=96
    )
    weight = torch.tensor(0.5, requires_grad=True)
    with preparer.iterator_context(batches()) as iterator:
        for index, batch in enumerate(iterator):
            assert batch["input_ids"] is source_ids[index]
            inputs = preparer.for_forward(batch, "cpu")
            assert torch.equal(inputs["pixel_values"], torch.full((2, 3), float(index)))
            if prefetch_batches:
                assert (
                    len(consumed) <= index + 3
                )  # Two admitted, one compact lookahead.
            (inputs["pixel_values"] * weight).sum().backward()
            assert "_prepared_multimodal_inputs" not in batch
    assert weight.grad.item() == 6 * sum(range(5))


@pytest.mark.parametrize("failure", ["prepare", "consumer", "budget"])
def test_image_prefetch_propagates_failure_and_joins_threads(failure):
    import threading

    from rlinf.utils.multimodal import MultimodalBatchPreparer

    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2),
        "multi_modal_inputs": [],
    }
    if failure == "prepare":
        batch["multi_modal_inputs"] = [
            {
                "format": "legacy",
                "pixel_values": torch.ones(1),
                "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
            }
        ]
    preparer = MultimodalBatchPreparer(
        "unused",
        token_space=True,
        prefetch_batches=2,
        prefetch_bytes=1 if failure == "budget" else 4096,
    )
    error = {
        "prepare": "captured image row",
        "budget": "exceeding",
        "consumer": "model forward",
    }[failure]
    with pytest.raises(ValueError, match=error):
        with preparer.iterator_context(iter([batch, batch])) as iterator:
            next(iterator)
            raise ValueError("model forward failed")
    assert not any(
        thread.name.startswith("rlinf-images") for thread in threading.enumerate()
    )


def test_native_token_space_disk_cache_shared_across_processes(
    native_multiturn, tmp_path
):
    import subprocess

    from rlinf.hybrid_engines.sglang.multimodal.token_space import tensor_digest
    from rlinf.utils.multimodal import MultimodalBatchPreparer

    case = native_multiturn
    if case.processor.cache_directory is None:
        pytest.skip("Cross-process cache check needs the disk-cache variant")
    request = tmp_path / "replay.pt"
    result = tmp_path / "replay.json"
    torch.save(case.batch, request)
    script = """
import json, sys, torch
from pathlib import Path
from sglang.srt.multimodal.cache import DiskPreprocessCache
from rlinf.hybrid_engines.sglang.multimodal.token_space import tensor_digest
from rlinf.utils.multimodal import MultimodalBatchPreparer
hits = []
original = DiskPreprocessCache.get
def get(cache, key):
    value = original(cache, key)
    if value is not None:
        hits.append(key)
    return value
DiskPreprocessCache.get = get
batch = torch.load(sys.argv[1], weights_only=True)
model_path = batch["multi_modal_inputs"][0]["processor_settings"]["model_path"]
preparer = MultimodalBatchPreparer(model_path, token_space=True)
inputs = preparer.for_forward(batch, "cpu")
Path(sys.argv[2]).write_text(json.dumps({"hits": len(hits), "digests": {
    name: tensor_digest(value) for name, value in inputs.items()
}}))
"""
    subprocess.run(
        [sys.executable, "-c", script, str(request), str(result)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    actual = json.loads(result.read_text())
    expected = MultimodalBatchPreparer(
        case.processor.settings["model_path"], token_space=True
    ).for_forward(case.batch, "cpu")
    assert actual["hits"] == 2
    assert actual["digests"] == {
        key: tensor_digest(value) for key, value in expected.items()
    }


def test_release_capture_replays_positions_and_verifies_shared_pixels(tmp_path):
    from rlinf.hybrid_engines.sglang.multimodal.replay import (
        REPLAY_FORMAT,
        store_pixels,
    )
    from rlinf.utils.multimodal import materialize_multimodal_inputs

    pixels = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    data = {
        "format": REPLAY_FORMAT,
        "prompt_token_ids": [10, 11, 12],
        "image_spans": [[1, 2]],
        "pixel_values": pixels,
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "mrope_positions": torch.tensor([[0, 1, 2]] * 3),
        "mrope_position_delta": torch.tensor([0]),
    }
    store_pixels(data, tmp_path)
    ids = torch.tensor([[10, 11, 12, 13, 0]])

    def replay():
        return materialize_multimodal_inputs(
            [data],
            "unused",
            "cpu",
            input_ids=ids,
            attention_mask=ids.ne(0),
            token_space=True,
        )

    result = replay()
    assert torch.equal(result["pixel_values"], pixels.bfloat16())
    assert result["image_input_mask"].tolist() == [[False, True, False, False, False]]
    assert result["multimodal_position_ids"][:, 0].tolist() == [[0, 1, 2, 3, 0]] * 3
    ids[0, 0] = 2
    with pytest.raises(ValueError, match="prompt"):
        replay()
    ids[0, 0] = 10
    Path(data["pixel_refs"][0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        replay()


def test_release_capture_checks_sampled_history_and_request_id():
    from rlinf.hybrid_engines.sglang.multimodal.replay import (
        CAPTURE_KEY,
        capture_file,
        generate_with_capture,
        save_capture,
    )

    rid = "a" * 32
    pixels = torch.ones(4, 2)

    async def generate(input_ids, image_data, sampling_params, return_logprob):
        capture = sampling_params["custom_params"][CAPTURE_KEY]
        output = SimpleNamespace(
            input_ids=input_ids,
            mrope_positions=torch.arange(len(input_ids)).repeat(3, 1),
            mrope_position_delta=0,
            mm_items=[
                SimpleNamespace(
                    offsets=[(1, 1)],
                    feature=pixels,
                    image_grid_thw=torch.tensor([[1, 2, 2]]),
                )
            ],
        )
        save_capture(capture_file(capture["ticket"], rid), output)
        return {"meta_info": {"id": rid}, "output_ids": [23]}

    def sample(ids, history=None):
        return asyncio.run(
            generate_with_capture(
                generate,
                ids,
                ["image"],
                {},
                {"image_spans": [[1, 2]], "history": history},
                False,
            )
        )

    _, _, history = sample([10, 11, 12])
    _, row, _ = sample([10, 11, 12, 23, 24], history)
    assert row["prompt_token_ids"] == [10, 11, 12, 23, 24]
    with pytest.raises(ValueError, match="sampled token history"):
        sample([10, 11, 12, 99, 24], history)
    pixels.zero_()
    with pytest.raises(ValueError, match="pixels"):
        sample([10, 11, 12, 23, 24], history)


def test_tokenspace_grouped_grpo_keeps_native_loss_scales_without_a_critic():
    from rlinf.agents.rpent.config_validation import validate_rpent_cfg

    cfg = compose(
        overrides=[
            "rollout.sglang.token_space_preprocess_cache_dir=/test/cache",
            "tools.rpent.repo_path=/test/rpent",
            "tools.rpent.remote_toolkit_endpoint=http://127.0.0.1:28490",
        ],
        config_name="train_qwen3.5_4b_episode_sync_grpo",
    )
    validate_rpent_cfg(cfg)
    assert cfg.algorithm.clip_ratio_high == 0.28
    assert list(cfg.algorithm.loss_scales) == [
        "group_level",
        "agent_level",
        "turn_level",
    ]
    assert cfg.algorithm.group_size == 8
    assert cfg.data.rollout_batch_size == 32
    assert cfg.data.sampling_batch_size == 40
    assert not cfg.critic.use_critic_model
    assert cfg.actor.megatron.mbridge


@pytest.fixture
def episode_launcher(tmp_path, monkeypatch):
    from rlinf.agents.rpent.launcher import RPentEpisodeLauncher

    (tmp_path / "episode_fixture.py").write_text(
        "import argparse,json,os,time\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--request',type=Path)\n"
        "p.add_argument('--output-dir',type=Path)\n"
        "a,extra=p.parse_known_args()\n"
        "r=json.loads(a.request.read_text())\n"
        "(a.output_dir/'pid').write_text(str(os.getpid()))\n"
        "(a.output_dir/'states.json').write_text('{}')\n"
        "time.sleep(r['task'].get('sleep',0))\n"
        "(a.output_dir/'outcome.json').write_text(json.dumps({\n"
        " 'success':False,'owner':os.environ.get('RPENT_REMOTE_OWNER_ID'),\n"
        " 'egl':os.environ.get('MUJOCO_EGL_DEVICE_ID'),'args':extra}))\n"
    )
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "inherited-device")
    cfg = OmegaConf.create(
        {
            "rollout": {"model": {"model_path": "unused"}},
            "agentloop": {"prompt": "simple"},
            "tools": {
                "rpent": {
                    "python": sys.executable,
                    "repo_path": str(tmp_path),
                    "remote_toolkit_endpoint": "http://unused",
                    "image_size": 1024,
                    "max_actions_per_call": 3,
                    "reasoning_history": -1,
                    "image_history": -1,
                    "max_episode_steps": 10000,
                    "libero_type": "pro",
                    "max_env_initializations_per_worker": 1,
                    "output_dir": str(tmp_path / "episodes"),
                    "keep_episodes": "none",
                    "process_timeout_s": 20,
                }
            },
        }
    )
    launcher = RPentEpisodeLauncher(cfg)
    command = launcher.command

    def fixture_command(output, gpu, *, task=None):
        args = command(output, gpu, task=task)
        return [args[0], "-m", "episode_fixture", *args[3:]]

    monkeypatch.setattr(launcher, "command", fixture_command)
    return launcher


def test_episode_launcher_keeps_candidates_and_cleans_process(episode_launcher):
    from rlinf.agents.rpent.episode import Episode

    launcher = episode_launcher
    candidate = {"cohort_id": "grouped", "candidate_id": 9}
    episode = Episode("candidate", {}, None)
    result = asyncio.run(launcher.run(episode, {}, 3, candidate=candidate))
    output = Path(launcher.cfg.tools.rpent.output_dir) / "candidate"
    assert result["owner"] == "candidate" and result["egl"] is None
    assert "--record-api-trace" not in result["args"]
    assert result["args"][result["args"].index("--max-actions-per-call") + 1] == "3"
    assert result["args"][result["args"].index("--cuda-device") + 1] == "3"
    assert json.loads((output / "sampling-candidate.json").read_text()) == {
        **candidate,
        "episode_id": "candidate",
    }
    with pytest.raises(ProcessLookupError):
        os.kill(int((output / "pid").read_text()), 0)
    asyncio.run(launcher.run(Episode("ordinary", {}, None), {}, 3))
    assert not (output.parent / "ordinary").exists()


def test_episode_launcher_cancellation_reaps_child_and_releases_permit(
    episode_launcher,
):
    from rlinf.agents.rpent.episode import Episode

    launcher = episode_launcher
    candidate = {"cohort_id": "cancelled", "candidate_id": 0}
    output = Path(launcher.cfg.tools.rpent.output_dir) / "cancelled"

    async def run():
        task = asyncio.create_task(
            launcher.run(
                Episode("cancelled", {"sleep": 60}, None), {}, 0, candidate=candidate
            )
        )

        async def started():
            while not (output / "pid").exists():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(started(), timeout=10)
        pid = int((output / "pid").read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        await asyncio.wait_for(
            launcher.run(Episode("after-cancel", {}, None), {}, 0), timeout=10
        )

    asyncio.run(run())


def test_episode_launcher_routes_without_legacy_cli_selectors(episode_launcher):
    from rlinf.agents.rpent.episode import Episode

    launcher = episode_launcher
    launcher.cfg.tools.rpent.remote_toolkit_endpoint = None
    with open_dict(launcher.cfg):
        launcher.cfg.env = {"rpent": {"robotwin": {"endpoint": "http://robotwin"}}}
    result = asyncio.run(
        launcher.run(
            Episode(
                "general",
                {"environment": "robotwin", "task_name": "adjust_bottle", "seed": 0},
                None,
            ),
            {"thinking": False},
            0,
            candidate={"cohort_id": "groups", "candidate_id": 1},
        )
    )
    command = result["args"]
    for removed in (
        "--workflow",
        "--prompt",
        "--observation-mode",
        "--perception-mode",
        "--world-cord",
        "--position-unit",
    ):
        assert removed not in command
    assert command[command.index("--remote-toolkit-endpoint") + 1] == "http://robotwin"
    assert "--sam3-endpoint" not in command
    output = Path(launcher.cfg.tools.rpent.output_dir) / "general"
    assert json.loads((output / "request.json").read_text())["task"] == {
        "environment": "robotwin",
        "task_name": "adjust_bottle",
        "seed": 0,
    }
    assert (
        json.loads((output / "request.json").read_text())["planner"]["thinking"]
        is False
    )


@pytest.mark.parametrize("version", [7, None])
def test_episode_keeps_synchronous_version_checks_and_training_turns(
    make_agent, version
):
    fake = FakeRPent(turns=1)
    worker = make_agent(fake)
    worker.cfg.tools.rpent.update(
        {
            "image_size": 1024,
            "remote_toolkit_endpoint": "http://127.0.0.1:28490",
            "max_actions_per_call": 3,
            "reasoning_history": -1,
            "image_history": -1,
        }
    )
    worker.cfg.runner.synchronous = True
    worker.cfg.algorithm.sampling_params.update(
        {
            "temperature": 1,
            "top_p": 1,
            "top_k": -1,
            "min_p": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1,
        }
    )
    generate = worker.generate

    async def versioned(*args, **kwargs):
        result = await generate(*args, **kwargs)
        if version is not None:
            result.update(version_start=version, version_end=version)
        return result

    worker.generate = versioned
    if version is None:
        with pytest.raises(ValueError, match="installed weight version"):
            run_query(worker)
        return
    result = run_query(worker)
    assert result.extra_fields["turns"] == 1
    assert set(result.single_turn_outputs[0].response_weight_versions) == {7}
    audits = list(
        Path(worker.cfg.tools.rpent.output_dir).glob("*/training_audit/*.json")
    )
    assert len(audits) == 1
    audit = json.loads(audits[0].read_text())
    assert audit["weight_version_start"] == audit["weight_version_end"] == 7
