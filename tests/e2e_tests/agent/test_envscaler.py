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

"""CPU rollout checks using the upstream scenarios and a local Qwen3 tokenizer."""

import asyncio
import json
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from rlinf.agents.envscaler.agent_loop import EnvScalerRollout
from rlinf.data.datasets.reasoning import create_reasoning_datasets

QWEN3_NO_THINKING_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


@pytest.mark.parametrize(
    ("variant", "mode", "source"),
    [("k1_pg", "k1", "sampled"), ("student_top64_bp", "forward_kl_topk", "student")],
)
def test_mopd_math_generation_budget(variant, mode, source):
    from unittest.mock import AsyncMock

    from hydra import compose, initialize_config_dir

    from examples.agent.envscaler.train_mopd import EnvScalerDAPOAgentLoopWorker
    from rlinf.algorithms.distillation.config import build_distillation_plan

    config = Path(__file__).resolve().parents[3] / "examples/agent/envscaler/config"
    with initialize_config_dir(version_base="1.1", config_dir=str(config)):
        cfg = compose(config_name=f"train_qwen3_8b_mopd_{variant}")
    plan = build_distillation_plan(cfg)
    assert (plan.canonical_loss_mode, plan.support_source) == (mode, source)
    assert plan.need_old_logprobs == cfg.rollout.return_logprobs == (mode == "k1")
    assert plan.topk == (64 if source == "student" else None)
    assert cfg.runner.seq_length == cfg.runner.max_tokens_per_mbs == 32768
    teachers = cfg.distillation.teacher_models
    assert teachers.tooluse.inference.max_model_len == 32769
    assert teachers.math.inference.max_model_len == 10241

    worker = object.__new__(EnvScalerDAPOAgentLoopWorker)
    worker.cfg = cfg
    worker.return_logprobs = plan.need_old_logprobs
    generation = {"output_ids": [1]}
    if plan.need_old_logprobs:
        generation["logprobs"] = [-0.5]
    if plan.need_rollout_topk:
        generation["topk_token_ids"] = [list(range(64))]
    worker.generate = AsyncMock(return_value=generation)
    for prompt_length, budget in [(100, 8192), (4096, 6144), (10239, 1)]:
        prompt = [1] * prompt_length
        result = asyncio.run(worker.run_one_query(prompt, ["2"]))
        worker.generate.assert_awaited_with(
            prompt, sampling_params={"max_new_tokens": budget}
        )
        (turn,) = result.single_turn_outputs
        assert turn.response_logprobs == generation.get("logprobs")
        assert turn.response_topk_token_ids == generation.get("topk_token_ids")
    with pytest.raises(ValueError, match="fewer than 10240"):
        asyncio.run(worker.run_one_query([1] * 10240, ["2"]))


def test_envscaler_dapo_mopd_batches(tmp_path):
    from hydra import compose, initialize_config_dir

    from examples.agent.envscaler.train_mopd import create_mopd_datasets
    from rlinf.data.datasets.reasoning.collate_fn import collate_fn
    from rlinf.data.schema.reasoning_requests import build_rollout_requests_from_batch

    config = Path(__file__).resolve().parents[3] / "examples/agent/envscaler/config"
    with initialize_config_dir(version_base="1.1", config_dir=str(config)):
        cfg = compose(config_name="train_qwen3_8b_mopd")
    root = Path(os.environ["ENVSCALER_ROOT"]) / "interact_with_env/envscaler_env/data"
    scenarios = json.loads((root / "rl_scenario_metadata.json").read_text())[:1]
    scenario_path = tmp_path / "scenarios.json"
    scenario_path.write_text(json.dumps(scenarios))
    math_path = tmp_path / "math.jsonl"
    opening = "Solve the following math problem step by step. "
    answer_format = (
        "The last line of your response should be of the form Answer: $Answer."
    )
    reasoning = "You must conduct reasoning inside <think> and </think> firstly. "
    questions = [
        {
            "prompt": [{"role": "user", "content": f"What is {i}+1?"}],
            "reward_model": {"ground_truth": str(i + 1)},
        }
        for i in range(3)
    ]
    questions[0]["prompt"][0]["content"] = opening + answer_format + "\n\nWhat is 0+1?"
    questions[1]["prompt"][0]["content"] = (
        opening + reasoning + answer_format + "\n\nWhat is 1+1?"
    )
    math_path.write_text("\n".join(map(json.dumps, questions)))
    cfg.data.train_data_paths = [str(scenario_path)]
    cfg.data.environment_path = str(root / "191_env_metadata.json")
    cfg.data.math_data_path = str(math_path)
    cfg.data.rollout_batch_size = 2
    cfg.runner.max_steps = 3
    tokenizer = AutoTokenizer.from_pretrained(os.environ["ENVSCALER_TEST_MODEL"])
    dataset, _ = create_mopd_datasets(cfg, tokenizer)

    seen_math = set()
    assert len(dataset) == 6
    for start in range(0, len(dataset), 2):
        batch = collate_fn([dataset[start], dataset[start + 1]])
        (request,) = build_rollout_requests_from_batch(
            batch, group_size=1, split_size=1, teacher_keys=("math", "tooluse")
        )
        assert {group[0].teacher_key for group in request.sample_contexts} == {
            "math",
            "tooluse",
        }
        for ids, answer, group in zip(
            request.input_ids, batch["answer"], request.sample_contexts, strict=True
        ):
            if group[0].teacher_key == "tooluse":
                assert answer["task_id"] == scenarios[0]["task_id"]
                assert answer["path"] == str(scenario_path)
            else:
                prompt = tokenizer.decode(ids)
                assert prompt.count(reasoning) == 1
                assert prompt.count(r"Answer: \boxed{...}") == 1
                assert "Answer: $Answer" not in prompt
                assert f"\n\nWhat is {int(answer[0]) - 1}+1?\n\n" in prompt
                assert prompt.endswith("<|im_end|>\n<|im_start|>assistant\n")
                seen_math.add(answer[0])
    assert seen_math == {"1", "2", "3"}


@pytest.mark.parametrize("conversational", [False, True])
def test_envscaler_rollout(conversational, monkeypatch):
    from envscaler_env.utils import user_agent

    root = Path(os.environ["ENVSCALER_ROOT"])
    data = root / "interact_with_env/envscaler_env/data"
    tokenizer = AutoTokenizer.from_pretrained(os.environ["ENVSCALER_TEST_MODEL"])
    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[3]
        / "examples/agent/envscaler/config/train.yaml"
    )
    cfg.data.environment_path = str(data / "191_env_metadata.json")
    cfg.data.train_data_paths = [str(data / "rl_scenario_metadata.json")]
    cfg.data.data_size = 1
    if conversational:
        cfg.agentloop.mode = "conversational"
        cfg.agentloop.user_model = "test-user"
    else:
        cfg.data.apply_chat_template_kwargs = {"enable_thinking": False}

    def user_reply(messages, **kwargs):
        if len(messages) == 2:
            return "# Thought: Ask for help.\n# Reply: Please cancel my subscription."
        if len(messages) == 4:
            assert messages[-1]["content"].endswith("Which subscription?")
            return "# Thought: Identify it.\n# Reply: Please cancel SUB1001 only."
        assert len(messages) == 6
        return "###STOP###"

    monkeypatch.setattr(user_agent, "llm_inference", user_reply)
    train, _ = create_reasoning_datasets(cfg, tokenizer)
    item = train[0]
    assert item.answer["task_id"] == "env_148_rl-task_1"
    episodes = EnvScalerRollout(cfg, tokenizer, return_logprobs=True)

    async def trajectory(modify):
        responses = ["Which subscription?"] if conversational else []
        if modify:
            responses.append(
                '<tool_call>{"name":"cancel_subscription","arguments":{"subscription_id":"SUB1001"}}</tool_call>'
            )
        responses.append("Task Completed")
        generated = []
        prompts = []

        async def generate(prompt, sampling_params):
            prompts.append(tokenizer.decode(prompt))
            ids = tokenizer.encode(responses[len(generated)], add_special_tokens=False)
            generated.append(ids)
            return {"output_ids": ids, "logprobs": [0.0] * len(ids)}

        result = await episodes.run(
            item.prompt[-item.length :].tolist(), item.answer, generate
        )
        assert [t.response_ids for t in result.single_turn_outputs] == generated
        assert result.single_turn_outputs[-1].is_end
        assert all(
            len(t.response_ids) == len(t.response_logprobs)
            for t in result.single_turn_outputs
        )
        if conversational:
            assert "Please cancel my subscription." in prompts[0]
            assert "Please cancel SUB1001 only." in prompts[1]
            assert (
                "Perform the following two-part administration request"
                not in prompts[0]
            )
        else:
            assert all(p.endswith(QWEN3_NO_THINKING_SUFFIX) for p in prompts)
        if modify:
            assert "Subscription cancelled successfully" in prompts[-1]
        return [turn.reward_score for turn in result.single_turn_outputs]

    async def run_group():
        return await asyncio.gather(trajectory(True), trajectory(False))

    rewards = asyncio.run(run_group())
    assert rewards == [
        [0.4167] * (3 if conversational else 2),
        [0.25] * (2 if conversational else 1),
    ]

    async def truncated_reply(prompt, sampling_params):
        ids = tokenizer.encode("Task Completed", add_special_tokens=False)
        return {
            "output_ids": ids,
            "logprobs": [0.0] * len(ids),
            "finish_reason": "length",
        }

    result = asyncio.run(
        episodes.run(item.prompt[-item.length :].tolist(), item.answer, truncated_reply)
    )
    assert len(result.single_turn_outputs) == 1
    turn = result.single_turn_outputs[0]
    assert turn.reward_score == 0 and not turn.is_end


def test_envscaler_eval_metrics():
    from examples.agent.envscaler.eval import summarize, trajectory_metrics
    from rlinf.data.schema.reasoning_results import DynamicRolloutResult

    result = DynamicRolloutResult(
        num_sequence=5,
        group_size=4,
        idx_to_traj=[0, 0, 1, 2, 3],
        prompt_lengths=[100, 114, 100, 100, 100],
        response_lengths=[7, 3, 8, 6, 5],
        input_ids=[[0] * n for n in [107, 117, 108, 106, 105]],
        is_end=[False, True, True, True, False],
        rewards=[1.0, 1.0, 0.75, 0.0, 0.0],
    )
    rows = trajectory_metrics(result)
    assert rows[0] == {
        "turns": 2,
        "input_tokens": 214,
        "generated_tokens": 10,
        "final_context_tokens": 117,
        "reward": 1.0,
        "passed": True,
    }
    failed = [dict(row, passed=False, reward=0.0) for row in rows]
    summary = summarize([rows, failed], [1, 4])
    assert summary["pass@1"] == 1 / 8
    assert summary["pass@4"] == 1 / 2
    assert summary["turns"]["mean"] == 1.25
    assert summary["generated_tokens"]["mean"] == 7.25
