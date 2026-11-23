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
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from rlinf.agents.envscaler.agent_loop import EnvScalerRollout
from rlinf.data.datasets.reasoning import create_reasoning_datasets


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
