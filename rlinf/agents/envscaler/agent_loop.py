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

import asyncio
import copy
from collections.abc import Awaitable, Callable
from typing import Any

from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import (
    AgentLoopOutput,
    MultiAgentLoopOutput,
    MultiAgentLoopWorker,
)


class EnvScalerRollout:
    """Train on independent EnvScaler episodes with per-turn policy outputs."""

    def __init__(
        self, cfg: DictConfig, tokenizer: PreTrainedTokenizerBase, return_logprobs: bool
    ):
        from agent.system_prompt_util import (
            conversational_system_prompt,
            non_conversational_system_prompt,
        )
        from envscaler_env import EnvScalerConvRLEnv, EnvScalerNonConvRLEnv

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.return_logprobs = return_logprobs
        self.catalogs = {}
        self.env_kwargs = {"mode": "train", "env_items_path": cfg.data.environment_path}
        mode = cfg.agentloop.mode
        if mode == "conversational":
            self.env_cls = EnvScalerConvRLEnv
            self.system_prompt = conversational_system_prompt
            if not cfg.agentloop.user_model:
                raise ValueError(
                    "Conversational EnvScaler requires agentloop.user_model."
                )
            self.env_kwargs.update(
                user_model=cfg.agentloop.user_model, provider="openai"
            )
        elif mode == "non_conversational":
            self.env_cls = EnvScalerNonConvRLEnv
            self.system_prompt = non_conversational_system_prompt
        else:
            raise ValueError(
                f"Expected conversational or non_conversational, got {mode!r}"
            )
        if (
            cfg.agentloop.max_turns < 1
            or cfg.algorithm.sampling_params.max_new_tokens < 1
        ):
            raise ValueError("Expected positive max_turns and max_new_tokens")

    def _new_episode(self, task: dict):
        path = task["path"]
        if path not in self.catalogs:
            self.catalogs[path] = self.env_cls(task_items_path=path, **self.env_kwargs)
        # reset() replaces all episode state. Only the read-only catalogs are shared.
        env = copy.copy(self.catalogs[path])
        if hasattr(env, "user_agent"):
            env.user_agent = copy.copy(env.user_agent)
        if env.task_items[task["index"]]["task_id"] != task["task_id"]:
            raise ValueError(f"Scenario file changed after dataset creation: {path}")
        return env

    async def run(
        self,
        prompt_ids: list[int],
        answer: dict,
        generate: Callable[..., Awaitable[dict[str, Any]]],
    ) -> MultiAgentLoopOutput:
        """Run one scenario until the environment finishes or the rollout budget expires."""
        from envscaler_env.utils.parse_util import parse_action, parse_response

        env = self._new_episode(answer)
        first_user, info = await asyncio.to_thread(
            env.reset, task_index=answer["index"]
        )
        if not first_user:
            raise RuntimeError("EnvScaler returned an empty initial user message")
        tools = info["tools"]
        allowed_actions = {schema["function"]["name"] for schema in tools}
        allowed_actions.add("chat_with_user")
        instructions = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        messages = [
            {"role": "system", "content": self.system_prompt + "\n" + instructions},
            {"role": "user", "content": first_user},
        ]
        turns = []
        reward = 0.0
        for _ in range(self.cfg.agentloop.max_turns):
            prompt = self.tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=True
            )
            turn_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            budget = min(
                self.cfg.runner.seq_length - len(turn_ids),
                self.cfg.algorithm.sampling_params.max_new_tokens,
            )
            if budget <= 0:
                if not turns:
                    raise ValueError(
                        f"Initial EnvScaler prompt for {answer['task_id']} has {len(turn_ids)} tokens; increase runner.seq_length"
                    )
                break
            generation = await generate(
                turn_ids, sampling_params={"max_new_tokens": budget}
            )
            ids = generation["output_ids"]
            if not ids:
                raise RuntimeError(
                    f"Rollout returned no tokens for {answer['task_id']}"
                )
            turn = AgentLoopOutput(
                prompt_ids=turn_ids,
                response_ids=ids,
                response_logprobs=generation["logprobs"]
                if self.return_logprobs
                else None,
                response_topk_token_ids=generation.get("topk_token_ids"),
            )
            turns.append(turn)
            if generation.get("finish_reason") == "length":
                break
            text = self.tokenizer.decode(ids, skip_special_tokens=False).removesuffix(
                self.tokenizer.eos_token
            )
            # EnvScaler uses one action per step. Validate against this episode's tool schemas.
            content = text.rsplit("</think>", 1)[-1]
            parsed, reply = parse_response(content)
            valid, action = parse_action(reply) if parsed else (False, {})
            name = action.get("name") if valid else None
            messages.append({"role": "assistant", "content": text})
            if not isinstance(name, str) or name not in allowed_actions:
                messages.append(
                    {
                        "role": "user",
                        "content": "Error: invalid action. Use a listed tool or reply to the user.",
                    }
                )
                continue
            observation, reward, done, truncated, _ = await asyncio.to_thread(
                env.step, reply
            )
            turn.is_end = done
            if done or truncated:
                break
            if not observation["content"] and observation["type"] == "user":
                raise RuntimeError("EnvScaler user simulator returned an empty reply")
            messages.append(
                {"role": observation["type"], "content": observation["content"]}
            )
        for turn in turns:
            turn.reward_score = reward
        return MultiAgentLoopOutput(
            single_turn_outputs=turns,
            trace_prints=messages if self.cfg.agentloop.print_outputs else [],
        )


class EnvScalerAgentLoopWorker(MultiAgentLoopWorker):
    """Collect EnvScaler training turns with the policy rollout backend."""

    def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement):
        super().__init__(cfg, placement)
        self.episodes = EnvScalerRollout(cfg, self.tokenizer, self.return_logprobs)

    async def run_one_query(
        self, prompt_ids: list[int], answer: dict
    ) -> MultiAgentLoopOutput:
        """Generate one independent trajectory for this scenario."""
        return await self.episodes.run(prompt_ids, answer, self.generate)

    def gen_extra_fields(
        self, task_results: list[MultiAgentLoopOutput], answer: dict
    ) -> tuple:
        """Identify the single policy within each GRPO trajectory."""
        if self.is_eval:
            return None, None, answer, {}
        turn_count = sum(len(result.single_turn_outputs) for result in task_results)
        return None, None, None, {"idx_to_sub_traj": [0] * turn_count}
