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

"""Run RPent's native episode process on a policy session."""

import asyncio
import json
from uuid import uuid4

from omegaconf import OmegaConf

from rlinf.agents.rpent.launcher import RPentEpisodeError, RPentEpisodeLauncher
from rlinf.workers.agent.agent_loop import MultiAgentLoopWorker


class RPentAgentLoopWorker(MultiAgentLoopWorker):
    """The simulator owns terminal success. The session server owns training rows."""

    def __init__(self, cfg, placement):
        super().__init__(cfg, placement)
        self._episode_launcher = RPentEpisodeLauncher(cfg)
        self._env_slots = asyncio.Queue()
        env_gpus = cfg.tools.rpent.env_gpus
        for slot in range(cfg.tools.rpent.episodes_per_worker):
            self._env_slots.put_nowait(env_gpus[(self._rank + slot) % len(env_gpus)])

    def get_session_policy(self) -> dict[str, object]:
        return {
            "ignore_sampling_params": True,
            "tool_choice_override": "auto",
        }

    async def run_session_trajectory(self, binding, input_ids, answer):
        base_url = self.session_server_client.session_base_url(
            binding.group_id, binding.session_id
        )
        _, outcome = await self.run_episode(answer[0], base_url)
        return None, {}, int(bool(outcome["success"]))

    async def run_episode(self, task_spec: str, base_url: str):
        task = json.loads(task_spec)
        episode_id = (
            f"{task['suite']}-t{task['task']}-s{task['seed']}-{uuid4().hex[:8]}"
        )
        gpu = await self._env_slots.get()
        try:
            outcome = await self._episode_launcher.run(
                episode_id, task, self._planner_options(base_url), gpu
            )
        finally:
            self._env_slots.put_nowait(gpu)
        if not outcome.get("terminal"):
            raise RPentEpisodeError(
                f"Episode {episode_id} ended without a native task outcome: "
                f"{outcome.get('error')}",
                kind=outcome.get("error_kind") or "planner",
            )
        return {"episode_id": episode_id, "task": task}, outcome

    async def evaluate_batch(self, input_channel, urls):
        request = await input_channel.get(async_op=True).async_wait()
        base_url = urls[self._rank % len(urls)] + "/v1"
        tasks = [
            asyncio.create_task(
                self.run_episode(
                    answer[0] if isinstance(answer, list) else answer, base_url
                )
            )
            for answer in request.answers
            for _ in range(request.n)
        ]
        try:
            return [
                dict(identity, **outcome)
                for identity, outcome in await asyncio.gather(*tasks)
            ]
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _planner_options(self, base_url: str) -> dict:
        """Planner options of an episode whose policy is served at ``base_url``."""
        agentloop = self.cfg.agentloop
        compaction = agentloop.get("history_compaction")
        sampling = self.cfg.algorithm.sampling_params
        return {
            "base_url": base_url,
            "model_name": (
                self.cfg.rollout.model.model_type if self.is_eval else "policy"
            ),
            "max_turns": agentloop.max_turns,
            "max_tokens": sampling.max_new_tokens,
            "request_timeout_s": agentloop.model_request_timeout_s,
            "episode_timeout_s": agentloop.episode_timeout_s,
            "history_compaction": (
                None
                if compaction is None
                else OmegaConf.to_container(compaction, resolve=True)
            ),
            "observation_mode": "adaptive",
            "prompt": agentloop.get("prompt", "simple"),
            "thinking": agentloop.chat_template_kwargs.enable_thinking,
            "sampling_params": {
                name: sampling[name]
                for name in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "presence_penalty",
                    "repetition_penalty",
                )
                if name in sampling
            },
        }
