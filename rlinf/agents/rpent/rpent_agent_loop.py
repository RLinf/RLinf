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

"""Text-only RPent planner episodes served by the RLinf rollout engine."""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path
from uuid import uuid4

from omegaconf import DictConfig, OmegaConf

from rlinf.agents.rpent.episode import Episode, RPentEpisodeError, merge_prefix_turns
from rlinf.agents.rpent.history import AppendOnlyHistory
from rlinf.agents.tokenization import get_agentic_tokenizer
from rlinf.scheduler import Channel
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import (
    AgentLoopOutput,
    MultiAgentLoopOutput,
    MultiAgentLoopWorker,
)
from rlinf.workers.agent.openai_session_server import (
    OpenAISessionServer,
    SessionRequestError,
    chat_completion,
)

_RETRY_KINDS = ("env_setup", "process", "model_transport")
# Exit status of an RPent episode whose environment or shared service did not
# start (robots.libero.rl_episode.EXIT_SETUP_FAILED).
_EXIT_SETUP_FAILED = 75
# A stopped episode process gets this long to release its simulator.
_STOP_GRACE_S = 60
# The planner names this model in its requests; each worker serves one policy.
_MODEL_NAME = "policy"
# A high retry rate stops the update only once this many episodes were retried,
# so a single failure in a small batch does not.
_MIN_RETRIED_EPISODES = 3


def _episode_task(answer: list[str]) -> dict:
    if len(answer) != 1:
        raise ValueError(
            f"an RPent dataset row must specify exactly one task, got {len(answer)}"
        )
    return json.loads(answer[0])


class RPentAgentLoopWorker(MultiAgentLoopWorker):
    """Run RPent planner episodes and keep the turns they sample from the policy.

    Each episode runs RPent's ``robots.libero.rl_episode`` in its own process on
    this worker's node, which hosts the LIBERO simulators. The planner runs
    unchanged in that process and calls this worker's OpenAI endpoint, which
    samples each request from the rollout engine. A request that extends the previous turn is rendered as the
    previous turn's sampled tokens plus the new messages, so consecutive turns
    can be merged into one training row. Only episodes with a task outcome are
    trained, so the turns of a failed episode are dropped with it.
    """

    def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement):
        super().__init__(cfg, placement)
        # Training compares these logprobs with its own recomputation (TIS).
        self.return_logprobs = cfg.rollout.return_logprobs
        if self.tokenizer.eos_token_id is None:
            raise ValueError("RPent chat tokenizer must define an EOS token")
        template_kwargs = cfg.agentloop.get("chat_template_kwargs")
        self.agentic_tokenizer = get_agentic_tokenizer(
            self.tokenizer,
            cfg.agentloop.agentic_tokenizer,
            OmegaConf.to_container(template_kwargs) if template_kwargs else None,
        )
        self.extra_keys_turn = ["turn_id", "turn_ids", "tool_names"]
        self.extra_keys_traj = [
            "success",
            "truncated",
            "task",
            "init_state_id",
            "memory_entries",
            "terminal",
            "termination_reason",
            "episode_id",
            "elapsed_s",
            "turns",
            "infra_retries",
            "infra_retry_errors",
        ]
        # One slot per concurrent episode, naming the GPU its simulator renders on;
        # workers start at different GPUs so that they spread over all of them.
        rpent = cfg.tools.rpent
        self._env_slots = asyncio.Queue()
        for slot in range(rpent.episodes_per_worker):
            index = self._rank * rpent.episodes_per_worker + slot
            self._env_slots.put_nowait(rpent.env_gpus[index % len(rpent.env_gpus)])
        # Episode processes run on this node and reach the endpoint locally.
        self._model_server = OpenAISessionServer(
            self._complete,
            bind_host="127.0.0.1",
            port=0,
            advertise_host="127.0.0.1",
        )

    async def run_agentloop_rollout(
        self, input_channel: Channel, output_channel: Channel | list[Channel]
    ):
        # The endpoint must serve on this worker's event loop, which first runs here.
        await self._model_server.start()
        return await super().run_agentloop_rollout(input_channel, output_channel)

    async def run_one_query(self, prompt_ids: list[int], answer: list[str]):
        """Run one episode, retrying service failures with fresh episode IDs."""
        task = _episode_task(answer)
        retries = self.cfg.agentloop.max_infra_retries
        errors = []
        for attempt in range(retries + 1):
            try:
                result = await self._run_episode(task)
            except RPentEpisodeError as error:
                errors.append(
                    {
                        "type": type(error).__name__,
                        "kind": error.kind,
                        "message": str(error),
                    }
                )
                if attempt == retries:
                    raise
                self.log_warning(
                    f"Retrying RPent infrastructure failure "
                    f"({attempt + 1}/{retries}): {error!r}"
                )
                continue
            result.extra_fields["infra_retries"] = attempt
            result.extra_fields["infra_retry_errors"] = errors
            return result

    async def _run_episode(self, task: dict) -> MultiAgentLoopOutput:
        gpu = await self._env_slots.get()
        try:
            episode = Episode(
                uuid4().hex, task, AppendOnlyHistory(self.agentic_tokenizer)
            )
            base_url = self._model_server.open(episode.episode_id, episode)
            try:
                outcome = await self._run_rpent(episode, base_url, gpu)
            finally:
                error = await self._model_server.close(episode.episode_id)
            if error is not None:
                # The planner only saw this sampling failure as HTTP 500.
                raise error
        finally:
            self._env_slots.put_nowait(gpu)
        self._check_planner_done(episode, outcome)
        return self._episode_output(episode, outcome)

    async def _run_rpent(self, episode: Episode, base_url: str, gpu: int) -> dict:
        """Run the episode's RPent process and return the outcome it wrote."""
        rpent = self.cfg.tools.rpent
        output = Path(rpent.output_dir) / episode.episode_id
        output.mkdir(parents=True)
        request = {"task": episode.task, "planner": self._planner_options(base_url)}
        (output / "request.json").write_text(json.dumps(request))
        command = self._rpent_command(output, gpu)
        log = output / "episode.log"
        with log.open("wb") as log_file:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=log_file, stderr=asyncio.subprocess.STDOUT
            )
            try:
                returncode = await asyncio.wait_for(
                    process.wait(), rpent.process_timeout_s
                )
            except asyncio.TimeoutError:
                raise RPentEpisodeError(
                    f"RPent episode {episode.episode_id} ran past "
                    f"tools.rpent.process_timeout_s; see {log}",
                    kind="process",
                ) from None
            finally:
                await _stop(process)
        if returncode == _EXIT_SETUP_FAILED:
            raise RPentEpisodeError(
                f"RPent episode {episode.episode_id} did not start; see {log}",
                kind="env_setup",
            )
        if returncode < 0:
            raise RPentEpisodeError(
                f"RPent episode {episode.episode_id} was stopped by signal "
                f"{-returncode}; see {log}",
                kind="process",
            )
        if returncode != 0:
            raise RuntimeError(
                f"RPent episode {episode.episode_id} failed with exit status "
                f"{returncode}; see {log}"
            )
        outcome = json.loads((output / "outcome.json").read_text())
        keep = rpent.keep_episodes == "all" or (
            rpent.keep_episodes == "failures" and not outcome["success"]
        )
        if not keep:
            shutil.rmtree(output)
        return outcome

    def _planner_options(self, base_url: str) -> dict:
        """Planner options of an episode whose policy is served at ``base_url``."""
        agentloop = self.cfg.agentloop
        compaction = agentloop.get("history_compaction")
        return {
            "base_url": base_url,
            "model_name": _MODEL_NAME,
            "max_turns": agentloop.max_turns,
            "max_tokens": self.cfg.algorithm.sampling_params.max_new_tokens,
            "request_timeout_s": agentloop.model_request_timeout_s,
            "episode_timeout_s": agentloop.episode_timeout_s,
            "no_images": True,
            "history_compaction": None
            if compaction is None
            else OmegaConf.to_container(compaction, resolve=True),
        }

    def _rpent_command(self, output: Path, gpu: int) -> list[str]:
        """Command line of an episode process writing to ``output``."""
        rpent = self.cfg.tools.rpent
        command = [
            rpent.python,
            "-m",
            "robots.libero.rl_episode",
            "--output-dir",
            str(output),
            "--request",
            str(output / "request.json"),
            "--vla-endpoint",
            rpent.vla_endpoint,
            "--sam3-endpoint",
            rpent.sam3_endpoint,
            "--libero-type",
            rpent.libero_type,
            "--max-episode-steps",
            str(rpent.max_episode_steps),
            "--cuda-device",
            str(gpu),
            "--tokenizer-path",
            self.cfg.rollout.model.model_path,
        ]
        if rpent.get("frozen_memory_dir") is not None:
            command += ["--frozen-memory-dir", rpent.frozen_memory_dir]
        return command

    @staticmethod
    def _check_planner_done(episode: Episode, status: dict) -> None:
        """Reject a planner exit that has no task outcome."""
        if status["terminal"]:
            return
        if status.get("error_kind") == "model_transport":
            raise RPentEpisodeError(
                f"RPent model request failed: {status.get('error')}",
                kind="model_transport",
            )
        # Unknown planner failures must not select a replacement trajectory.
        raise RuntimeError(
            f"RPent planner of episode {episode.episode_id} stopped without an "
            f"outcome ({status.get('error_kind')}): {status.get('error')}"
        )

    async def _complete(self, episode: Episode, request: dict) -> dict:
        """Sample one planner request, record the turn, and return its completion."""
        turn_idx = len(episode.turns)
        prompt_ids = await asyncio.to_thread(episode.history.render, request)
        generated = await self.generate(
            prompt_ids,
            sampling_params={
                "max_new_tokens": self._output_budget(len(prompt_ids)),
                # SGLang skips tokenizer initialization for token-ID requests,
                # and the model config's EOS can differ from the chat EOS.
                "stop_token_ids": [self.tokenizer.eos_token_id],
            },
        )
        self._check_generation(generated, prompt_ids)
        response_ids = generated["output_ids"]
        response_text, calls, message = await self._parse_response(
            response_ids, request.get("tools"), turn_idx
        )
        episode.history.record(request, prompt_ids, response_ids, message)
        episode.turns.append(
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_text=response_text,
                response_logprobs=generated.get("logprobs"),
                is_end=generated["finish_reason"] == "stop",
                extra_fields={
                    "turn_id": turn_idx,
                    "tool_names": [call["name"] for call in calls],
                },
            )
        )
        if generated["finish_reason"] == "length":
            finish_reason = "length"
        else:
            finish_reason = "tool_calls" if calls else "stop"
        return chat_completion(
            f"rpent-{episode.episode_id}-{turn_idx}",
            request["model"],
            message,
            finish_reason,
            len(prompt_ids),
            len(response_ids),
        )

    def _output_budget(self, prompt_tokens: int) -> int:
        """Return the turn's output budget within the training sequence length."""
        seq_length = self.cfg.runner.seq_length
        max_new_tokens = min(
            self.cfg.algorithm.sampling_params.max_new_tokens,
            seq_length - prompt_tokens,
        )
        reserved = self.cfg.agentloop.minimum_output_tokens
        if max_new_tokens < reserved:
            raise SessionRequestError(
                400,
                "context_length_exceeded",
                f"the request has {prompt_tokens} input tokens; cannot reserve "
                f"{reserved} output tokens in runner.seq_length={seq_length}",
            )
        return max_new_tokens

    def _check_generation(self, generated: dict, prompt_ids: list[int]) -> None:
        """Reject a response that cannot be trained against its behavior policy."""
        response_ids = generated["output_ids"]
        if not response_ids:
            raise RuntimeError("rollout returned an empty response")
        if generated["prompt_tokens"] != len(prompt_ids):
            raise ValueError(
                f"rollout read {generated['prompt_tokens']} prompt tokens, "
                f"training has {len(prompt_ids)}"
            )
        logprobs = generated.get("logprobs")
        if self.return_logprobs and (
            logprobs is None
            or len(logprobs) != len(response_ids)
            or not all(math.isfinite(value) for value in logprobs)
        ):
            raise ValueError("each sampled token requires a finite behavior logprob")

    async def _parse_response(
        self, response_ids: list[int], tools: list | None, turn_idx: int
    ) -> tuple[str, list[dict], dict]:
        """Decode a response into its text, tool calls and assistant message."""
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        content, requests = await self.toolcall_parser(
            response_text.removesuffix(self.tokenizer.eos_token or ""), tools=tools
        )
        calls = [{"name": item.name, "arguments": item.arguments} for item in requests]
        message = {"role": "assistant", "content": content.strip() or None}
        if calls:
            message["tool_calls"] = [
                {
                    "id": f"call_{turn_idx}_{idx}",
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
                for idx, call in enumerate(calls)
            ]
        return response_text, calls, message

    def _episode_output(self, episode: Episode, status: dict) -> MultiAgentLoopOutput:
        """Score a finished episode's turns with its task outcome."""
        success = status["success"]
        for turn in episode.turns:
            turn.reward_score = float(success)
        turns = episode.turns
        if self.cfg.agentloop.merge_prefix_turns:
            turns = merge_prefix_turns(
                turns,
                self.cfg.agentloop.get(
                    "max_scored_tokens",
                    self.cfg.algorithm.sampling_params.max_new_tokens,
                ),
            )
        return MultiAgentLoopOutput(
            single_turn_outputs=turns,
            extra_fields={
                "success": success,
                "init_state_id": status["init_state_id"],
                "memory_entries": status["memory_entries"],
                "truncated": status["truncated"],
                "task": episode.task,
                "terminal": status["terminal"],
                "termination_reason": status["termination_reason"],
                "episode_id": episode.episode_id,
                "elapsed_s": time.monotonic() - episode.started,
                "turns": len(episode.turns),
            },
        )

    def gen_extra_fields(self, task_results, answer):
        turn, traj, group, train = super().gen_extra_fields(task_results, answer)
        group = {"task": _episode_task(answer)}
        train["idx_to_sub_traj"] = [
            0 for result in task_results for _ in result.single_turn_outputs
        ]
        return turn, traj, group, train

    def get_rollout_metrics(self, rollout_result):
        successes = rollout_result.extra_fields_traj["success"]
        retry_errors = rollout_result.extra_fields_traj["infra_retry_errors"]
        counts = Counter(error["kind"] for errors in retry_errors for error in errors)
        episodes = len(successes)
        metrics = {
            "__mean__/rpent/infra_retry_rate": (sum(map(bool, retry_errors)), episodes),
            "__sum__/rpent/infra_retry_attempts": sum(map(len, retry_errors)),
            "__sum__/rpent/infra_retry_episodes": sum(map(bool, retry_errors)),
            "__mean__/rpent/success_rate": (sum(successes), episodes),
            "__sum__/rpent/episodes": episodes,
            "__mean__/rpent/turns": (
                sum(rollout_result.extra_fields_traj["turns"]),
                episodes,
            ),
            # Rows after consecutive turns are merged.
            "__mean__/rpent/rows": (rollout_result.num_sequence, episodes),
        }
        for kind in _RETRY_KINDS:
            metrics[f"__mean__/rpent/infra_retries/{kind}"] = (counts[kind], episodes)
        return metrics

    def post_process_metric(self, agent_metrics_list: list[dict]) -> dict | None:
        """Stop the update when too many episodes needed infrastructure retries."""
        metrics = super().post_process_metric(agent_metrics_list)
        if not metrics:
            return metrics
        threshold = self.cfg.agentloop.max_infra_retry_rate
        if metrics["rpent/infra_retry_rate"] > threshold:
            message = (
                f"RPent infrastructure retry rate "
                f"{metrics['rpent/infra_retry_rate']:.2%} exceeds {threshold:.2%} "
                f"({metrics['rpent/infra_retry_episodes']} episodes retried)"
            )
            if metrics["rpent/infra_retry_episodes"] >= _MIN_RETRIED_EPISODES:
                raise RuntimeError(message + "; inspect episode failures")
            self.log_warning(
                message + f"; stop threshold is {_MIN_RETRIED_EPISODES} episodes"
            )
        return metrics


async def _stop(process: asyncio.subprocess.Process) -> None:
    """Stop an episode process, giving it time to release its simulator."""
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), _STOP_GRACE_S)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
