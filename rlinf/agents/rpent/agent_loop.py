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

"""In-process RPent multi-turn agent loop with synchronous oversampling."""

from __future__ import annotations

import asyncio
import json
import struct
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from omegaconf import DictConfig, OmegaConf

from rlinf.agents.rpent.history import (
    AppendOnlyHistory,
    CompactionConfig,
    EpisodeConversation,
    merge_prefix_turns,
)
from rlinf.agents.rpent.simulators import episode_task, mixed_simulators
from rlinf.agents.tokenization import get_agentic_tokenizer
from rlinf.config import release_image_replay_enabled
from rlinf.data.schema.reasoning_requests import RolloutRequest
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.scheduler import Channel
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import (
    AgentLoopOutput,
    MultiAgentLoopOutput,
    MultiAgentLoopWorker,
)

_THINK_END = "</think>"
_RETRY_KINDS = ("env_setup", "model_transport", "tool_transport")
_MIN_RETRIED_EPISODES = 3
_BINARY_EXECUTE_MEDIA_TYPE = "application/vnd.rpent.execute+octet-stream"


def _unpack_execute_response(payload: bytes) -> tuple[dict[str, Any], list[bytes]]:
    """Unpack binary /sessions/{sid}/execute payload into (meta, raw_png_bytes)."""
    mv = memoryview(payload)
    if len(mv) < 8:
        raise ValueError("Truncated binary execute response")
    (meta_len,) = struct.unpack_from("!I", mv, 0)
    pos = 4 + meta_len
    meta = json.loads(bytes(mv[4:pos]).decode("utf-8"))
    (n_images,) = struct.unpack_from("!I", mv, pos)
    pos += 4
    images: list[bytes] = []
    for _ in range(n_images):
        (img_len,) = struct.unpack_from("!I", mv, pos)
        pos += 4
        images.append(bytes(mv[pos : pos + img_len]))
        pos += img_len
    return meta, images


class RPentEpisodeError(RuntimeError):
    """Typed infrastructure error eligible for episode-level retry."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


class RPentAgentLoopWorker(MultiAgentLoopWorker):
    """Run RPent embodied episodes directly in-process over SimPoolServer HTTP."""

    def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement):
        super().__init__(cfg, placement)
        self.return_logprobs = (
            cfg.runner.task_type != "reasoning_eval" and cfg.rollout.return_logprobs
        )
        template_kwargs = cfg.agentloop.get("chat_template_kwargs")
        self._thinking_enabled = bool(
            (template_kwargs or {}).get("enable_thinking", True)
        )
        if self.tokenizer.eos_token_id is None:
            raise ValueError("RPent chat tokenizer must define an EOS token")
        self.agentic_tokenizer = get_agentic_tokenizer(
            self.tokenizer,
            cfg.agentloop.agentic_tokenizer,
            OmegaConf.to_container(template_kwargs, resolve=True)
            if template_kwargs
            else None,
        )
        self._compaction = CompactionConfig.from_cfg(
            OmegaConf.to_container(
                cfg.agentloop.get("history_compaction"), resolve=True
            )
            if cfg.agentloop.get("history_compaction") is not None
            else None
        )
        self._simulators = mixed_simulators(cfg)
        self._default_endpoint = (
            str(cfg.tools.rpent.remote_toolkit_endpoint).rstrip("/")
            if cfg.tools.rpent.get("remote_toolkit_endpoint")
            else None
        )
        ep_limit = cfg.tools.rpent.get("episodes_per_worker", None)
        self._episode_sem: asyncio.Semaphore | None = (
            asyncio.Semaphore(int(ep_limit))
            if ep_limit is not None and int(ep_limit) > 0
            else None
        )
        self._http: httpx.AsyncClient | None = None
        self._cohort: dict[str, Any] | None = None

        self.extra_keys_turn = ["turn_id", "turn_ids", "tool_names"]
        self.extra_keys_traj = [
            "success",
            "truncated",
            "task",
            "environment",
            "init_state_id",
            "terminal",
            "termination_reason",
            "episode_id",
            "elapsed_s",
            "turns",
            "infra_retries",
            "infra_retry_errors",
        ]

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            read_timeout = float(
                self.cfg.tools.rpent.get(
                    "tool_timeout_s", self.cfg.agentloop.get("episode_timeout_s", 3600)
                )
            )
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=30.0, read=read_timeout, write=60.0, pool=60.0
                ),
                limits=httpx.Limits(
                    max_connections=512,
                    max_keepalive_connections=128,
                    keepalive_expiry=30.0,
                ),
                trust_env=False,
            )
        return self._http

    def _endpoint_for(self, task: dict[str, Any]) -> str:
        env = task.get("environment")
        if env in self._simulators:
            return self._simulators[env].endpoint
        if self._default_endpoint:
            return self._default_endpoint
        raise ValueError(
            f"No SimPoolServer endpoint configured for environment {env!r}"
        )

    def _output_budget(self, prompt_len: int) -> int:
        seq_limit = int(self.cfg.runner.seq_length)
        max_new = int(self.cfg.algorithm.sampling_params.max_new_tokens)
        return min(max_new, seq_limit - prompt_len)

    async def _open_session(
        self, client: httpx.AsyncClient, endpoint: str, task: dict[str, Any]
    ) -> dict[str, Any]:
        """Open a simulator session, falling back across ``fallback_seeds`` if needed."""
        rpent = self.cfg.tools.rpent
        base_seed = task.get("seed", task.get("init_state_id"))
        seeds = [base_seed, *(task.get("fallback_seeds") or ())]
        last_error: Exception | None = None
        for seed in seeds:
            attempt_task = dict(task)
            if seed is not None:
                attempt_task["seed"] = seed
                if "init_state_id" in attempt_task:
                    attempt_task["init_state_id"] = seed
            try:
                resp = await client.post(
                    f"{endpoint}/sessions/open",
                    json={
                        "task": attempt_task,
                        "cohort_id": self._generation_cohort_id,
                        "max_episode_steps": int(rpent.get("max_episode_steps", 10000)),
                        "image_size": int(rpent.get("image_size", 1024)),
                        "max_actions_per_call": int(
                            rpent.get("max_actions_per_call", 3)
                        ),
                        "reasoning": self._thinking_enabled,
                        "record_artifacts": bool(rpent.get("record_artifacts", False)),
                        "output_dir": str(rpent.get("output_dir"))
                        if rpent.get("record_artifacts", False)
                        and rpent.get("output_dir")
                        else None,
                    },
                )
                if resp.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code} on {endpoint}/sessions/open: {resp.text[:300]}",
                        request=resp.request,
                        response=resp,
                    )
                return resp.json()
            except (httpx.HTTPError, OSError) as exc:
                last_error = exc
        raise RPentEpisodeError(
            f"SimPoolServer open failed for {task.get('task_name') or task.get('suite')}: {last_error}",
            kind="env_setup",
        ) from last_error

    async def _close_session(
        self, client: httpx.AsyncClient, endpoint: str, session_id: str
    ) -> None:
        try:
            await client.post(
                f"{endpoint}/sessions/{session_id}/close", json={}, timeout=30.0
            )
        except Exception as exc:
            self.log_warning(f"Failed closing session {session_id}: {exc!r}")

    async def _parse_response(
        self, response_ids: list[int], tools: list[dict] | None, turn_idx: int
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Decode sampled tokens into text, parsed tool calls, and assistant wire dict."""
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        answer = response_text.removesuffix(self.tokenizer.eos_token or "")
        reasoning, found, after = answer.partition(_THINK_END)
        if not self._thinking_enabled:
            found = ""
        if found:
            answer = after
        else:
            reasoning = ""
        content, requests = await self.toolcall_parser(answer, tools=tools)
        calls = [
            {
                "id": f"call_{turn_idx}_{idx}",
                "name": item.name,
                "arguments": item.arguments,
            }
            for idx, item in enumerate(requests)
        ]
        tool_calls_wire = [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for call in calls
        ]
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content.strip() or None,
        }
        if found:
            message["reasoning_content"] = reasoning.strip()
        if tool_calls_wire:
            message["tool_calls"] = tool_calls_wire
        return response_text, calls, message

    async def _sample_turn(
        self,
        *,
        episode_id: str,
        turn_idx: int,
        conv: EpisodeConversation,
        history: AppendOnlyHistory,
        weight_version: int | None,
    ) -> tuple[
        AgentLoopOutput | None,
        list[dict[str, Any]],
        dict[str, Any],
        str,
        int | None,
    ]:
        """Render prompt tokens, run SGLang generation with image capture, and parse."""
        max_new = int(self.cfg.algorithm.sampling_params.max_new_tokens)
        seq_limit = int(self.cfg.runner.seq_length)
        release_capture = release_image_replay_enabled(self.cfg.rollout)
        store_dir = (
            str(Path(self.cfg.tools.rpent.output_dir) / episode_id / "training_images")
            if self.cfg.tools.rpent.get("store_images_on_disk", False)
            and self.cfg.tools.rpent.get("output_dir")
            else None
        )

        for overflow_attempt in range(2):
            messages = conv.wire_messages()
            conv_images = conv.images()
            with history.request():
                rendered = await asyncio.to_thread(
                    history.render_turn, messages, conv.tools, conv_images
                )
                prompt_ids, images = rendered.input_ids, rendered.images
                budget = self._output_budget(len(prompt_ids))
                if budget < max_new:
                    if overflow_attempt == 0 and conv.maybe_compact(
                        current_tokens=seq_limit
                    ):
                        continue
                    return None, [], {}, "context_length_exceeded", weight_version

                sampling_params = {
                    "max_new_tokens": budget,
                    "stop_token_ids": [self.tokenizer.eos_token_id],
                }
                capture_images = None
                if release_capture and images:
                    capture_images = {
                        "image_spans": rendered.image_spans,
                        "history": history.image_history,
                        "token_budget": (
                            seq_limit,
                            int(self.cfg.algorithm.sampling_params.max_new_tokens),
                        ),
                        "store_dir": store_dir,
                    }
                try:
                    generated = await self.generate(
                        prompt_ids,
                        sampling_params=sampling_params,
                        image_data=images or None,
                        capture_images=capture_images,
                    )
                except RuntimeError as exc:
                    msg = str(exc)
                    if "synchronous_oversample_drop" in msg:
                        raise asyncio.CancelledError(msg) from exc
                    if (
                        "context_length_exceeded" in msg
                        and overflow_attempt == 0
                        and conv.maybe_compact(current_tokens=seq_limit)
                    ):
                        continue
                    if "context_length_exceeded" in msg:
                        return None, [], {}, "context_length_exceeded", weight_version
                    raise RPentEpisodeError(
                        f"Rollout generate failed: {msg}", kind="model_transport"
                    ) from exc

                multimodal = None
                if release_capture and images:
                    multimodal = generated["multimodal_input"]
                    prompt_ids = history.accept_capture(multimodal)
                    history.image_history = generated["image_history"]
                else:
                    generated["prompt_token_ids"] = prompt_ids

                if self.cfg.runner.get("synchronous", True):
                    ver = generated.get("version_start")
                    if ver is None or ver != generated.get("version_end"):
                        raise RuntimeError(
                            "Synchronous RPent requires constant weight version across a turn"
                        )
                    if weight_version is not None and weight_version != ver:
                        raise RuntimeError(
                            f"Policy weight version drifted mid-episode: {weight_version} -> {ver}"
                        )
                    weight_version = ver
                    weight_versions = [ver] * len(generated["output_ids"])
                else:
                    weight_versions = None

                response_ids = generated["output_ids"]
                response_text, calls, message = await self._parse_response(
                    response_ids, conv.tools, turn_idx
                )
                history.record(messages, conv.tools, prompt_ids, response_ids, message)

                turn_output = AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_text=response_text,
                    response_logprobs=generated.get("logprobs"),
                    response_weight_versions=weight_versions,
                    response_images=multimodal,
                    version_start=generated.get("version_start"),
                    version_end=generated.get("version_end"),
                    is_end=generated.get("finish_reason") == "stop",
                    extra_fields={
                        "turn_id": turn_idx,
                        "tool_names": [call["name"] for call in calls],
                        "chain_id": history.chain_id,
                    },
                )
                finish = generated.get("finish_reason", "stop")
                return turn_output, calls, message, finish, weight_version

        return None, [], {}, "context_length_exceeded", weight_version

    async def _run_episode(self, task: dict[str, Any]) -> MultiAgentLoopOutput:
        episode_id = uuid4().hex
        started = time.monotonic()
        endpoint = self._endpoint_for(task)
        client = self._client()
        max_turns = int(self.cfg.agentloop.max_turns)
        episode_timeout_s = float(self.cfg.agentloop.get("episode_timeout_s", 7200))

        async with (
            self._episode_sem if self._episode_sem is not None else nullcontext()
        ):
            opened = await self._open_session(client, endpoint, task)
            sid = opened["session_id"]
            conv = EpisodeConversation(
                system_prompt=opened["system_prompt"],
                user_message=opened["user_message"],
                initial_observation_text=opened["initial_observation"],
                tools=opened["tools"],
                environment=opened["environment"],
                compaction=self._compaction,
            )
            history = AppendOnlyHistory(self.agentic_tokenizer)
            turns: list[AgentLoopOutput] = []
            weight_version: int | None = None
            done = bool(opened["done"])
            success = bool(opened["success"])
            truncated = bool(opened["truncated"])
            reason = "success" if success else "planner_completed"

            try:
                while not done and len(turns) < max_turns:
                    if time.monotonic() - started > episode_timeout_s:
                        reason = "episode_timeout"
                        break
                    conv.maybe_compact(current_tokens=history.token_length or None)
                    (
                        turn_output,
                        calls,
                        message,
                        finish,
                        weight_version,
                    ) = await self._sample_turn(
                        episode_id=episode_id,
                        turn_idx=len(turns),
                        conv=conv,
                        history=history,
                        weight_version=weight_version,
                    )
                    if turn_output is None:
                        reason = finish
                        break
                    turns.append(turn_output)
                    conv.append_assistant(
                        content=message.get("content"),
                        reasoning_content=message.get("reasoning_content"),
                        tool_calls=message.get("tool_calls"),
                    )
                    if finish == "length":
                        reason = "response_truncated"
                        break
                    if not calls:
                        reason = "planner_completed"
                        break

                    err_kind, _err_msg, args, should_retry = conv.validate_or_retry(
                        calls
                    )
                    if err_kind is not None:
                        if should_retry:
                            continue
                        reason = err_kind
                        break

                    try:
                        exec_resp = await client.post(
                            f"{endpoint}/sessions/{sid}/execute",
                            json={"name": calls[0]["name"], "arguments": args},
                            headers={"Accept": _BINARY_EXECUTE_MEDIA_TYPE},
                        )
                        exec_resp.raise_for_status()
                        if _BINARY_EXECUTE_MEDIA_TYPE in exec_resp.headers.get(
                            "content-type", ""
                        ):
                            executed, images = _unpack_execute_response(
                                exec_resp.content
                            )
                        else:
                            executed = exec_resp.json()
                            images = []
                    except (httpx.HTTPError, OSError, ValueError) as exc:
                        raise RPentEpisodeError(
                            f"Tool {calls[0]['name']} failed on {endpoint}: {exc}",
                            kind="tool_transport",
                        ) from exc

                    if executed.get("infrastructure_error"):
                        raise RPentEpisodeError(
                            str(executed["infrastructure_error"]),
                            kind="tool_transport",
                        )
                    conv.append_tool_result(
                        tool_name=calls[0]["name"],
                        tool_call_id=calls[0]["id"],
                        text=executed["text"],
                        images=images,
                    )
                    success = bool(executed["success"])
                    truncated = bool(executed["truncated"])
                    done = bool(executed["done"])
                else:
                    if not done and len(turns) >= max_turns:
                        reason = "request_limit"
                    elif success:
                        reason = "success"
                    elif truncated:
                        reason = "environment_step_limit"
            finally:
                await self._close_session(client, endpoint, sid)

        if not turns:
            raise RPentEpisodeError(
                f"Episode {episode_id} ended with 0 sampled turns ({reason})",
                kind="env_setup",
            )

        reward = float(success)
        for turn in turns:
            turn.reward_score = reward
        max_scored = int(
            self.cfg.agentloop.get(
                "max_scored_tokens", self.cfg.algorithm.sampling_params.max_new_tokens
            )
        )
        output_turns = (
            merge_prefix_turns(turns, max_scored)
            if self.cfg.agentloop.get("merge_prefix_turns", True)
            else turns
        )
        return MultiAgentLoopOutput(
            single_turn_outputs=output_turns,
            extra_fields={
                "success": success,
                "truncated": truncated,
                "task": task,
                "environment": opened["environment"],
                "init_state_id": opened.get("init_state_id"),
                "terminal": True,
                "termination_reason": "success" if success else reason,
                "episode_id": episode_id,
                "elapsed_s": time.monotonic() - started,
                "turns": len(turns),
            },
        )

    async def run_one_query(
        self, prompt_ids: list[int], answer: list[str]
    ) -> MultiAgentLoopOutput:
        task = episode_task(answer)
        retries = int(self.cfg.agentloop.get("max_infra_retries", 1))
        errors: list[dict[str, str]] = []
        for attempt in range(retries + 1):
            try:
                result = await self._run_episode(task)
            except RPentEpisodeError as exc:
                errors.append(
                    {"type": type(exc).__name__, "kind": exc.kind, "message": str(exc)}
                )
                if attempt == retries:
                    raise
                self.log_warning(
                    f"Retrying RPent infrastructure failure ({attempt + 1}/{retries}): {exc!r}"
                )
                continue
            result.extra_fields["infra_retries"] = attempt
            result.extra_fields["infra_retry_errors"] = errors
            return result
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------
    # Synchronous oversampling orchestration (owned here, not in base)
    # ------------------------------------------------------------------

    async def run_agentloop_rollout(
        self, input_channel: Channel, output_channel: Channel
    ):
        request: RolloutRequest = await input_channel.get(async_op=True).async_wait()
        if request.cohort_id is None:
            await input_channel.put(request, async_op=True).async_wait()
            return await super().run_agentloop_rollout(input_channel, output_channel)
        return await self._run_synchronous_candidates(request, output_channel)

    async def _run_synchronous_candidates(
        self, request: RolloutRequest, candidate_channel: Channel
    ) -> dict[str, Any]:
        cohort_id = request.cohort_id
        group_ids = request.group_ids
        if group_ids is None or len(group_ids) != len(request.input_ids):
            raise ValueError("Oversampled rollout request must specify group_ids")

        state: dict[str, Any] = {
            "id": cohort_id,
            "closed": asyncio.Event(),
            "tasks": {},
            "results": {},
            "records": {},
        }
        self._cohort = state
        self._generation_cohort_id = cohort_id

        async def _run_candidate(
            candidate_id: int, prompt_ids: list[int], answer: list[str]
        ) -> None:
            task_info = episode_task(answer)
            env_name = task_info.get("environment")
            try:
                traj = await self.run_one_query(prompt_ids, answer)
                extra_turn, extra_traj, extra_group, extra_train = (
                    self.gen_extra_fields([traj], answer)
                )
                rollout_result = self.get_rollout_result(
                    [traj], extra_turn, extra_traj, extra_group, extra_train
                )
                rollout_result.group_id = candidate_id
                state["results"][candidate_id] = rollout_result
                record = {
                    "cohort_id": cohort_id,
                    "candidate_id": candidate_id,
                    "environment": env_name,
                    "status": "completed",
                    "success": bool(traj.extra_fields["success"]),
                    "turns": int(traj.extra_fields["turns"]),
                }
                state["records"][candidate_id] = record
                if not state["closed"].is_set():
                    await candidate_channel.put(record, async_op=True).async_wait()
            except asyncio.CancelledError:
                state["records"][candidate_id] = {
                    "cohort_id": cohort_id,
                    "candidate_id": candidate_id,
                    "environment": env_name,
                    "status": "cancelled",
                }
                raise
            except BaseException as exc:
                record = {
                    "cohort_id": cohort_id,
                    "candidate_id": candidate_id,
                    "environment": env_name,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                state["records"][candidate_id] = record
                if not state["closed"].is_set():
                    await candidate_channel.put(record, async_op=True).async_wait()

        for prompt_ids, answer, base_id in zip(
            request.input_ids, request.answers, group_ids, strict=True
        ):
            for offset in range(request.n):
                cid = base_id + offset
                state["tasks"][cid] = asyncio.create_task(
                    _run_candidate(cid, prompt_ids, answer)
                )

        await asyncio.gather(*state["tasks"].values(), return_exceptions=True)
        return {"cohort_id": cohort_id, "candidates": len(state["tasks"])}

    async def cancel_candidate_ids(
        self, cohort_id: str, candidate_ids: list[int]
    ) -> dict[str, Any]:
        """Cancel specific unselected candidate IDs early (e.g. when one simulator meets quota)."""
        state = self._cohort
        if state is None or state["id"] != cohort_id:
            return {"cohort_id": cohort_id, "cancelled": 0}
        targets = []
        for cid in candidate_ids:
            task = state["tasks"].get(cid)
            if task is not None and not task.done():
                task.cancel()
                targets.append(task)
        if targets:
            await asyncio.gather(*targets, return_exceptions=True)
        return {"cohort_id": cohort_id, "cancelled": len(targets)}

    async def cancel_rollout_candidates(
        self, cohort_id: str, selected_ids: list[int]
    ) -> dict[str, Any]:
        """Cancel unselected candidates and await their simulator cleanup."""
        state = self._cohort
        if state is None or state["id"] != cohort_id:
            return {"cohort_id": cohort_id, "candidates": []}
        selected = set(selected_ids)
        state["selected"] = selected
        state["closed"].set()
        for cid, task in state["tasks"].items():
            if cid not in selected and not task.done():
                task.cancel()
        await asyncio.gather(*state["tasks"].values(), return_exceptions=True)
        self._generation_cohort_id = None
        return {
            "cohort_id": cohort_id,
            "candidates": list(state["records"].values()),
        }

    async def admit_rollout_candidates(
        self, cohort_id: str, output_channel: Channel | None
    ) -> dict[str, Any]:
        """Emit selected candidate payloads and return admitted rollout metrics."""
        state = self._cohort
        self._cohort = None
        if state is None or state["id"] != cohort_id or output_channel is None:
            return {"admitted_ids": [], "metrics": []}

        selected = state.get("selected", set())
        admitted_ids: list[int] = []
        raw_metrics: list[dict[str, Any]] = []
        for cid in sorted(selected):
            result = state["results"].get(cid)
            if result is None:
                continue
            await output_channel.put(result, async_op=True).async_wait()
            admitted_ids.append(cid)
            raw_metrics.append(self.get_rollout_metrics(result))
        return {"admitted_ids": admitted_ids, "metrics": raw_metrics}

    # ------------------------------------------------------------------
    # Metrics and training metadata
    # ------------------------------------------------------------------

    def gen_extra_fields(self, task_results, answer):
        turn, traj, _group, train = super().gen_extra_fields(task_results, answer)
        group = {"task": episode_task(answer)}
        train["idx_to_sub_traj"] = [
            0 for result in task_results for _ in result.single_turn_outputs
        ]
        return turn, traj, group, train

    def get_rollout_metrics(self, rollout_result: DynamicRolloutResult):
        successes = rollout_result.extra_fields_traj["success"]
        envs = rollout_result.extra_fields_traj["environment"]
        retry_errors = rollout_result.extra_fields_traj["infra_retry_errors"]
        counts = Counter(error["kind"] for errors in retry_errors for error in errors)
        episodes = len(successes)
        metrics: dict[str, Any] = {
            "__mean__/rpent/infra_retry_rate": (sum(map(bool, retry_errors)), episodes),
            "__sum__/rpent/infra_retry_attempts": sum(map(len, retry_errors)),
            "__sum__/rpent/infra_retry_episodes": sum(map(bool, retry_errors)),
            "__mean__/rpent/success_rate": (sum(successes), episodes),
            "__sum__/rpent/episodes": episodes,
            "__mean__/rpent/turns": (
                sum(rollout_result.extra_fields_traj["turns"]),
                episodes,
            ),
            "__mean__/rpent/rows": (rollout_result.num_sequence, episodes),
        }
        for env in sorted(set(envs)):
            env_succ = [s for s, e in zip(successes, envs, strict=True) if e == env]
            metrics[f"__mean__/rpent/{env}/success_rate"] = (
                sum(env_succ),
                len(env_succ),
            )
        for kind in _RETRY_KINDS:
            metrics[f"__mean__/rpent/infra_retries/{kind}"] = (counts[kind], episodes)
        return metrics

    def post_process_metric(
        self, agent_metrics_list: list[dict], *, gather_workers: bool = True
    ) -> dict | None:
        metrics = super().post_process_metric(
            agent_metrics_list, gather_workers=gather_workers
        )
        if not metrics:
            return metrics
        threshold = self.cfg.agentloop.get("max_infra_retry_rate", 1.0)
        if metrics.get("rpent/infra_retry_rate", 0.0) > threshold:
            msg = (
                f"RPent infra retry rate {metrics['rpent/infra_retry_rate']:.2%} "
                f"exceeds {threshold:.2%} "
                f"({metrics['rpent/infra_retry_episodes']} episodes retried)"
            )
            if metrics["rpent/infra_retry_episodes"] >= _MIN_RETRIED_EPISODES:
                raise RuntimeError(msg)
            self.log_warning(msg)
        return metrics
