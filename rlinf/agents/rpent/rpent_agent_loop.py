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

"""RPent planner episodes with token-preserving synchronous RLinf sampling."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import Counter, defaultdict
from contextvars import ContextVar
from pathlib import Path
from uuid import uuid4

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.agents.rpent.cohort import rpc
from rlinf.agents.rpent.episode import (
    Episode,
    GroupStartup,
    RPentEpisodeError,
    episode_output,
    episode_task,
)
from rlinf.agents.rpent.history import (
    AppendOnlyHistory,
    request_has_images,
)
from rlinf.agents.rpent.launcher import RPentEpisodeLauncher
from rlinf.agents.rpent.model_server import (
    OpenAISessionServer,
    SessionRequestError,
    chat_completion,
)
from rlinf.agents.rpent.records import write_training_turn
from rlinf.agents.rpent.thinking_budget import (
    ThinkingBudget,
    serialized_thinking_processor,
)
from rlinf.agents.tokenization import get_agentic_tokenizer
from rlinf.config import (
    release_image_replay_enabled,
    shared_multi_modal_processor_enabled,
)
from rlinf.scheduler import Channel
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import (
    AgentLoopOutput,
    MultiAgentLoopOutput,
    MultiAgentLoopWorker,
)

_GROUP_STARTUP: ContextVar[GroupStartup | None] = ContextVar(
    "rpent_group_startup", default=None
)

_RETRY_KINDS = ("env_setup", "process", "model_transport", "tool_transport")
# The planner names this model in its requests; each worker serves one policy.
_MODEL_NAME = "policy"
# Closes the reasoning the generation prompt opened; the answer follows it.
_THINK_END = "</think>"
# A high retry rate stops the update only once this many episodes were retried,
# so a single failure in a small batch does not.
_MIN_RETRIED_EPISODES = 3


class RPentAgentLoopWorker(MultiAgentLoopWorker):
    """Run RPent planner episodes and keep the turns they sample from the policy.

    Each episode runs its planner in a local process and delegates the
    simulator and tools to the remote backend. The planner
    calls this worker's OpenAI endpoint, which
    samples each request from the rollout engine. A request that extends the previous turn is rendered as the
    previous turn's sampled tokens plus the new messages, so consecutive turns
    can be merged into one training row. Only episodes with a task outcome are
    trained, so the turns of a failed episode are dropped with it.
    """

    def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement):
        super().__init__(cfg, placement)
        # Training compares these logprobs with its own recomputation (TIS).
        self.return_logprobs = cfg.rollout.return_logprobs
        self._thinking_limit = cfg.agentloop.get("thinking_token_budget")
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
            "terminal",
            "termination_reason",
            "episode_id",
            "elapsed_s",
            "turns",
            "infra_retries",
            "infra_retry_errors",
        ]
        # Mixed backends choose their own render GPUs when an agent starts an episode.
        rpent = cfg.tools.rpent
        self._env_slots = asyncio.Queue()
        gpus = [None] if (cfg.get("env") or {}).get("rpent") else rpent.env_gpus
        for slot in range(rpent.episodes_per_worker):
            self._env_slots.put_nowait(gpus[(self._rank + slot) % len(gpus)])
        self._launcher = RPentEpisodeLauncher(cfg)
        # Loaded when view_env first returns images.
        self._image_processor = None
        self._environment_ready: dict[str, asyncio.Event] = {}
        self._active_environment_ready = None
        self._candidate_attempts: dict[
            tuple[str, int], list[tuple[str | None, Path]]
        ] = {}
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

    async def run_async_rollout(
        self, input_channel: Channel, output_channel: Channel
    ) -> None:
        """Serve episodes continuously; the endpoint must run before any starts."""
        await self._model_server.start()
        return await super().run_async_rollout(input_channel, output_channel)

    async def run_one_query(self, prompt_ids: list[int], answer: list[str]):
        """Run one episode, retrying service failures with fresh episode IDs."""
        task = episode_task(answer)
        task.pop("fallback_seeds", None)
        retries = self.cfg.agentloop.max_infra_retries
        errors = []
        for attempt in range(retries + 1):
            try:
                result = await self._run_episode(task)
            except RPentEpisodeError as error:
                startup = _GROUP_STARTUP.get()
                if (
                    startup is not None
                    and not startup.admitted
                    and error.kind == "env_setup"
                ):
                    raise
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
                uuid4().hex,
                task,
                AppendOnlyHistory(self.agentic_tokenizer),
                startup=_GROUP_STARTUP.get(),
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

    async def _run_rpent(
        self, episode: Episode, base_url: str, gpu: int | None
    ) -> dict:
        candidate = self.rollout_candidate()
        if candidate is not None:
            key = (candidate["cohort_id"], candidate["candidate_id"])
            output = Path(self.cfg.tools.rpent.output_dir) / episode.episode_id
            endpoint = self._launcher.endpoint(episode.task)
            self._candidate_attempts.setdefault(key, []).append((endpoint, output))
        return await self._launcher.run(
            episode, self._planner_options(base_url), gpu, candidate=candidate
        )

    async def _run_synchronous_candidates(self, request, output_channel):
        if self.cfg.tools.rpent.get("full_cohort_ready_timeout_s") is None:
            return await super()._run_synchronous_candidates(request, output_channel)
        event = self._environment_ready.setdefault(request.cohort_id, asyncio.Event())
        self._active_environment_ready = event
        try:
            return await super()._run_synchronous_candidates(request, output_channel)
        finally:
            self._environment_ready.pop(request.cohort_id, None)
            self._active_environment_ready = None

    async def release_rpent_rollout(self, cohort_id: str) -> None:
        """Allow the first model request after the runner verifies native readiness."""
        self._environment_ready.setdefault(cohort_id, asyncio.Event()).set()

    async def _run_synchronous_candidate_group(
        self,
        state: dict,
        candidate_ids: range,
        input_ids: list[int],
        answer: list[str],
        output_channel: Channel,
    ) -> None:
        task = episode_task(answer)
        domains = (self.cfg.get("env") or {}).get("rpent") or {}
        router = domains.get(task["environment"], {}).get("group_router", False)
        if not router:
            return await self._run_admitted_candidate_group(
                state, candidate_ids, input_ids, answer, output_channel
            )
        endpoint = self._launcher.endpoint(task)
        group = f"{state['id']}:{candidate_ids.start // self.cfg.algorithm.group_size}"
        try:
            while not state["closed"].is_set():
                result = await asyncio.to_thread(
                    rpc, endpoint, "backend.acquire_group", group
                )
                if result == "admitted":
                    await self._run_admitted_candidate_group(
                        state, candidate_ids, input_ids, answer, output_channel
                    )
                    return
                if result == "closed":
                    return
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            for member in candidate_ids:
                await self._fail_synchronous_candidate(
                    state, member, error, output_channel
                )
        finally:
            cleanup = asyncio.create_task(
                asyncio.to_thread(
                    rpc, endpoint, "backend.finish_group", group, timeout_s=185
                )
            )
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise

    async def _run_admitted_candidate_group(
        self,
        state: dict,
        candidate_ids: range,
        input_ids: list[int],
        answer: list[str],
        output_channel: Channel,
    ) -> None:
        task = episode_task(answer)
        fallback = task.pop("fallback_seeds", ())
        if not fallback:
            return await super()._run_synchronous_candidate_group(
                state, candidate_ids, input_ids, answer, output_channel
            )
        seeds = (task["seed"], *fallback)
        audit = []
        try:
            for index, seed in enumerate(seeds):
                selected = [json.dumps({**task, "seed": seed})]
                startup = GroupStartup(len(candidate_ids))
                try:
                    sampled = await self._sample_group_attempt(
                        state, candidate_ids, input_ids, selected, startup
                    )
                except RPentEpisodeError as error:
                    if error.kind != "env_setup" or startup.admitted:
                        raise
                    audit.append({"seed": seed, "error": str(error)})
                    for member in candidate_ids:
                        state["records"][member]["group_seed_attempts"] = audit
                    await self._release_failed_group(state, candidate_ids, index, seed)
                    if index == len(seeds) - 1:
                        raise
                    continue
                audit.append({"seed": seed})
                for member, result in zip(candidate_ids, sampled, strict=True):
                    state["records"][member]["group_seed_attempts"] = audit
                    await self._publish_synchronous_candidate(
                        state, member, result, output_channel
                    )
                return
        except asyncio.CancelledError:
            for member in candidate_ids:
                self._mark_synchronous_candidate_cancelled(state, member)
            raise
        except Exception as error:
            for member in candidate_ids:
                await self._fail_synchronous_candidate(
                    state, member, error, output_channel
                )

    async def _sample_group_attempt(
        self,
        state: dict,
        members: range,
        input_ids: list[int],
        answer: list[str],
        startup: GroupStartup,
    ) -> list:
        token = _GROUP_STARTUP.set(startup)
        try:
            tasks = [
                asyncio.create_task(
                    self._collect_synchronous_candidate(
                        state, member, input_ids, answer
                    )
                )
                for member in members
            ]
            try:
                return await asyncio.gather(*tasks)
            except BaseException:
                startup.abort()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        finally:
            _GROUP_STARTUP.reset(token)

    async def _release_failed_group(
        self, state: dict, members: range, attempt: int, seed: int
    ) -> None:
        cohort_id = state["id"]
        paths = []
        for member in members:
            record = state["records"][member]
            abandoned = record.setdefault("abandoned_episode_paths", {})
            for endpoint, path in self._candidate_attempts.get((cohort_id, member), ()):
                if str(path) not in abandoned:
                    abandoned[str(path)] = seed
                    paths.append((endpoint, path))
        if paths:
            endpoint = paths[0][0]
            await self._release_remote_candidates(
                f"{cohort_id}-group-{members.start}-attempt-{attempt}",
                endpoint,
                [path.name for _, path in paths],
            )

    async def finalize_rollout_candidates(self, records: list[dict]) -> None:
        """Keep each attempt and its disposition, including cancelled partial traces."""
        cohort_id = self._candidate_cohort["id"]
        owners_by_endpoint = defaultdict(list)
        for record in records:
            for endpoint, path in self._candidate_attempts.get(
                (cohort_id, record["candidate_id"]), ()
            ):
                if endpoint is not None:
                    owners_by_endpoint[endpoint].append(path.name)

        receipts = await asyncio.gather(
            *(
                self._release_remote_candidates(
                    cohort_id
                    if len(owners_by_endpoint) == 1
                    else f"{cohort_id}-backend-{index}",
                    endpoint,
                    owner_ids,
                )
                for index, (endpoint, owner_ids) in enumerate(
                    owners_by_endpoint.items()
                )
            ),
            return_exceptions=True,
        )
        cleanup = dict(zip(owners_by_endpoint, receipts, strict=True))

        for record in records:
            key = (cohort_id, record["candidate_id"])
            attempts = self._candidate_attempts.get(key, ())
            receipt = cleanup.get(attempts[0][0]) if attempts else None
            error = receipt if isinstance(receipt, BaseException) else None
            record["episode_paths"] = [str(path) for _, path in attempts]
            record["cleanup_complete"] = error is None
            record["cleanup_error"] = repr(error) if error is not None else None
            record["remote_cleanup"] = receipt if error is None else None
            for _, path in attempts:
                if path.is_dir():
                    (path / "sampling-selection.json").write_text(
                        json.dumps({**record, "cohort_id": cohort_id}, indent=2) + "\n"
                    )
            if error is None:
                self._candidate_attempts.pop(key, None)

        for receipt in receipts:
            if isinstance(receipt, BaseException):
                raise receipt

    async def _release_remote_candidates(
        self, cohort_id: str, endpoint: str, owner_ids: list[str]
    ) -> dict:
        """Use the configured RPent runtime to tombstone and release exact attempts."""
        output = Path(self.cfg.tools.rpent.output_dir) / "sampling"
        output.mkdir(parents=True, exist_ok=True)
        request = output / f"{cohort_id}-worker-{self._rank}-owners.json"
        receipt = request.with_name(request.stem + "-receipt.json")
        log = request.with_suffix(".log")
        request.write_text(json.dumps(owner_ids) + "\n")
        code = (
            "import json,sys; from pathlib import Path; "
            "from robots.rl.remote import cancel_remote_owners; "
            "result=cancel_remote_owners(sys.argv[1],"
            "json.loads(Path(sys.argv[2]).read_text())); "
            "Path(sys.argv[3]).write_text(json.dumps(result))"
        )
        with log.open("wb") as log_file:
            process = await asyncio.create_subprocess_exec(
                self.cfg.tools.rpent.python,
                "-c",
                code,
                endpoint,
                str(request),
                str(receipt),
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                env=self._launcher.environment(),
            )
            try:
                await asyncio.wait_for(process.wait(), timeout=240)
            finally:
                await self._launcher.stop(process)
        if process.returncode != 0:
            raise RuntimeError(
                f"Remote candidate cleanup failed with {process.returncode}; see {log}."
            )
        result = json.loads(receipt.read_text())
        if result["owner_ids"] != owner_ids or not result["cleanup_complete"]:
            raise RuntimeError(f"Remote candidate cleanup was not verified: {receipt}.")
        return result

    def _planner_options(self, base_url: str) -> dict:
        """Planner options of an episode whose policy is served at ``base_url``."""
        agentloop = self.cfg.agentloop
        compaction = agentloop.get("history_compaction")
        options = {
            "base_url": base_url,
            "model_name": _MODEL_NAME,
            "max_turns": agentloop.max_turns,
            "max_tokens": self.cfg.algorithm.sampling_params.max_new_tokens,
            "request_timeout_s": agentloop.model_request_timeout_s,
            "episode_timeout_s": agentloop.episode_timeout_s,
            "history_compaction": None
            if compaction is None
            else OmegaConf.to_container(compaction, resolve=True),
        }
        sampling = self.cfg.algorithm.sampling_params
        options.update(
            thinking=agentloop.chat_template_kwargs.enable_thinking,
            reasoning_effort="none",
            sampling_params={
                name: sampling[name]
                for name in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "presence_penalty",
                    "repetition_penalty",
                )
            },
        )
        return options

    @staticmethod
    def _check_planner_done(episode: Episode, status: dict) -> None:
        """Reject a planner exit that has no task outcome."""
        if status["terminal"]:
            return
        if status.get("error_kind") in {"model_transport", "tool_transport"}:
            raise RPentEpisodeError(
                f"RPent service request failed: {status.get('error')}",
                kind=status["error_kind"],
            )
        # Unknown planner failures must not select a replacement trajectory.
        raise RuntimeError(
            f"RPent planner of episode {episode.episode_id} stopped without an "
            f"outcome ({status.get('error_kind')}): {status.get('error')}"
        )

    def _processor(self):
        """Image processor of the served policy, loaded on first use."""
        if self._image_processor is None:
            from transformers import AutoProcessor

            self._image_processor = AutoProcessor.from_pretrained(
                self.cfg.rollout.model.model_path, trust_remote_code=True
            )
        return self._image_processor

    async def _render_request(self, episode: Episode, request: dict):
        """Return this turn's prompt tokens and the images they expand.

        A request carrying images appends only its new messages to the recorded
        tokens, expanding their images with the model's processor, so the turn
        stays a token prefix of the next and consecutive turns can merge into
        one training row. Text-only turns append to the sampled tokens as usual.
        """
        if shared_multi_modal_processor_enabled(
            OmegaConf.select(self.cfg, "rollout.sglang")
        ):
            rendered = await asyncio.to_thread(
                episode.history.render_token_space, request
            )
            if not rendered.images:
                return rendered.input_ids, [], None
            directory = (
                Path(self.cfg.tools.rpent.output_dir)
                / episode.episode_id
                / "training_images"
            )
            shared_cache = self.cfg.rollout.sglang.get(
                "enable_token_space_preprocess_cache", False
            )
            images = rendered.images
            if shared_cache:
                payload = await asyncio.to_thread(rendered.store_images, directory)
                images = [reference["path"] for reference in payload["image_refs"]]
            prepared = await self.prepare_multimodal(
                rendered.input_ids,
                images,
                prefix_length=rendered.prefix_length,
                image_spans=rendered.image_spans,
            )
            rendered = episode.history.accept_token_space(prepared)
            if not shared_cache:
                payload = await asyncio.to_thread(rendered.store_images, directory)
            return rendered.input_ids, images, {**payload, **prepared}
        if not request_has_images(request):
            prompt_ids = await asyncio.to_thread(episode.history.render, request)
            return prompt_ids, [], None
        rendered = await asyncio.to_thread(
            episode.history.render_multimodal, request, self._processor()
        )
        if self.cfg.tools.rpent.get("store_images_on_disk", False):
            directory = (
                Path(self.cfg.tools.rpent.output_dir)
                / episode.episode_id
                / "training_images"
            )
            multimodal = await asyncio.to_thread(rendered.store_images, directory)
        else:
            multimodal = {
                "pixel_values": rendered.pixel_values.to(torch.bfloat16),
                "image_grid_thw": rendered.image_grid_thw,
                "num_images": len(rendered.images),
            }
        return rendered.input_ids, rendered.images, multimodal

    async def _complete(self, episode: Episode, request: dict) -> dict:
        """Sample one planner request, record the turn, and return its completion."""
        if (
            episode.startup is not None
            and not episode.turns
            and not episode.startup.admitted
        ):
            episode.startup.mark_ready()
            await episode.startup.wait()
        if self._active_environment_ready is not None and not episode.turns:
            await self._active_environment_ready.wait()
        turn_idx = len(episode.turns)
        started = time.perf_counter()
        started_at = time.time()
        release_capture = release_image_replay_enabled(self.cfg.rollout)
        if release_capture:
            rendered = await asyncio.to_thread(
                episode.history.render_token_space, request
            )
            prompt_ids, images, multimodal = rendered.input_ids, rendered.images, None
        else:
            prompt_ids, images, multimodal = await self._render_request(
                episode, request
            )
        rendered_at = time.perf_counter()
        sampling_params = {
            "max_new_tokens": self._output_budget(len(prompt_ids)),
            # SGLang skips tokenizer initialization for token-ID requests,
            # and the model config's EOS can differ from the chat EOS.
            "stop_token_ids": [self.tokenizer.eos_token_id],
        }
        budget = None
        if self._thinking_limit is not None:
            if release_capture:
                raise ValueError(
                    "Thinking budget is unavailable with release image capture"
                )
            budget = ThinkingBudget.from_prompt(
                prompt_ids,
                self._thinking_limit,
                self.tokenizer.convert_tokens_to_ids("<think>"),
                self.tokenizer.convert_tokens_to_ids("</think>"),
            )
            sampling_params["custom_params"] = budget.custom_params()
            sampling_params["custom_logit_processor"] = serialized_thinking_processor()
        extra = {"image_data": images} if images else {}
        if release_capture:
            if images:
                extra.update(
                    image_spans=rendered.image_spans,
                    image_history=episode.history.image_history,
                    image_token_budget=(
                        self.cfg.runner.seq_length,
                        self.cfg.agentloop.minimum_output_tokens,
                    ),
                )
                if self.cfg.tools.rpent.get("store_images_on_disk", False):
                    extra["image_store_dir"] = str(
                        Path(self.cfg.tools.rpent.output_dir)
                        / episode.episode_id
                        / "training_images"
                    )
        elif shared_multi_modal_processor_enabled(
            OmegaConf.select(self.cfg, "rollout.sglang")
        ):
            extra["return_prompt_token_ids"] = True
            extra["mm_token_expansion_start_len"] = len(prompt_ids)
        generated = await self.generate(
            prompt_ids, sampling_params=sampling_params, **extra
        )
        if release_capture:
            if images:
                multimodal = generated["multimodal_input"]
                prompt_ids = episode.history.accept_token_space(multimodal).input_ids
                episode.history.image_history = generated["image_history"]
            else:
                # The release uses the supplied token IDs directly for text.
                generated["prompt_token_ids"] = prompt_ids
        generated_at = time.perf_counter()
        self._check_generation(generated, prompt_ids)
        if self.cfg.runner.get("synchronous", False):
            version = generated.get("version_start")
            if version is None or version != generated.get("version_end"):
                raise ValueError(
                    "Synchronous RPent requires one installed weight version throughout a generation."
                )
            if episode.weight_version is not None and episode.weight_version != version:
                raise ValueError(
                    "The policy version changed during a synchronous RPent episode."
                )
            episode.weight_version = version
            generated["weight_versions"] = [version] * len(generated["output_ids"])
        response_ids = generated["output_ids"]
        response_text, calls, message = await self._parse_response(
            response_ids, request.get("tools"), turn_idx
        )
        episode.history.record(request, prompt_ids, response_ids, message)
        extra_fields = {
            "turn_id": turn_idx,
            "tool_names": [call["name"] for call in calls],
            "chain_id": episode.history.chain_id,
        }
        if budget is not None:
            extra_fields["forced_token_positions"] = generated["forced_token_positions"]
        episode.turns.append(
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_text=response_text,
                response_mask=None
                if budget is None
                else budget.response_mask(
                    response_ids, generated["forced_token_positions"]
                ),
                response_logprobs=generated.get("logprobs"),
                response_weight_versions=generated.get("weight_versions"),
                response_images=multimodal,
                version_start=generated.get("version_start"),
                version_end=generated.get("version_end"),
                is_end=generated["finish_reason"] == "stop",
                extra_fields=extra_fields,
            )
        )
        await asyncio.to_thread(
            write_training_turn,
            self.cfg.tools.rpent.output_dir,
            episode,
            episode.turns[-1],
            generated,
            images,
            started_at,
            {
                "render_seconds": rendered_at - started,
                "generation_seconds": generated_at - rendered_at,
                "parse_seconds": time.perf_counter() - generated_at,
            },
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
        if shared_multi_modal_processor_enabled(
            OmegaConf.select(self.cfg, "rollout.sglang")
        ):
            if generated.get("prompt_token_ids") != prompt_ids:
                raise ValueError(
                    "SGLang consumed different prompt IDs from the recorded trajectory"
                )
            if not release_image_replay_enabled(self.cfg.rollout):
                vision_ids = {
                    self.tokenizer.convert_tokens_to_ids(marker)
                    for marker in (
                        "<|image_pad|>",
                        "<|video_pad|>",
                        "<|vision_start|>",
                        "<|vision_end|>",
                    )
                }
                if vision_ids.intersection(generated["output_ids"]):
                    raise ValueError(
                        "Generated vision markers are unsupported in token-space chats"
                    )
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
        """Decode a response into its text, tool calls and assistant message.

        The generation prompt opens the think block, so a response runs from
        the reasoning to ``</think>`` and then the answer. The reasoning goes
        back as ``reasoning_content``, the field SGLang's reasoning parser
        returns, so the planner sends the same message structure on both paths
        and the chat template renders the block again from it.
        """
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        answer = response_text.removesuffix(self.tokenizer.eos_token or "")
        reasoning, found, after = answer.partition(_THINK_END)
        if found:
            answer = after
        else:
            reasoning = ""
        content, requests = await self.toolcall_parser(answer, tools=tools)
        calls = [{"name": item.name, "arguments": item.arguments} for item in requests]
        message = {"role": "assistant", "content": content.strip() or None}
        if found:
            message["reasoning_content"] = reasoning.strip()
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
        return episode_output(
            episode.turns,
            task=episode.task,
            episode_id=episode.episode_id,
            status=status,
            elapsed_s=time.monotonic() - episode.started,
            merge=self.cfg.agentloop.merge_prefix_turns,
            max_scored_tokens=self.cfg.agentloop.get(
                "max_scored_tokens", self.cfg.algorithm.sampling_params.max_new_tokens
            ),
        )

    def gen_extra_fields(self, task_results, answer):
        turn, traj, group, train = super().gen_extra_fields(task_results, answer)
        group = {"task": episode_task(answer)}
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

    def post_process_metric(
        self, agent_metrics_list: list[dict], *, gather_workers: bool = True
    ) -> dict | None:
        """Stop the update when too many episodes needed infrastructure retries."""
        metrics = super().post_process_metric(
            agent_metrics_list, gather_workers=gather_workers
        )
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
