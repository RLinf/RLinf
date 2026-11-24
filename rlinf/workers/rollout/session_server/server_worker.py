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

"""The session server: an HTTP proxy between agent loops and the router.

One worker process holds a FastAPI + uvicorn server (the form of
:class:`SGLangAgentWorkerWithHTTPServer`) and the per-group state of every
group routed to it. Agent loops call the data-plane route to generate; the
component that manages the agents calls the lifecycle routes to close and
score trajectories and to abort groups. Each generate response becomes one
:class:`~rlinf.workers.rollout.session_server.payload.PayloadPart` put on a
same-node payload channel in the background, so an agent's turn latency does
not include the channel's queueing; a group whose close and reward sets both
cover every trajectory is finalized: its payload is completed, then its
:class:`~rlinf.data.schema.reasoning_results.GroupMetadata` is reported.
"""

import asyncio
import copy
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from omegaconf import DictConfig

from rlinf.data.schema.reasoning_results import (
    DiscardedGroup,
    GroupMetadata,
    PayloadRef,
)
from rlinf.scheduler import Channel, Worker
from rlinf.utils.http_client import InferenceHTTPClient
from rlinf.workers.rollout.session_server.chat_completions import (
    ChatCheckpoint,
    ChatCompletions,
    ChatConversation,
)
from rlinf.workers.rollout.session_server.payload import (
    PayloadDiscard,
    PayloadPart,
    PayloadWriter,
    payload_key_for,
    payload_locator_for,
    routing_key_for,
)
from rlinf.workers.rollout.session_server.turns import (
    TurnRecord,
    UnderreportedTruncation,
    classify_turns,
    kept_len,
    new_turn_record,
)
from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

_WRITE_FAILED = "payload_write_failed"
# A session serves one request at a time; a second request, or a close,
# sent while one runs gets this answer.
_SESSION_BUSY = (
    "a request of this session is still running; wait for it or abort the group"
)

# Bounded history of groups whose completion metadata was sent; guards a
# straggler abort from producing a second record for the buffer worker.
_REPORTED_HISTORY_LIMIT = 4096


@dataclass
class _SessionState:
    """Per-trajectory state: the turn records and the last engine I/O.

    The last input and output token ids are kept only for the next turn's
    prefix check; logprobs never live here. ``extra_fields_train`` is the
    trajectory's slice of the training extra fields, declared at close and
    moved into the finalize part's verdicts. ``not_training_turns`` holds
    turn indexes a close dropped from the training data (a rolled-back
    retry); those turns stay in the payload but produce no sequence.
    """

    session_id: Optional[str]
    group_id: int
    traj_index: int
    turns: list[TurnRecord] = field(default_factory=list)
    last_input: Optional[list[int]] = None
    last_output: Optional[list[int]] = None
    extra_fields_train: dict = field(default_factory=dict)
    not_training_turns: set = field(default_factory=set)
    # "chat" once this session is driven through the chat entry; None for
    # sessions only the generate entry has served. The turn classification
    # differs between the two entries.
    entry: Optional[str] = None
    # The chat entry's server-side conversation.
    chat: Optional[ChatConversation] = None
    # Set while a request of this session is being served, by either
    # endpoint: a session handles one request at a time.
    inflight: bool = False


@dataclass
class _GroupState:
    """Per-group state; the completion condition reads the two sets.

    ``discard_reason`` doubles as the invalidation record: once set, late
    generate responses, closes, and rewards are dropped. A group is discarded
    by an abort (the agent's reason) or by a failed payload write
    (``payload_write_failed``). ``write_failed`` is the payload-failure flag
    finalize consults; it is recorded whatever the group's state, because a
    settlement can fail while finalize is already waiting. ``finalized``
    marks finalize as started (one-shot guard), ``reported`` as completion
    metadata actually sent; the state is dropped once reported, a discarded
    group keeps its record — without the token buffers — until its TTL
    (``rollout.router.max_idle_secs``) expires. ``payload_bytes`` counts the
    bytes this group has written to the payload channel, accumulated over
    the per-turn parts only (the finalize part is small and not counted);
    it rides on the finalize put's weight for the payload_queued_bytes
    gauge. ``verify_result`` is the agent's assembly-verification
    reference, arriving with a close.
    """

    group_id: int
    task_name: str
    group_size: int
    # Payload channel this group writes to, taken in creation order. It is
    # recorded here so every part of the group lands in one channel process,
    # and so ``payload_ref`` can name that channel.
    payload_shard: int = 0
    sessions: dict[int, _SessionState] = field(default_factory=dict)
    closed: set[int] = field(default_factory=set)
    rewarded: set[int] = field(default_factory=set)
    rewards: list[Optional[float]] = field(default_factory=list)
    discard_reason: Optional[str] = None
    discarded_at: Optional[float] = None
    earliest_version: Optional[int] = None
    completion_version: Optional[int] = None
    num_gen_tokens: int = 0
    num_generate_calls: int = 0
    payload_bytes: int = 0
    verify_result: Optional[dict] = None
    finalized: bool = False
    reported: bool = False
    write_failed: bool = False
    pending_settles: list[asyncio.Task] = field(default_factory=list)


class SessionServerWorker(Worker):
    """Proxies generates to the router and collects the payload of each group.

    The runner launches one worker per session server instance (its
    ``server_index`` fixes the port), calls the resident
    :meth:`serve` and holds the returned handle for liveness monitoring.

    Two entries forward generates: ``/generate`` for agents that tokenize
    themselves, and the OpenAI chat completions route for agents that send
    messages. Both share :meth:`_generate_core`; the chat route renders the
    messages and parses the engine's answer with
    :class:`~rlinf.workers.rollout.session_server.chat_completions.ChatCompletions`,
    and owns the conversation state the two of them need.
    """

    def __init__(self, cfg: DictConfig, server_index: int):
        super().__init__()
        self._cfg = cfg
        self._server_index = server_index

        # Defaults live in rlinf.config.validate_session_server_cfg; a cfg
        # reaching here always carries the full block.
        session_server_cfg = cfg.session_server
        self._host = str(session_server_cfg.host)
        self._port = int(session_server_cfg.port) + server_index
        self._payload_put_timeout_s = float(session_server_cfg.payload_put_timeout_s)
        self._generate_timeout_s = float(session_server_cfg.generate_timeout_s)
        self._keep_alive_s = int(session_server_cfg.keep_alive_s)
        self._per_task_generate_timeout_s = {
            str(task_name): float(settings.generate_timeout_s)
            for task_name, settings in session_server_cfg.per_task.items()
        }

        self._expected_group_size = int(cfg.algorithm.group_size)
        # The session server serves only the training policy model, whose
        # rollout worker is built without config_rollout; if that ever
        # changes, the runner must hand this server the same block.
        self._sampling_params = SGLangWorker.get_default_sampling_params(cfg)
        self._return_logprob = bool(cfg.rollout.return_logprobs)
        self._discard_record_ttl_s = float(cfg.rollout.router.max_idle_secs)
        # Assembly verification: the agent's close carries a reference
        # result, the generate responses carry logprobs, and the finalize
        # part carries both sides' sampling params.
        self._verify_assembly = bool(session_server_cfg.verify_assembly)

        self._groups: dict[int, _GroupState] = {}
        self._sessions: dict[str, _SessionState] = {}
        self._loaded_versions: list[int] = []
        self._router_client: Optional[InferenceHTTPClient] = None
        self._agent_output_channel: Optional[Channel] = None
        # Built by serve(), once the payload channels exist.
        self._payload_writer: Optional[PayloadWriter] = None
        # Shard handed to the next group created; advanced in creation order
        # so every shard of this session server is used, whatever
        # num_servers is.
        self._next_payload_shard = 0
        self._http_server: Optional[uvicorn.Server] = None
        self._http_server_task: Optional[asyncio.Task] = None
        # Set when serving has ended (normally or not), with the exception
        # serve() re-raised, if any: wait_until_serving reports an early
        # end instead of hanging on a server that never listened.
        self._serve_ended = False
        self._serve_error: Optional[BaseException] = None
        # Recently delivered groups, oldest first, with their session ids.
        # The group state is gone after delivery, so this is what answers
        # late calls for it: an abort must not add a second record for the
        # buffer worker, and a close or reward that the client resends after
        # losing the response must still succeed.
        self._reported_groups: dict[int, tuple[str, ...]] = {}
        # Session id to its delivered group, for the resent close and reward.
        self._reported_sessions: dict[str, int] = {}
        self._fatal_error: Optional[RuntimeError] = None
        self._chat_template_kwargs = dict(session_server_cfg.chat_template_kwargs)
        # The chat route's format converter, built by _setup_chat when the
        # config names an agentic tokenizer family; None serves the
        # generate entry only.
        self._chat_entry: Optional[ChatCompletions] = None
        self._metrics = {
            "session_server/chat_tool_call_parse_failed": 0,
            "session_server/chat_tool_call_responses": 0,
            "session_server/chat_reasoning_responses": 0,
            "session_server/prefix_continuous": 0,
            "session_server/prefix_underreported": 0,
            "session_server/prefix_rewritten": 0,
            "session_server/forward_count": 0,
            "session_server/forward_time_total_s": 0.0,
            "session_server/payload_write_failed": 0,
        }
        self._setup_http_routes()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _setup_http_routes(self) -> None:
        app = FastAPI(title="RLinf-SessionServer", version="1.0.0")

        @app.post("/sessions")
        async def handle_create_session(raw_request: Request):
            body = await raw_request.json()
            return await self._create_session(body)

        @app.post("/sessions/{sid}/generate")
        async def handle_generate(sid: str, raw_request: Request):
            body = await raw_request.json()
            return await self._generate(sid, body)

        @app.post("/sessions/{sid}/v1/chat/completions")
        async def handle_chat_completions(sid: str, raw_request: Request):
            body = await raw_request.json()
            return await self._chat_completions(sid, body)

        @app.post("/sessions/{sid}/close")
        async def handle_close(sid: str, raw_request: Request):
            body = await raw_request.json()
            return await self._close_session(sid, body)

        @app.post("/sessions/{sid}/reward")
        async def handle_reward(sid: str, raw_request: Request):
            body = await raw_request.json()
            return await self._reward(sid, body)

        @app.post("/groups/{group_id}/abort")
        async def handle_abort(group_id: int, raw_request: Request):
            body = await raw_request.json()
            return await self._abort(group_id, body)

        @app.get("/health")
        async def handle_health():
            return {"status": "healthy", "server_index": self._server_index}

        self._http_server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=self._host,
                port=self._port,
                log_level="warning",
                access_log=False,
                # Explicit so the agent-loop client's keepalive (this
                # value minus 1, set by the runner) stays shorter than
                # the server's for any configured value.
                timeout_keep_alive=self._keep_alive_s,
                # A fatal error only asks uvicorn to exit; graceful shutdown
                # would otherwise wait for every in-flight request — up to a
                # full generate timeout — before serve() can re-raise.
                timeout_graceful_shutdown=5,
            )
        )

    def _setup_chat(self) -> None:
        """Build the chat route's format converter.

        Called from serve(), so a wrong model path or chat template fails at
        startup instead of on the first chat request. A config without
        ``agentloop.agentic_tokenizer`` serves the generate entry only;
        chat requests then fail with 500.
        """
        self._chat_entry = None
        # A config without an agentloop block (or without the family key)
        # serves the generate entry only.
        agentloop_cfg = self._cfg.get("agentloop", None)
        family = (
            agentloop_cfg.get("agentic_tokenizer", None)
            if agentloop_cfg is not None
            else None
        )
        if family is None:
            return
        from rlinf.agents.tokenization.agentic_tokenizer import get_agentic_tokenizer
        from rlinf.models.tokenization.hf import hf_tokenizer

        model_path = self._cfg.rollout.model.model_path
        tokenizer = hf_tokenizer(
            model_path,
            trust_remote_code=self._cfg.rollout.model.get("trust_remote_code", False),
        )
        self._chat_entry = ChatCompletions(
            get_agentic_tokenizer(
                tokenizer,
                tokenizer_type=family,
                chat_template_kwargs=dict(self._chat_template_kwargs),
            ),
            os.path.basename(str(model_path)),
        )
        self.log_info(
            f"Session server {self._server_index} chat entry ready (family {family})"
        )

    async def serve(
        self,
        router_url: str,
        agent_output_channel: Channel,
        payload_channels: list[Channel],
    ) -> None:
        """Serve until the HTTP server exits.

        Binds the channels the traffic flows through, starts uvicorn on this
        actor's event loop, and awaits the server task itself, so this handle
        returns or raises exactly when uvicorn does. The runner holds the
        handle for liveness monitoring; :meth:`stop` ends it.

        Args:
            router_url: Base URL (``http://host:port``) requests are forwarded
                to, with the group's routing key in the
                ``X-SMG-Routing-Key`` header.
            agent_output_channel: Channel completed-group metadata and
                discard records are reported on.
            payload_channels: The same-node payload channels, one per shard.
                Every part of a group goes to one of them.
        """
        assert self._http_server_task is None, "serve is already running."
        assert payload_channels, "serve needs at least one payload channel."
        try:
            # The length the actor trains on, which the chat entry enforces.
            self._max_seq_length = int(self._cfg.runner.seq_length)
            # Load the chat entry's tokenizer now, so a bad model path or chat
            # template fails at startup instead of on the first chat request.
            self._setup_chat()
            self._router_client = InferenceHTTPClient(router_url, keepalive_timeout=4.0)
            self._agent_output_channel = agent_output_channel
            self._payload_writer = PayloadWriter(
                payload_channels,
                max_pending_bytes=int(
                    self._cfg.session_server.max_pending_payload_bytes
                ),
                put_timeout_s=self._payload_put_timeout_s,
                on_write_failure=self._note_write_failure,
                on_peer_failure=lambda exc: self._fail_serving("payload put", exc),
            )
            self._http_server_task = asyncio.create_task(self._run_http_server())
            self.log_info(
                f"Session server {self._server_index} serving on "
                f"{self._host}:{self._port}"
            )
            await self._http_server_task
        except BaseException as exc:
            self._serve_error = exc
            raise
        finally:
            self._http_server_task = None
            self._serve_ended = True
        if self._fatal_error is not None:
            # Component death (an agent_output_channel we cannot write to)
            # ends serving; re-raise so the runner's liveness monitor ends
            # the run instead of leaking admission slots.
            raise self._fatal_error

    async def _run_http_server(self) -> None:
        """Run uvicorn, turning its process-exit startup failures into errors.

        uvicorn calls ``sys.exit`` when it cannot bind (the port is taken),
        and a ``SystemExit`` escaping a task is not delivered to whoever
        awaits it: it unwinds the event loop instead, killing the actor and
        leaving the failure unattributable. Catching it inside the coroutine
        keeps it on the ordinary exception path.
        """
        try:
            await self._http_server.serve()
        except SystemExit as exc:
            raise RuntimeError(
                f"session server {self._server_index} failed to start on "
                f"{self._host}:{self._port} (see the uvicorn error above)"
            ) from exc

    async def stop(self) -> None:
        """Stop the HTTP server and the background put settlements."""
        if self._http_server_task is not None:
            self._http_server.should_exit = True
            await self._http_server_task
            self._http_server_task = None
            self._serve_ended = True
        if self._payload_writer is not None:
            await self._payload_writer.aclose()
        if self._router_client is not None:
            await self._router_client.aclose()
            self._router_client = None

    async def wait_until_serving(self) -> None:
        """Wait until the HTTP server accepts connections.

        Raises when serving ends before that, chaining serve()'s exception
        if it had one: a startup failure such as the fixed port already
        being taken must surface here, not as the agent loop's
        connection-refused error after the URLs were handed out. Call
        only after serve() has been dispatched.
        """
        while not self._http_server.started:
            if self._serve_ended:
                raise RuntimeError(
                    f"session server {self._server_index} stopped serving "
                    f"before accepting connections on {self._host}:{self._port}"
                ) from self._serve_error
            await asyncio.sleep(0.05)

    def get_server_address(self) -> str:
        """Return the ``host:port`` agents reach this server on."""
        host = self._host
        if host == "0.0.0.0":
            import ray.util

            host = ray.util.get_node_ip_address()
        return f"{host}:{self._port}"

    async def set_loaded_versions(self, versions: dict[int, int]) -> None:
        """Record the weight versions every rollout instance has loaded.

        Called by the runner after each weight sync. The minimum becomes the
        ``version_start`` lower bound of every later generate.
        """
        self._loaded_versions = list(versions.values())

    async def get_metrics(self) -> dict:
        """Drain the per-step counters under the ``session_server/`` prefix.

        The runner reads once per training step, so every count and total
        covers the window since its previous call (the buffer's
        get_step_metrics drains the same way). The pending-bytes gauge is
        instantaneous and read separately below.
        """
        self._expire_discard_records()
        metrics = dict(self._metrics)
        for key in self._metrics:
            self._metrics[key] = 0
        if self._payload_writer is not None:
            metrics["session_server/pending_payload_bytes"] = (
                self._payload_writer.pending_bytes
            )
        else:
            metrics["session_server/pending_payload_bytes"] = 0
        return metrics

    def _expire_discard_records(self) -> None:
        """Drop discard records older than their TTL.

        The sweep piggybacks on the runner's metrics polling. The TTL matches
        the router's routing-key idle timeout: past it the group's prefix
        affinity is gone anyway, and any straggler still holding the group is
        older than any request the run will ever send for it.
        """
        now = time.monotonic()
        expired = [
            group_id
            for group_id, group in self._groups.items()
            if group.discarded_at is not None
            and now - group.discarded_at > self._discard_record_ttl_s
        ]
        for group_id in expired:
            group = self._groups.pop(group_id)
            for session in group.sessions.values():
                if session.session_id is not None:
                    self._sessions.pop(session.session_id, None)

    # ------------------------------------------------------------------
    # Session and group state
    # ------------------------------------------------------------------
    async def _create_session(self, body: dict) -> JSONResponse:
        try:
            group_id = self._int_field(body, "group_id")
            traj_index = self._int_field(body, "traj_index")
            group_size = self._int_field(body, "group_size")
        except ValueError as exc:
            return self._error(400, str(exc))
        task_name = body.get("task_name")
        if not isinstance(task_name, str) or not task_name:
            return self._error(400, "task_name must be a non-empty string")

        if group_size != self._expected_group_size:
            return self._error(
                400,
                f"group_size {group_size} does not match the configured "
                f"algorithm.group_size {self._expected_group_size}",
            )

        group = self._groups.get(group_id)
        if group is None:
            group = _GroupState(
                group_id=group_id,
                task_name=task_name,
                group_size=group_size,
                payload_shard=self._take_payload_shard(),
                rewards=[None] * group_size,
            )
            self._groups[group_id] = group
        elif group.discard_reason is not None:
            return self._error(
                410,
                f"group {group_id} was discarded ({group.discard_reason}); "
                f"issue the group again with a new group_id",
            )
        elif group.task_name != task_name or group.group_size != group_size:
            return self._error(
                400,
                f"group {group_id} exists with task_name={group.task_name}, "
                f"group_size={group.group_size}; got task_name={task_name}, "
                f"group_size={group_size}",
            )
        if traj_index in group.sessions:
            return self._error(
                400,
                f"group {group_id} already has a session for traj_index {traj_index}",
            )
        if traj_index < 0 or traj_index >= group_size:
            return self._error(
                400,
                f"traj_index {traj_index} is outside group_size {group_size}",
            )

        session = _SessionState(
            session_id=None, group_id=group_id, traj_index=traj_index
        )
        session_id = uuid.uuid4().hex[:12]
        session.session_id = session_id
        self._sessions[session_id] = session
        group.sessions[traj_index] = session
        return JSONResponse(content={"session_id": session_id})

    def _session(
        self, sid: str
    ) -> tuple[Optional[_SessionState], Optional[tuple[int, str]]]:
        """The session of this id, or the status and message to answer with."""
        session = self._sessions.get(sid)
        if session is None:
            return None, (404, f"unknown session {sid}")
        return session, None

    def _live_group(
        self, session: _SessionState
    ) -> tuple[Optional[_GroupState], Optional[tuple[int, str]]]:
        """The group of a session, or the status and message to answer with.

        A group discarded by an abort drops everything that arrives late; one
        discarded by a failed payload write answers generate and close with
        an error so the caller learns its trajectory was not recorded. The
        abort case carries a status below 400: the caller answers its entry's
        success shape instead of an error — the agent's own abort asked for
        the group to go away, so the refusal needs no message.
        """
        group = self._groups.get(session.group_id)
        if group is None:
            return None, (
                404,
                f"group {session.group_id} is unknown or already finished",
            )
        if group.discard_reason is not None:
            if group.discard_reason == _WRITE_FAILED:
                return None, (
                    410,
                    f"group {session.group_id} was discarded ({_WRITE_FAILED})",
                )
            return None, (200, "")
        return group, None

    def _turn_refusal(self, session: _SessionState) -> Optional[tuple[int, str]]:
        """The status and message that refuse a new turn of this session, or
        None when the trajectory may take one.

        Both entries call this before they change any session state, so a
        request refused here leaves the session as it was: a trajectory
        closed since the last request keeps the turns, the not-training
        turns and the extra fields that its close already classified.

        Returns:
            The status and message to answer with, or None.
        """
        group, error = self._live_group(session)
        if error is not None:
            # An abort-discarded group answers 410 so the agent fails its
            # turn; other failures pass through.
            status, message = error
            if status < 400:
                status, message = 410, f"group {session.group_id} was discarded"
            return status, message
        if group.finalized or session.traj_index in group.closed:
            # A closed trajectory takes no more turns: its turns are already
            # classified, so another one would reach the payload without a
            # place in the training data. A close cannot arrive while a
            # request runs (a session serves one request at a time), so the
            # check holds until the request's turn is recorded.
            return (
                410,
                f"trajectory {session.traj_index} of group {session.group_id} "
                f"is already closed",
            )
        return None

    # ------------------------------------------------------------------
    # Data plane
    # ------------------------------------------------------------------
    async def _generate(self, sid: str, body: dict) -> JSONResponse:
        session, error = self._session(sid)
        if error is not None:
            return self._error(*error)
        if session.inflight:
            return self._error(409, _SESSION_BUSY)
        session.inflight = True
        try:
            error = self._turn_refusal(session)
            if error is not None:
                return self._error(*error)
            rlinf_block = body.get("rlinf") or {}
            # Every other body field rides through to the engine unchanged:
            # the request is sglang's /generate shape apart from the rlinf
            # block.
            passthrough = {
                key: value
                for key, value in body.items()
                if key not in ("rlinf", "input_ids", "sampling_params")
            }
            result, error = await self._generate_core(
                session,
                body.get("input_ids"),
                body.get("sampling_params"),
                rlinf_block.get("prev_kept_len"),
                passthrough,
            )
            if error is not None:
                return self._error(*error)
            content = {
                "output_ids": list(result["output_ids"]),
                "finish_reason": dict(result["finish_reason"]),
                "weight_version": result["weight_version"],
                "cached_tokens": result["cached_tokens"],
            }
            if self._verify_assembly:
                # Only the verification needs the agent-side logprobs; normal
                # responses keep them out of the data plane.
                content["logprobs"] = result["logprobs"]
            return JSONResponse(content=content)
        finally:
            session.inflight = False

    async def _generate_core(
        self,
        session: _SessionState,
        input_ids: list,
        sampling_overrides: Any,
        prev_kept_len: Optional[int],
        body_fields: Optional[dict] = None,
    ) -> tuple[Optional[dict], Optional[tuple[int, str]]]:
        """Run one generation turn for ``session``; both entries share it.

        Validates the inputs, records the prefix bookkeeping, forwards to
        the router, checks the abort / weight-version contracts, writes the
        payload part, and updates the metrics and versions. The caller has
        already checked with :meth:`_turn_refusal` that the trajectory may
        take a turn, before changing any session state.

        Returns:
            The engine result (``text``, ``output_ids``, ``finish_reason``,
            ``weight_version``, ``cached_tokens``, ``logprobs``) —
            ``logprobs`` only when the assembly check needs them — or the
            status and message of the failure, which each entry wraps in its
            own error shape.

        Args:
            body_fields: Extra request fields forwarded to the engine as-is
                (the /generate entry passes everything but the rlinf block,
                input_ids and sampling_params; the chat entry passes none).
                ``sampling_params`` and ``return_logprob`` are always
                overwritten by this server's merged values.
        """
        if not isinstance(input_ids, list) or not input_ids:
            return None, (400, "input_ids must be a non-empty list")
        if prev_kept_len is not None and (
            not isinstance(prev_kept_len, int) or prev_kept_len < 0
        ):
            return None, (400, "rlinf.prev_kept_len must be a non-negative int")
        if not self._loaded_versions:
            return None, (
                500,
                "no loaded weight versions were recorded; the runner must "
                "call set_loaded_versions before the first generate",
            )

        group = self._groups[session.group_id]
        version_start = min(self._loaded_versions)
        turn = new_turn_record(
            input_ids, session.last_input, session.last_output, prev_kept_len
        )

        forwarded = dict(body_fields or {})
        forwarded["input_ids"] = input_ids
        forwarded["sampling_params"] = self._merged_sampling_params(sampling_overrides)
        forwarded["return_logprob"] = self._return_logprob

        try:
            result = await self._forward_generate(
                forwarded, routing_key_for(session.group_id), group.task_name
            )
        except Exception as exc:
            self.log_error(
                f"Session server generate forward failed (group "
                f"{session.group_id}, traj {session.traj_index}): {exc!r}"
            )
            return None, (500, f"forwarding to the router failed: {exc!r}")

        meta_info = result.get("meta_info") or {}
        output_ids = result.get("output_ids")
        if not isinstance(output_ids, list):
            return None, (
                500,
                f"the router response has no output_ids; got keys {sorted(result)}",
            )
        finish_reason = meta_info.get("finish_reason") or {}
        if finish_reason.get("type") == "abort":
            # HTTP 200 with a truncated output is how an engine-side abort
            # surfaces; half a turn must never reach the payload.
            return None, (
                503,
                "the engine aborted this generation; the turn failed and "
                "no payload was written",
            )
        weight_version = meta_info.get("weight_version")
        if weight_version == "default":
            return None, (
                500,
                'the engine reported weight_version "default": it has not '
                "been written a version yet; staleness cannot be computed",
            )
        try:
            weight_version = int(weight_version)
        except (TypeError, ValueError):
            return None, (
                500,
                f"the engine reported a non-integer weight_version {weight_version!r}",
            )

        # The group may have been discarded while this request was in
        # flight; a late response writes neither payload nor state.
        group = self._groups.get(session.group_id)
        if group is None or group.discard_reason is not None:
            self.log_warning(
                f"Dropping a generate response for discarded group "
                f"{session.group_id} (traj {session.traj_index})."
            )
            return None, (410, f"group {session.group_id} was discarded")

        logprobs = None
        if self._return_logprob:
            token_logprobs = meta_info.get("output_token_logprobs") or []
            logprobs = [float(item[0]) for item in token_logprobs]

        turn.output_len = len(output_ids)
        part = PayloadPart(
            group_id=session.group_id,
            traj_index=session.traj_index,
            turn_index=len(session.turns),
            input_ids=list(input_ids),
            output_ids=list(output_ids),
            logprobs=logprobs,
            version_start=version_start,
            weight_version=weight_version,
            finish_reason=dict(finish_reason),
            prev_kept_len=prev_kept_len,
        )
        await self._emit_part(session, part)
        if group.discard_reason is not None:
            # The settlement of this turn's put (inline backpressure) or of
            # an earlier one (a background task that ran during the await)
            # failed and voided the group: the turn is not recorded, so the
            # response is rejected like any other post-discard request.
            return None, (410, f"group {session.group_id} was discarded")
        session.turns.append(turn)
        if turn.continues_prev is False:
            # The arrival-time classification found a break inside the
            # previous input, which is a history rewrite. Counted here: a
            # turn a failed forward never records is no rewrite either.
            self._metrics["session_server/prefix_rewritten"] += 1
        session.last_input = list(input_ids)
        session.last_output = list(output_ids)
        if self._verify_assembly:
            turn.merged_sampling_params = forwarded["sampling_params"]
        group.num_gen_tokens += len(output_ids)
        group.num_generate_calls += 1
        self._apply_version(group, version_start)
        self._apply_version(group, weight_version)

        return (
            {
                "text": result.get("text"),
                "output_ids": list(output_ids),
                "finish_reason": dict(finish_reason),
                "weight_version": weight_version,
                "cached_tokens": meta_info.get("cached_tokens", 0),
                "logprobs": logprobs if self._verify_assembly else None,
            },
            None,
        )

    def _merged_sampling_params(self, overrides: Any) -> dict:
        """Config defaults merged with this call's overrides, in the order
        the rollout worker would have applied them."""
        params = copy.deepcopy(self._sampling_params)
        if isinstance(overrides, dict):
            params.update(overrides)
        return params

    async def _forward_generate(
        self, body: dict, routing_key: int, task_name: str
    ) -> dict:
        """POST to the router's /generate; connect failures are retried.

        Only ``ClientConnectorError`` is retried: the connection was never
        established, so the request was never sent and resending draws no new
        sample. Timeouts and every other error raise without a retry — a
        read timeout means the request may already have been sampled, so
        resending would re-roll the turn, and the failure goes back to the
        agent instead.
        """
        timeout_s = self._per_task_generate_timeout_s.get(
            task_name, self._generate_timeout_s
        )
        headers = {"X-SMG-Routing-Key": str(routing_key)}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await self._post_generate(body, headers, timeout_s)
            except aiohttp.ClientConnectorError as exc:
                self.log_warning(
                    f"Router connection error on generate attempt "
                    f"{attempt + 1}/{max_retries + 1} "
                    f"({type(exc).__name__}: {exc}), retrying"
                )
                await asyncio.sleep(0.5 * (attempt + 1))
        return await self._post_generate(body, headers, timeout_s)

    async def _post_generate(
        self, body: dict, headers: dict[str, str], timeout_s: float
    ) -> dict:
        """One /generate POST, with the forward-time metrics accounted."""
        started = asyncio.get_running_loop().time()
        result = await self._router_client.async_post(
            "/generate", body, headers=headers, timeout_s=timeout_s
        )
        elapsed = asyncio.get_running_loop().time() - started
        self._metrics["session_server/forward_count"] += 1
        self._metrics["session_server/forward_time_total_s"] += elapsed
        return result

    # ------------------------------------------------------------------
    # Chat completions entry
    # ------------------------------------------------------------------
    async def _chat_completions(self, sid: str, body: dict) -> Response:
        """One POST /sessions/{sid}/v1/chat/completions.

        The chat entry is the one that answers an OpenAI client, so its
        errors and its response body use the OpenAI shapes; between those
        two ends it renders the request, forwards it through the same
        generate core, and parses the engine's answer back into a message.
        """
        session, error = self._session(sid)
        if error is not None:
            return self._openai_error(*error, "unknown_session")
        if self._verify_assembly:
            # The assembly check compares against a reference the agent side
            # assembles from token ids; a chat client holds no tokens.
            return self._openai_error(
                400,
                "verify_assembly is enabled: the chat completions entry "
                "cannot provide a reference result",
                "verify_assembly_enabled",
            )
        if self._chat_entry is None:
            return self._openai_error(
                500,
                "this session server has no agentic tokenizer configured "
                "(agentloop.agentic_tokenizer); the chat entry needs one",
                "no_agentic_tokenizer",
            )
        reason = self._chat_entry.check_request(
            body, self._merged_sampling_params(None)
        )
        if reason is not None:
            return self._openai_error(400, reason, "invalid_request")
        if session.inflight:
            return self._openai_error(409, _SESSION_BUSY, "session_busy")
        session.inflight = True
        try:
            error = self._turn_refusal(session)
            if error is not None:
                return self._openai_error(*error, "generation_failed")
            if session.entry is None:
                session.entry = "chat"
            messages = body["messages"]
            tools = body.get("tools")
            rendered = self._chat_entry.render(
                session.chat,
                (session.last_input or []) + (session.last_output or []),
                messages,
                tools,
            )
            # A turn's prompt and answer together have to fit in the length
            # the actor trains on. A chat client cannot know that limit, so
            # the entry holds it: the answer gets whatever room the prompt
            # leaves, and a prompt that does not fit is refused before
            # anything is recorded.
            max_len = self._session_max_len(session)
            if len(rendered.input_ids) >= max_len:
                return self._openai_error(
                    400,
                    f"the prompt has {len(rendered.input_ids)} tokens and the "
                    f"training sequence length is {max_len}",
                    "context_length_exceeded",
                )
            overrides = self._chat_entry.sampling_overrides(body)
            overrides["max_new_tokens"] = min(
                int(self._merged_sampling_params(overrides)["max_new_tokens"]),
                max_len - len(rendered.input_ids),
            )
            if rendered.rollback:
                self._roll_back_last_turn(session)
            # Captured before the forward: this is where a later retry of
            # this request rolls back to.
            checkpoint = ChatCheckpoint(
                messages=(
                    list(session.chat.messages) if session.chat is not None else []
                ),
                last_input=session.last_input,
                last_output=session.last_output,
                tools=session.chat.tools if session.chat is not None else tools,
            )
            result, error = await self._generate_core(
                session,
                list(rendered.input_ids),
                overrides,
                None,
            )
            if error is not None:
                return self._openai_error(*error, "generation_failed")
            text = result.get("text")
            if text is None:
                # The engines were started without a tokenizer
                # (rollout.detokenize off), so the response has no text to
                # parse: decode the ids here instead.
                text = self._chat_entry.decode_output(
                    result["output_ids"],
                    (result.get("finish_reason") or {}).get("matched"),
                )
            parsed = self._chat_entry.parse(
                text,
                rendered.input_ids,
                tools,
                body.get("tool_choice") or "auto",
            )
            if parsed.has_reasoning:
                self._metrics["session_server/chat_reasoning_responses"] += 1
            if parsed.has_tool_calls:
                self._metrics["session_server/chat_tool_call_responses"] += 1
            if parsed.tool_call_parse_failed:
                self._metrics["session_server/chat_tool_call_parse_failed"] += 1
            session.chat = ChatConversation(
                messages=[*rendered.messages, parsed.message],
                tools=tools,
                checkpoint=checkpoint,
            )
            response = self._chat_entry.response_body(
                body, parsed, result, rendered.input_ids
            )
            if body.get("stream"):
                return Response(
                    content=self._chat_entry.stream_body(response),
                    status_code=200,
                    media_type="text/event-stream",
                    headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
                )
            return JSONResponse(content=response)
        finally:
            session.inflight = False

    def _session_max_len(self, session: _SessionState) -> int:
        """The longest sequence this session's turns may produce.

        The one place a length limit is decided, so a task that needs its
        own (a per-task key beside the generate timeout, for example) is
        added here alone. Today every session gets the length the actor
        trains on.
        """
        return self._max_seq_length

    def _roll_back_last_turn(self, session: _SessionState) -> None:
        """Drop the session's last turn from the training data.

        The rollback happens before the forward, and a failure of the
        retried request does not undo it: the request repeating the previous
        one means the agent never used that turn's answer, so that turn must
        not train whatever the retry does. The conversation and
        ``last_input`` / ``last_output`` are restored from the checkpoint
        here because the generate core derives the new turn's prefix match
        from them, and the match must be taken against the kept turn, not
        the abandoned one. The checkpoint is consumed: restoring it again
        would roll back the same turn twice, and a request that needs to go
        back further re-renders the whole conversation. The abandoned turn
        keeps its payload: the engine already sampled it.
        """
        checkpoint = session.chat.checkpoint
        session.not_training_turns.add(len(session.turns) - 1)
        session.chat = ChatConversation(
            messages=list(checkpoint.messages),
            tools=checkpoint.tools,
            checkpoint=None,
        )
        session.last_input = (
            None if checkpoint.last_input is None else list(checkpoint.last_input)
        )
        session.last_output = (
            None if checkpoint.last_output is None else list(checkpoint.last_output)
        )

    # ------------------------------------------------------------------
    # Payload writes
    # ------------------------------------------------------------------
    async def _emit_part(self, session: _SessionState, part: PayloadPart) -> None:
        """Emit one turn's part and register its settlement on the group.

        The write itself is the payload writer's; the group's byte count and
        the task finalize waits for are the session's state, so they are
        recorded here.
        """
        group = self._groups.get(part.group_id)
        if group is None:
            return
        nbytes, task = await self._payload_writer.emit_part(group.payload_shard, part)
        group.payload_bytes += nbytes
        if task is not None and not group.finalized:
            group.pending_settles.append(task)
            task.add_done_callback(group.pending_settles.remove)

    async def _note_write_failure(self, group_id: int) -> None:
        """Flag a group's payload as not fully written, voiding it if live.

        The flag is recorded whatever the group's state: a settlement can
        fail while finalize is already waiting, and finalize must see the
        failure. Only a group whose completion metadata was already sent is
        left alone — its settlements were all confirmed before reporting, so
        this is defensive only.
        """
        group = self._groups.get(group_id)
        if group is None:
            return
        group.write_failed = True
        if group.reported:
            self.log_warning(
                f"Group {group_id} was reported complete but a payload write "
                f"failed afterwards."
            )
            return
        if group.discard_reason is not None:
            return
        await self._void_group(group_id, _WRITE_FAILED)

    # ------------------------------------------------------------------
    # Lifecycle routes
    # ------------------------------------------------------------------
    async def _close_session(self, sid: str, body: dict) -> JSONResponse:
        if sid in self._reported_sessions:
            # A close resent after its response was lost: the group it
            # belonged to is already delivered, so nothing is left to close.
            return JSONResponse(content={})
        session, error = self._session(sid)
        if error is not None:
            return self._error(*error)
        group, error = self._live_group(session)
        if group is None:
            status, message = error
            if status < 400:
                return JSONResponse(content={})
            return self._error(status, message)

        if session.inflight:
            return self._error(409, _SESSION_BUSY)
        if session.traj_index in group.closed:
            # A close retried after its response was lost, or sent twice: the
            # trajectory is already closed, so there is nothing to classify.
            return JSONResponse(content={})

        turns = session.turns
        verdict = body.get("verdict") or {}
        if not isinstance(verdict, dict):
            return self._error(400, f"verdict must be an object, got {verdict!r}")
        unsupported = sorted(set(verdict) - {"extra_fields_train"})
        if unsupported:
            return self._error(
                400,
                f"unsupported verdict fields {unsupported!r}: only "
                f"extra_fields_train is supported; kept lengths are "
                f"declared via kept_lens",
            )
        extra_fields_train = verdict.get("extra_fields_train") or {}
        if not isinstance(extra_fields_train, dict):
            return self._error(
                400,
                f"verdict extra_fields_train must map a field name to a list, "
                f"got {extra_fields_train!r}",
            )
        trained_turn_count = len(turns) - len(session.not_training_turns)
        if trained_turn_count == 0:
            return self._error(
                400,
                f"trajectory {session.traj_index} of group {session.group_id} "
                f"has no trained turn; abort the group instead of closing it",
            )
        for name, values in extra_fields_train.items():
            if (
                not isinstance(values, list)
                or len(values) != trained_turn_count
                or any(not isinstance(value, (bool, int, float)) for value in values)
            ):
                return self._error(
                    400,
                    f"unsupported verdict extra_fields_train {name!r}: each field "
                    f"must be a list of int / float / bool, one value per trained "
                    f"turn of this trajectory, got {values!r}",
                )
        kept_lens = body.get("kept_lens") or {}
        if not isinstance(kept_lens, dict):
            return self._error(400, "kept_lens must map turn_index to a kept length")
        is_end = body.get("is_end") or {}
        if not isinstance(is_end, dict):
            return self._error(400, "is_end must map turn_index to a bool")
        for turn_index, value in is_end.items():
            if not isinstance(value, bool):
                return self._error(
                    400, f"is_end[{turn_index}] must be a bool, got {value!r}"
                )
        verify_result = body.get("verify_result")
        if verify_result is not None:
            if not isinstance(verify_result, dict) or not isinstance(
                verify_result.get("result"), dict
            ):
                return self._error(
                    400,
                    "verify_result must be an object with a 'result' object; "
                    f"got {verify_result!r}",
                )
            # Expected with the last close of the group; kept until finalize.
            group.verify_result = verify_result

        try:
            kept = [kept_len(i, turns, kept_lens) for i in range(len(turns))]
            continues, counts = classify_turns(
                turns,
                kept,
                session.not_training_turns,
                session.entry,
                session.traj_index,
                session.group_id,
            )
        except UnderreportedTruncation as exc:
            self._metrics["session_server/prefix_underreported"] += 1
            # Nothing is committed: a retried close re-runs the full check
            # instead of skipping over half-classified state.
            return self._error(500, str(exc))
        except ValueError as exc:
            return self._error(500, str(exc))
        # Counted once the classification has committed: a close rejected by
        # the checks above is retried and re-runs the full pass.
        self._metrics["session_server/prefix_continuous"] += counts.continuous
        self._metrics["session_server/prefix_rewritten"] += counts.rewritten
        for turn_index, turn in enumerate(turns):
            turn.kept_len = kept[turn_index]
            turn.continues_prev = continues[turn_index]
            # JSON keys arrive as strings, so look up both forms, as kept_len
            # does; a turn the agent declares nothing about did not end.
            turn.is_end = is_end.get(turn_index, is_end.get(str(turn_index), False))
        session.extra_fields_train = extra_fields_train

        group.closed.add(session.traj_index)
        await self._maybe_finalize(group)
        return JSONResponse(content={})

    async def _reward(self, sid: str, body: dict) -> JSONResponse:
        if sid in self._reported_sessions:
            # A reward resent after its response was lost: the group is
            # already delivered, so nothing is left to record.
            return JSONResponse(content={"group_complete": True})
        session, error = self._session(sid)
        if error is not None:
            return self._error(*error)
        group, error = self._live_group(session)
        if group is None:
            status, message = error
            if status < 400:
                return JSONResponse(content={})
            return self._error(status, message)

        reward = body.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            return self._error(400, "reward must be a number")
        group.rewarded.add(session.traj_index)
        group.rewards[session.traj_index] = float(reward)
        group_complete = await self._maybe_finalize(group)
        return JSONResponse(content={"group_complete": group_complete})

    async def _abort(self, group_id: int, body: dict) -> JSONResponse:
        reason = body.get("reason")
        if not isinstance(reason, str) or not reason:
            return self._error(400, "reason must be a non-empty string")
        group = self._groups.get(group_id)
        if group is None:
            if group_id in self._reported_groups:
                # Already reported complete; a second record would make the
                # buffer worker fail the run.
                return JSONResponse(content={})
            # The buffer worker admitted the group, so an abort can arrive
            # before any session does (a failed preprocess, say). It still
            # gets a discard record — a 404 here would leak the admission
            # slot until the run ends.
            group = _GroupState(
                group_id=group_id,
                task_name="",
                group_size=0,
                payload_shard=self._take_payload_shard(),
                finalized=True,
            )
            group.discard_reason = reason
            group.discarded_at = time.monotonic()
            self._groups[group_id] = group
            await self._report_discard(group_id, reason)
            return JSONResponse(content={})
        if group.reported or group.finalized or group.discard_reason is not None:
            # First discard wins: a second discard record would make the
            # buffer worker fail the run, and an already-finalized group has
            # nothing left to discard.
            return JSONResponse(content={})
        await self._void_group(group_id, reason)
        return JSONResponse(content={})

    # ------------------------------------------------------------------
    # Finalization and discard
    # ------------------------------------------------------------------
    def _complete(self, group: _GroupState) -> bool:
        return (
            not group.finalized
            and group.discard_reason is None
            and group.closed == set(group.sessions)
            and group.rewarded == set(group.sessions)
            and len(group.sessions) == group.group_size
        )

    async def _maybe_finalize(self, group: _GroupState) -> bool:
        """Finalize the group when close and reward both cover it, and report.

        Every put the group has issued is awaited first, and the group's
        write-failure flag consulted, so the metadata is sent only for a
        payload that is complete and confirmed written — a finalize part
        whose put settled tells nothing about the success of the puts
        before it. The finalize part is then sent and awaited; on any
        failure the group is voided and any container the part produced is
        dropped. On success the group's state is dropped: the group is
        finished, and its sessions answer unknown from here on.
        """
        if not self._complete(group):
            return False
        group.finalized = True
        group_id = group.group_id
        # Wait for every put this group has issued. Same-key channel work is
        # serial, so this costs no reordering — but the last one's completion
        # only means the earlier ones ended, not that they succeeded, so the
        # per-group failure flag is checked afterwards.
        pending = [t for t in group.pending_settles if not t.done()]
        if pending:
            await asyncio.gather(*pending)
        if group.write_failed:
            await self._void_group(group_id, _WRITE_FAILED)
            return False
        verify = None
        if self._verify_assembly:
            if group.verify_result is None:
                raise RuntimeError(
                    f"verify_assembly is enabled but no close carried a "
                    f"verify_result for group {group_id}"
                )
            verify = dict(group.verify_result)
            verify["merged_sampling_params"] = {
                traj_index: [turn.merged_sampling_params for turn in session.turns]
                for traj_index, session in group.sessions.items()
            }
        verdicts = {
            traj_index: {
                "kept_lens": {
                    turn_index: turn.kept_len
                    for turn_index, turn in enumerate(session.turns)
                },
                "continues_prev": [
                    turn.continues_prev is True for turn in session.turns
                ],
                "is_end": {
                    turn_index: turn.is_end
                    for turn_index, turn in enumerate(session.turns)
                },
                "not_training": dict.fromkeys(session.not_training_turns, True),
                "extra_fields_train": dict(session.extra_fields_train),
            }
            for traj_index, session in group.sessions.items()
        }
        finalize_part = PayloadPart(
            group_id=group_id,
            traj_index=-1,
            turn_index=-1,
            input_ids=[],
            output_ids=[],
            logprobs=None,
            version_start=group.earliest_version,
            weight_version=group.completion_version,
            finish_reason={},
            last=True,
            rewards=list(group.rewards),
            verdicts=verdicts,
            verify=verify,
        )
        if not await self._payload_writer.put(
            group.payload_shard,
            group_id,
            finalize_part,
            await_completion=True,
            weight=group.payload_bytes,
        ):
            # The writer already reported the failure, which voided the group
            # and removed any container the part produced.
            return False
        payload_channel = self._payload_writer.channel_for(group.payload_shard)
        metadata = GroupMetadata(
            group_id=group_id,
            task_name=group.task_name,
            rewards=list(group.rewards),
            earliest_version=group.earliest_version,
            completion_version=group.completion_version,
            num_gen_tokens=group.num_gen_tokens,
            num_generate_calls=group.num_generate_calls,
            payload_ref=PayloadRef(
                locator=payload_locator_for(payload_channel),
                key=payload_key_for(group_id),
            ),
        )
        try:
            work = self._agent_output_channel.put(metadata, async_op=True)
            await asyncio.wait_for(
                work.async_wait(), timeout=self._payload_put_timeout_s
            )
        except BaseException as exc:
            # The payload container exists but its consumer will never be
            # told. Drop the container; the dead agent_output_channel makes
            # every future result equally undeliverable, so the run fails
            # instead of leaking admission slot after slot.
            await self._payload_writer.remove_key(group.payload_shard, group_id)
            self._fail_serving(f"GroupMetadata put for group {group_id}", exc)
            return False
        group.reported = True
        session_ids = tuple(
            session.session_id
            for session in group.sessions.values()
            if session.session_id is not None
        )
        self._remember_reported(group_id, session_ids)
        for sid in session_ids:
            self._sessions.pop(sid, None)
        self._groups.pop(group_id, None)
        return True

    async def _void_group(self, group_id: int, reason: str) -> None:
        """Mark a group voided, report it, and strip it to the bare record.

        The record — group identity, reason, timestamp — stays so late
        traffic for the group is answered (410 for write failures, drop for
        aborts) and a straggler abort cannot produce a second record; the
        token buffers and the declared turn state go. The record itself
        expires after its TTL (see ``_expire_discard_records``). First
        voiding only.
        """
        group = self._groups.get(group_id)
        if group is None or group.reported or group.discard_reason is not None:
            return
        group.discard_reason = reason
        group.discarded_at = time.monotonic()
        group.closed.clear()
        group.rewarded.clear()
        group.rewards = []
        for session in group.sessions.values():
            session.turns.clear()
            session.last_input = None
            session.last_output = None
            session.chat = None
            session.extra_fields_train.clear()
            session.not_training_turns.clear()
        group.verify_result = None
        if reason == _WRITE_FAILED:
            self._metrics["session_server/payload_write_failed"] += 1
        await self._report_discard(group_id, reason)
        await self._payload_writer.put(
            group.payload_shard,
            group_id,
            PayloadDiscard(group_id=group_id),
            await_completion=False,
        )
        if reason == _WRITE_FAILED:
            # A part can land although its settlement failed or timed out;
            # with the finalize part, the collector will then have emitted a
            # container no one will fetch. remove_key chains after all puts
            # already issued for the key and drops the whole queue.
            await self._payload_writer.remove_key(group.payload_shard, group_id)

    async def _report_discard(self, group_id: int, reason: str) -> None:
        """Report a discard record to the buffer worker, awaited."""
        try:
            work = self._agent_output_channel.put(
                DiscardedGroup(group_id=group_id, reason=reason), async_op=True
            )
            await asyncio.wait_for(
                work.async_wait(), timeout=self._payload_put_timeout_s
            )
        except BaseException as exc:
            # The admission slot this record would have released leaks, and
            # so will every later group's: the agent_output_channel is dead.
            self._fail_serving(f"DiscardedGroup put for group {group_id}", exc)

    def _fail_serving(self, context: str, exc: BaseException) -> None:
        """Record a fatal serving error and stop the HTTP server.

        For an agent_output_channel that cannot be written: quietly leaking
        admission slots would decay throughput until everything stalls, so
        the run ends instead. The first error is kept; ``serve()`` re-raises
        it once the uvicorn task has ended, and the runner's liveness
        monitor terminates the run.
        """
        if self._fatal_error is None:
            self._fatal_error = RuntimeError(f"{context} failed: {exc!r}")
        self.log_error(f"{context} failed ({exc!r}); stopping the session server.")
        self._http_server.should_exit = True

    def _take_payload_shard(self) -> int:
        """The shard for the next group, taken in creation order."""
        shard = self._next_payload_shard
        self._next_payload_shard = (shard + 1) % self._payload_writer.num_shards
        return shard

    def _remember_reported(self, group_id: int, session_ids: tuple[str, ...]) -> None:
        """Track recently reported groups and their sessions, bounded.

        The group state is gone once it is delivered, so this history is the
        only memory of it: a straggler abort for one of them must not produce
        a second record, which would make the buffer worker fail the run, and
        a close or reward whose success response was lost is resent by the
        client, so it must still be answered with success. Groups are dropped
        oldest first, together with their session ids.
        """
        self._reported_groups[group_id] = session_ids
        for sid in session_ids:
            self._reported_sessions[sid] = group_id
        while len(self._reported_groups) > _REPORTED_HISTORY_LIMIT:
            evicted = next(iter(self._reported_groups))
            for sid in self._reported_groups.pop(evicted):
                self._reported_sessions.pop(sid, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _apply_version(self, group: _GroupState, version: Optional[int]) -> None:
        if version is None:
            return
        if group.earliest_version is None or version < group.earliest_version:
            group.earliest_version = version
        if group.completion_version is None or version > group.completion_version:
            group.completion_version = version

    @staticmethod
    def _int_field(body: dict, name: str) -> int:
        value = body.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an int, got {value!r}")
        return value

    @staticmethod
    def _error(status_code: int, message: str) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"error": message})

    @staticmethod
    def _openai_error(status_code: int, message: str, code: str) -> JSONResponse:
        """An error body of the OpenAI shape, for the chat entry."""
        error_type = "invalid_request_error" if status_code < 500 else "server_error"
        return JSONResponse(
            status_code=status_code,
            content={"error": {"message": message, "type": error_type, "code": code}},
        )
