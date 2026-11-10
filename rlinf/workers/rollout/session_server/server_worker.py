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
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from omegaconf import DictConfig

from rlinf.data.schema.reasoning_results import (
    DiscardedGroup,
    GroupMetadata,
    PayloadRef,
)
from rlinf.scheduler import Channel, Worker
from rlinf.utils.http_client import InferenceHTTPClient
from rlinf.workers.rollout.session_server.payload import (
    PayloadDiscard,
    PayloadPart,
    payload_key_for,
    payload_locator_for,
    payload_shard_for,
    routing_key_for,
)
from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

# Rough per-element byte cost of a serialized part's token and logprob lists;
# used for the pending-bytes backpressure and the payload_bytes report, which
# need an estimate, not an exact size.
_BYTES_PER_LIST_ELEMENT = 8
_PART_OVERHEAD_BYTES = 256

_WRITE_FAILED = "payload_write_failed"

# Bounded history of groups whose completion metadata was sent; guards a
# straggler abort from producing a second record for the buffer worker.
_REPORTED_HISTORY_LIMIT = 4096


@dataclass
class _TurnRecord:
    """Prefix bookkeeping for one successful generate call of a session.

    ``lcp`` is the longest common prefix length of this turn's input with the
    previous turn's input+output, None on the first turn. ``kept_len`` and
    ``continues_prev`` are filled when the agent declares them (close, or the
    per-turn ``prev_kept_len`` declaration); until then the kept length is the
    full output and the classification is pending. In verify mode
    ``merged_sampling_params`` holds the params this call was forwarded with.
    """

    input_len: int
    output_len: int = 0
    lcp: Optional[int] = None
    declared_prev_kept_len: Optional[int] = None
    kept_len: Optional[int] = None
    continues_prev: Optional[bool] = None
    merged_sampling_params: Optional[dict] = None


@dataclass
class _SessionState:
    """Per-trajectory state: the turn records and the last engine I/O.

    The last input and output token ids are kept only for the next turn's
    prefix check; logprobs never live here. ``extra_fields_train`` is the
    trajectory's slice of the training extra fields, declared at close and
    moved into the finalize part's verdicts.
    """

    session_id: Optional[str]
    group_id: int
    traj_index: int
    turns: list[_TurnRecord] = field(default_factory=list)
    last_input: Optional[list[int]] = None
    last_output: Optional[list[int]] = None
    extra_fields_train: dict = field(default_factory=dict)


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
        self._max_pending_payload_bytes = int(
            session_server_cfg.max_pending_payload_bytes
        )
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
        self._pending_payload_bytes = 0
        self._router_client: Optional[InferenceHTTPClient] = None
        self._agent_output_channel: Optional[Channel] = None
        self._payload_channels: list[Channel] = []
        self._http_server: Optional[uvicorn.Server] = None
        self._http_server_task: Optional[asyncio.Task] = None
        # Set when serving has ended (normally or not), with the exception
        # serve() re-raised, if any: wait_until_serving reports an early
        # end instead of hanging on a server that never listened.
        self._serve_ended = False
        self._serve_error: Optional[BaseException] = None
        self._background_tasks: set[asyncio.Task] = set()
        # Reported groups, most recent last, bounded; a late abort for one of
        # them must not produce a second record for the buffer worker.
        self._reported_group_ids: set[int] = set()
        self._reported_group_order: deque = deque()
        self._fatal_error: Optional[RuntimeError] = None
        self._metrics = {
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
        async def handle_chat_completions():
            return JSONResponse(
                status_code=501,
                content={
                    "error": "the chat-completions entry is reserved but not implemented"
                },
            )

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
            payload_channels: The same-node payload channels, one per shard,
                indexed by :func:`payload_shard_for`.
        """
        assert self._http_server_task is None, "serve is already running."
        assert payload_channels, "serve needs at least one payload channel."
        self._router_client = InferenceHTTPClient(router_url, keepalive_timeout=4.0)
        self._agent_output_channel = agent_output_channel
        self._payload_channels = list(payload_channels)
        self._http_server_task = asyncio.create_task(self._run_http_server())
        self.log_info(
            f"Session server {self._server_index} serving on {self._host}:{self._port}"
        )
        try:
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
        for task in list(self._background_tasks):
            task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
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
        metrics["session_server/pending_payload_bytes"] = self._pending_payload_bytes
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
    ) -> tuple[Optional[_SessionState], Optional[JSONResponse]]:
        session = self._sessions.get(sid)
        if session is None:
            return None, self._error(404, f"unknown session {sid}")
        return session, None

    def _live_group(
        self, session: _SessionState
    ) -> tuple[Optional[_GroupState], Optional[JSONResponse]]:
        """The group of a session, or the error to answer with.

        A group discarded by an abort drops everything that arrives late; one
        discarded by a failed payload write answers generate and close with
        an error so the caller learns its trajectory was not recorded.
        """
        group = self._groups.get(session.group_id)
        if group is None:
            return None, self._error(
                404, f"group {session.group_id} is unknown or already finished"
            )
        if group.discard_reason is not None:
            if group.discard_reason == _WRITE_FAILED:
                return None, self._error(
                    410,
                    f"group {session.group_id} was discarded ({_WRITE_FAILED})",
                )
            return None, JSONResponse(content={})
        return group, None

    # ------------------------------------------------------------------
    # Data plane
    # ------------------------------------------------------------------
    async def _generate(self, sid: str, body: dict) -> JSONResponse:
        session, error = self._session(sid)
        if error is not None:
            return error
        _, error = self._live_group(session)
        if error is not None:
            # An abort-discarded group answers 410 so the agent fails its
            # turn; other error responses pass through.
            return (
                error
                if error.status_code >= 400
                else self._error(410, f"group {session.group_id} was discarded")
            )

        input_ids = body.get("input_ids")
        if not isinstance(input_ids, list) or not input_ids:
            return self._error(400, "input_ids must be a non-empty list")
        rlinf_block = body.get("rlinf") or {}
        prev_kept_len = rlinf_block.get("prev_kept_len")
        if prev_kept_len is not None and (
            not isinstance(prev_kept_len, int) or prev_kept_len < 0
        ):
            return self._error(400, "rlinf.prev_kept_len must be a non-negative int")
        if not self._loaded_versions:
            return self._error(
                500,
                "no loaded weight versions were recorded; the runner must "
                "call set_loaded_versions before the first generate",
            )

        group = self._groups[session.group_id]
        version_start = min(self._loaded_versions)
        turn = self._new_turn_record(session, input_ids, prev_kept_len)

        forwarded = {key: value for key, value in body.items() if key != "rlinf"}
        forwarded["sampling_params"] = self._merged_sampling_params(
            body.get("sampling_params")
        )
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
            return self._error(500, f"forwarding to the router failed: {exc!r}")

        meta_info = result.get("meta_info") or {}
        output_ids = result.get("output_ids")
        if not isinstance(output_ids, list):
            return self._error(
                500,
                f"the router response has no output_ids; got keys {sorted(result)}",
            )
        finish_reason = meta_info.get("finish_reason") or {}
        if finish_reason.get("type") == "abort":
            # HTTP 200 with a truncated output is how an engine-side abort
            # surfaces; half a turn must never reach the payload.
            return self._error(
                503,
                "the engine aborted this generation; the turn failed and "
                "no payload was written",
            )
        weight_version = meta_info.get("weight_version")
        if weight_version == "default":
            return self._error(
                500,
                'the engine reported weight_version "default": it has not '
                "been written a version yet; staleness cannot be computed",
            )
        try:
            weight_version = int(weight_version)
        except (TypeError, ValueError):
            return self._error(
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
            return self._error(410, f"group {session.group_id} was discarded")

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
            return self._error(410, f"group {session.group_id} was discarded")
        session.turns.append(turn)
        session.last_input = list(input_ids)
        session.last_output = list(output_ids)
        if self._verify_assembly:
            turn.merged_sampling_params = forwarded["sampling_params"]
        group.num_gen_tokens += len(output_ids)
        group.num_generate_calls += 1
        self._apply_version(group, version_start)
        self._apply_version(group, weight_version)

        content = {
            "output_ids": list(output_ids),
            "finish_reason": dict(finish_reason),
            "weight_version": weight_version,
            "cached_tokens": meta_info.get("cached_tokens", 0),
        }
        if self._verify_assembly:
            # Only the verification needs the agent-side logprobs; normal
            # responses keep them out of the data plane.
            content["logprobs"] = logprobs
        return JSONResponse(content=content)

    def _new_turn_record(
        self, session: _SessionState, input_ids: list[int], prev_kept_len: Optional[int]
    ) -> _TurnRecord:
        """The turn's prefix bookkeeping, classified now where possible.

        The lcp with the previous turn's input+output is computed against the
        ids still held for the previous turn. A break inside the previous
        input is a history rewrite and classifies immediately; with a
        declared ``prev_kept_len`` the classification completes now too.
        """
        if session.last_input is None:
            return _TurnRecord(
                input_len=len(input_ids), declared_prev_kept_len=prev_kept_len
            )
        previous = session.last_input + (session.last_output or [])
        lcp = 0
        for cur, prev in zip(input_ids, previous):
            if cur != prev:
                break
            lcp += 1
        record = _TurnRecord(
            input_len=len(input_ids),
            lcp=lcp,
            declared_prev_kept_len=prev_kept_len,
        )
        if lcp < len(session.last_input):
            record.continues_prev = False
            self._metrics["session_server/prefix_rewritten"] += 1
        return record

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
    # Payload writes
    # ------------------------------------------------------------------
    async def _emit_part(self, session: _SessionState, part: PayloadPart) -> None:
        """Put one part in the background, with backpressure.

        The put is issued synchronously here (issuance order is the queue
        order); it is awaited only when the pending byte budget is exceeded.
        Its settlement runs as a task that flags the group on failure.
        """
        nbytes = self._estimate_part_bytes(part)
        group = self._groups.get(part.group_id)
        channel = self._payload_channels[
            payload_shard_for(part.group_id, len(self._payload_channels))
        ]
        # Every-turn parts carry no weight: the byte estimate rides on the
        # finalize part's put (_maybe_finalize), the queue element the
        # runner's payload_queued_bytes gauge sums.
        work = channel.put(part, key=payload_key_for(part.group_id), async_op=True)
        self._pending_payload_bytes += nbytes
        if group is not None:
            group.payload_bytes += nbytes
        if self._pending_payload_bytes > self._max_pending_payload_bytes:
            await self._settle_put(part.group_id, nbytes, work)
        else:
            task = asyncio.create_task(self._settle_put(part.group_id, nbytes, work))
            self._register_settle(part.group_id, task)

    def _register_settle(self, group_id: int, task: asyncio.Task) -> None:
        """Track a settlement task for cancellation and for finalize's wait."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        group = self._groups.get(group_id)
        if group is not None and not group.finalized:
            group.pending_settles.append(task)
            task.add_done_callback(group.pending_settles.remove)

    @staticmethod
    def _estimate_part_bytes(part: PayloadPart) -> int:
        nbytes = (len(part.input_ids) + len(part.output_ids)) * _BYTES_PER_LIST_ELEMENT
        if part.logprobs is not None:
            nbytes += len(part.logprobs) * _BYTES_PER_LIST_ELEMENT
        return nbytes + _PART_OVERHEAD_BYTES

    async def _settle_put(self, group_id: int, nbytes: int, work: Any) -> bool:
        """Wait for one payload put to settle.

        Returns:
            True if the put completed successfully, False if it failed or
            timed out. A timed-out put may still land in the channel later,
            so the failure flag this sets is never cleared again.

        On failure the group's write-failure flag is recorded (and the group
        voided if it is still live); the return value lets an awaiting caller
        react as well.
        """
        success = True
        try:
            await asyncio.wait_for(
                work.async_wait(), timeout=self._payload_put_timeout_s
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            success = False
            self.log_error(f"Payload put failed for group {group_id} ({exc!r}).")
        finally:
            self._pending_payload_bytes -= nbytes
        if not success:
            await self._note_write_failure(group_id)
        return success

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

    async def _put_payload(
        self, group_id: int, item: Any, await_completion: bool, weight: int = 0
    ) -> bool:
        """Put a payload-channel item under the group's key.

        Args:
            group_id: Group whose key the item is put under.
            item: A :class:`PayloadPart` (the finalize part, awaited) or a
                :class:`PayloadDiscard` marker (background).
            await_completion: True to await the put's settlement and return
                its success; False to monitor it in the background and
                return True.
            weight: Weight handed to the put; the finalize part carries the
                group's byte estimate so the runner's payload_queued_bytes
                gauge sums it.

        Returns:
            The settlement result when awaited, True when backgrounded.
        """
        channel = self._payload_channels[
            payload_shard_for(group_id, len(self._payload_channels))
        ]
        work = channel.put(
            item, weight=weight, key=payload_key_for(group_id), async_op=True
        )
        if await_completion:
            return await self._settle_put(group_id, 0, work)
        task = asyncio.create_task(self._settle_put(group_id, 0, work))
        self._register_settle(group_id, task)
        return True

    async def _remove_payload_key(self, group_id: int) -> None:
        """Drop the group's payload queue, after everything already issued.

        Chains on the same key as the puts (AsyncChannelWork order), so this
        runs after the finalize part and discard marker already queued. It
        matters when a finalize part landed although its settlement timed
        out: the collector emitted a container no one will fetch.
        """
        channel = self._payload_channels[
            payload_shard_for(group_id, len(self._payload_channels))
        ]
        try:
            work = channel.remove_key(key=payload_key_for(group_id), async_op=True)
            await asyncio.wait_for(
                work.async_wait(), timeout=self._payload_put_timeout_s
            )
        except BaseException as exc:
            self.log_warning(
                f"remove_key for group {group_id} failed ({exc!r}); its "
                f"payload queue may hold an unfetched container."
            )

    # ------------------------------------------------------------------
    # Lifecycle routes
    # ------------------------------------------------------------------
    async def _close_session(self, sid: str, body: dict) -> JSONResponse:
        session, error = self._session(sid)
        if error is not None:
            return error
        group, error = self._live_group(session)
        if group is None:
            status = error.status_code
            return error if status >= 400 else JSONResponse(content={})

        turns = session.turns
        verdict = body.get("verdict") or {}
        if not isinstance(verdict, dict):
            return self._error(400, f"verdict must be an object, got {verdict!r}")
        unsupported = sorted(set(verdict) - {"extra_fields_train"})
        if unsupported:
            return self._error(
                400,
                f"unsupported verdict fields {unsupported!r}: only "
                f"extra_fields_train is supported; kept lengths are declared "
                f"via kept_lens",
            )
        extra_fields_train = verdict.get("extra_fields_train") or {}
        if not isinstance(extra_fields_train, dict):
            return self._error(
                400,
                f"verdict extra_fields_train must map a field name to a list, "
                f"got {extra_fields_train!r}",
            )
        for name, values in extra_fields_train.items():
            if (
                not isinstance(values, list)
                or len(values) != len(turns)
                or any(not isinstance(value, (bool, int, float)) for value in values)
            ):
                return self._error(
                    400,
                    f"unsupported verdict extra_fields_train {name!r}: each field "
                    f"must be a list of int / float / bool, one value per turn "
                    f"of this trajectory, got {values!r}",
                )
        kept_lens = body.get("kept_lens") or {}
        if not isinstance(kept_lens, dict):
            return self._error(400, "kept_lens must map turn_index to a kept length")
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
            kept = [self._kept_len(i, turns, kept_lens) for i in range(len(turns))]
            continues = self._classify_turns(session, kept)
        except ValueError as exc:
            # Nothing is committed: a retried close re-runs the full check
            # instead of skipping over half-classified state.
            return self._error(500, str(exc))
        for turn_index, turn in enumerate(turns):
            turn.kept_len = kept[turn_index]
            turn.continues_prev = continues[turn_index]
        session.extra_fields_train = extra_fields_train

        group.closed.add(session.traj_index)
        await self._maybe_finalize(group)
        return JSONResponse(content={})

    @staticmethod
    def _kept_len(turn_index: int, turns: list[_TurnRecord], kept_lens: dict) -> int:
        """The declared kept length of a turn, defaulting to the full output."""
        raw = kept_lens.get(turn_index, kept_lens.get(str(turn_index)))
        turn = turns[turn_index]
        if raw is None:
            return turn.output_len
        if not isinstance(raw, int) or raw < 0:
            raise ValueError(
                f"kept_lens[{turn_index}] must be a non-negative int, got {raw!r}"
            )
        if raw > turn.output_len:
            raise ValueError(
                f"kept_lens[{turn_index}] is {raw} but the turn's output has "
                f"only {turn.output_len} tokens"
            )
        return raw

    def _classify_turns(self, session: _SessionState, kept: list[int]) -> list[bool]:
        """Classify each turn pair, failing loud on under-reporting.

        For turn k the pair is its ``lcp`` against turn k-1's input length
        ``L`` and kept length ``kept``: ``lcp >= L + kept`` continues the
        previous turn, ``L <= lcp < L + kept`` means the agent used less
        output than it declared (an under-reported truncation), and
        ``lcp < L`` rewrites history and starts a new segment. A turn whose
        input already broke inside the previous input was classified as a
        rewrite when it arrived.

        Args:
            session: The session whose turns are classified.
            kept: The kept length of each turn, as declared at close.

        Returns:
            ``continues_prev`` per turn; turn 0 is False.

        Raises:
            ValueError: On an under-reported truncation, or a per-turn
                ``prev_kept_len`` declaration that disagrees with the close
                declaration. No state is changed.
        """
        turns = session.turns
        continues: list[bool] = []
        continuous = 0
        rewritten = 0
        for turn_index, turn in enumerate(turns):
            if turn.continues_prev is not None:
                continues.append(turn.continues_prev)
                continue
            if turn_index == 0:
                continues.append(False)
                continue
            previous = turns[turn_index - 1]
            boundary = previous.input_len + kept[turn_index - 1]
            if turn.lcp >= boundary:
                continues.append(True)
                continuous += 1
            elif turn.lcp >= previous.input_len:
                self._metrics["session_server/prefix_underreported"] += 1
                raise ValueError(
                    f"turn {turn_index} of trajectory "
                    f"{session.traj_index} in group {session.group_id} "
                    f"breaks inside the previous turn's declared kept "
                    f"output: the agent used {turn.lcp - previous.input_len} "
                    f"of the {kept[turn_index - 1]} tokens it declared"
                )
            else:
                continues.append(False)
                rewritten += 1

        for turn_index in range(1, len(turns)):
            declared = turns[turn_index].declared_prev_kept_len
            if declared is not None and declared != kept[turn_index - 1]:
                raise ValueError(
                    f"turn {turn_index} of trajectory "
                    f"{session.traj_index} in group {session.group_id} "
                    f"declared prev_kept_len {declared} but close declared "
                    f"{kept[turn_index - 1]} for the previous turn"
                )
        # Count once the classification has committed: a close rejected by
        # the checks above is retried and re-runs the full pass.
        self._metrics["session_server/prefix_continuous"] += continuous
        self._metrics["session_server/prefix_rewritten"] += rewritten
        return continues

    async def _reward(self, sid: str, body: dict) -> JSONResponse:
        session, error = self._session(sid)
        if error is not None:
            return error
        group, error = self._live_group(session)
        if group is None:
            status = error.status_code
            return error if status >= 400 else JSONResponse(content={})

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
            if group_id in self._reported_group_ids:
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
        if not await self._put_payload(
            group_id, finalize_part, await_completion=True, weight=group.payload_bytes
        ):
            # _settle_put already recorded the failure, voided the group,
            # and removed any container the part produced.
            return False
        payload_channel = self._payload_channels[
            payload_shard_for(group_id, len(self._payload_channels))
        ]
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
                parts={"tokens": group.payload_bytes},
            ),
            payload_bytes=group.payload_bytes,
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
            await self._remove_payload_key(group_id)
            self._fail_serving(f"GroupMetadata put for group {group_id}", exc)
            return False
        group.reported = True
        self._remember_reported(group_id)
        for session in group.sessions.values():
            if session.session_id is not None:
                self._sessions.pop(session.session_id, None)
        self._groups.pop(group_id, None)
        return True

    async def _void_group(self, group_id: int, reason: str) -> None:
        """Mark a group voided, report it, and strip it to the bare record.

        The record — group identity, reason, timestamp — stays so late
        traffic for the group is answered (410 for write failures, drop for
        aborts) and a straggler abort cannot produce a second record; the
        token buffers go. The record itself expires after its TTL (see
        ``_expire_discard_records``). First voiding only.
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
        if reason == _WRITE_FAILED:
            self._metrics["session_server/payload_write_failed"] += 1
        await self._report_discard(group_id, reason)
        await self._put_payload(
            group_id, PayloadDiscard(group_id=group_id), await_completion=False
        )
        if reason == _WRITE_FAILED:
            # A part can land although its settlement failed or timed out;
            # with the finalize part, the collector will then have emitted a
            # container no one will fetch. remove_key chains after all puts
            # already issued for the key and drops the whole queue.
            await self._remove_payload_key(group_id)

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

    def _remember_reported(self, group_id: int) -> None:
        """Track recently reported groups, bounded.

        A straggler abort for one of them must not produce a second record,
        which would make the buffer worker fail the run, and the group state is
        gone by then; this bounded history is the only memory of it.
        """
        self._reported_group_ids.add(group_id)
        self._reported_group_order.append(group_id)
        if len(self._reported_group_order) > _REPORTED_HISTORY_LIMIT:
            self._reported_group_ids.discard(self._reported_group_order.popleft())

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
