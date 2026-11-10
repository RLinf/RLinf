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

"""The agent loop's HTTP transport for the session server path.

Replaces the generate channel pair for agents on the session server path:
each generate is one POST, and the group lifecycle (create, close, reward,
abort) is driven by the group runner. The training payload never passes
through this side — the session server writes it and reports the group
itself.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from rlinf.utils.http_client import HTTPStatusError, InferenceHTTPClient

# Status the session server answers for calls on a discarded group.
_DISCARDED_STATUS = 410


class SessionDiscardedError(RuntimeError):
    """The session server discarded the group this call belongs to.

    Raised from a 410 response. The agent loop's generate turns it into a
    ``GroupDiscarded`` (which lives in agent_loop.py and would make this
    import circular).
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class SessionBinding:
    """One trajectory's session, bound to its asyncio task by a ContextVar.

    The group runner creates one per trajectory after POST /sessions;
    ``generate`` reads it back and appends one call record per turn. The
    records hold what the close-time assertions and the assembly
    verification need: the ids sent, the engine's output ids, and the
    sampling params the caller asked for.
    """

    group_id: int
    traj_index: int
    session_id: str
    # One record per generate call, in turn order.
    turns: list[dict[str, Any]] = field(default_factory=list)

    def record_turn(
        self,
        input_ids: list[int],
        output_ids: list[int],
        sampling_params: Optional[dict],
    ) -> None:
        """Append this generate call's call record."""
        self.turns.append(
            {
                "input_ids": list(input_ids),
                "output_ids": list(output_ids),
                "sampling_params": sampling_params,
            }
        )


class SessionServerClient:
    """The session-server transport of one agent loop worker.

    One :class:`InferenceHTTPClient` per session server, each with its own
    pooled aiohttp session, so concurrent generates share connections
    without a lock. A group is pinned to one server by
    ``group_id % num_servers``: the server holds per-group state, so all of
    a group's calls must reach the same one.
    """

    def __init__(
        self,
        base_urls: list[str],
        generate_timeout_s: float,
        keepalive_timeout_s: float,
    ):
        assert base_urls, "the session server transport needs at least one URL"
        # keepalive_timeout must be shorter than the server's keep-alive
        # (session_server.keep_alive_s, computed by the runner as this
        # value plus 1) so we never reuse a connection the server is
        # about to close (an ECONNRESET race).
        self._clients = [
            InferenceHTTPClient(url, keepalive_timeout=keepalive_timeout_s)
            for url in base_urls
        ]
        self._generate_timeout_s = generate_timeout_s

    def _client_for(self, group_id: int) -> InferenceHTTPClient:
        return self._clients[group_id % len(self._clients)]

    async def create_session(
        self, group_id: int, traj_index: int, group_size: int, task_name: str
    ) -> str:
        """Open the session of one trajectory; returns its session id."""
        resp = await self._checked_post(
            group_id,
            "/sessions",
            {
                "group_id": group_id,
                "traj_index": traj_index,
                "group_size": group_size,
                "task_name": task_name,
            },
        )
        return resp["session_id"]

    async def generate(
        self,
        group_id: int,
        session_id: str,
        input_ids: list[int],
        sampling_params: Optional[dict] = None,
    ) -> dict:
        """Run one generate call; returns the server's response dict.

        The response carries ``logprobs`` only in verify mode, but the key
        is always present (None otherwise) — task code reads it
        unconditionally.
        """
        body: dict[str, Any] = {"input_ids": input_ids}
        if sampling_params is not None:
            body["sampling_params"] = sampling_params
        resp = await self._checked_post(
            group_id,
            f"/sessions/{session_id}/generate",
            body,
            timeout_s=self._generate_timeout_s,
        )
        resp.setdefault("logprobs", None)
        return resp

    async def close(
        self,
        group_id: int,
        session_id: str,
        kept_lens: dict[int, int],
        extra_fields_train: Optional[dict[str, list]] = None,
        verify_result: Optional[dict] = None,
    ) -> None:
        """Declare the trajectory's verdict; it produces no more tokens."""
        body: dict[str, Any] = {"kept_lens": kept_lens}
        if extra_fields_train:
            body["verdict"] = {"extra_fields_train": extra_fields_train}
        if verify_result is not None:
            body["verify_result"] = verify_result
        await self._checked_post(group_id, f"/sessions/{session_id}/close", body)

    async def reward(self, group_id: int, session_id: str, reward: float) -> bool:
        """Report the trajectory's score; returns whether the group is complete."""
        resp = await self._checked_post(
            group_id, f"/sessions/{session_id}/reward", {"reward": reward}
        )
        return bool(resp.get("group_complete", False))

    async def abort(self, group_id: int, reason: str) -> None:
        """Void the whole group; the session server reports the discard."""
        await self._checked_post(
            group_id, f"/groups/{group_id}/abort", {"reason": reason}
        )

    async def _checked_post(
        self,
        group_id: int,
        path: str,
        body: dict,
        timeout_s: Optional[float] = None,
    ) -> dict:
        """POST to the group's server and turn a 410 into a discard.

        Lifecycle calls can also hit a discarded group (a payload write
        failed after the last generate), so every route gets the same
        treatment; the discard record is already on its way to the buffer
        worker.
        """
        try:
            return await self._client_for(group_id).async_post(
                path, body, timeout_s=timeout_s
            )
        except HTTPStatusError as exc:
            if exc.status == _DISCARDED_STATUS:
                raise SessionDiscardedError(self._error_message(exc)) from exc
            raise

    @staticmethod
    def _error_message(exc: HTTPStatusError) -> str:
        """The server's ``error`` field out of a 410 body, best effort."""
        try:
            message = json.loads(exc.body).get("error")
        except (ValueError, AttributeError):
            message = None
        return str(message) if message else str(exc)
