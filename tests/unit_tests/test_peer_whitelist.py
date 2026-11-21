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

"""Tests for the peer-address whitelist of sglang servers.

Two parts:

- The middleware itself, driven directly with hand-built scopes (a plain
  inner app counts whether the handler ran), so no server, FastAPI, or
  sglang is needed — this file must run in the embodied-only CI
  environment too.
- The URL split the whitelist forces on ``SGLangServerProcess``: the
  control-plane base URL stays loopback even when the server advertises
  its node IP, and the advertise URL keeps pointing at the node IP for
  router registration.
"""

import asyncio
from typing import Any, MutableMapping

import pytest

from rlinf.hybrid_engines.sglang.common.peer_whitelist import PeerWhitelistMiddleware

# The three pickled control routes plus an arbitrary other endpoint.
LOCAL_ONLY_PATHS = [
    "/rlinf/run_task_method",
    "/rlinf/sync_hf_weight",
    "/update_weight_version",
    "/pause_generation",
]

# Variants of whitelisted or control paths that must NOT match exactly.
PATH_DEFORMATIONS = [
    "/rlinf%2Frun_task_method",  # undecoded percent-encoding
    "/rlinf/run_task_method",  # decoded form
    "//generate",  # doubled leading slash
    "/generate/",  # trailing slash
    "/GENERATE",  # case change
    "/Generate",
    "/gen%65rate",  # percent-encoded letter
    "/generate/..",
    "/health/",
    "/server_info/",
    "/model_info?",
]


def _drive(path: str, client: Any) -> tuple[bool, int]:
    """Run one request through the middleware over a counting inner app.

    Args:
        path: The request path as the server would deliver it in
            ``scope["path"]``.
        client: The ``scope["client"]`` tuple, or ``None`` for a scope
            without a client address.

    Returns:
        ``(handler_ran, status)`` — whether the inner app was reached and
        the HTTP status it (or the middleware) produced.
    """
    handler_ran = False

    async def app(scope, receive, send):
        nonlocal handler_ran
        handler_ran = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    messages: list[MutableMapping[str, Any]] = []

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": client,
        "server": ("10.0.0.1", 80),
    }
    middleware = PeerWhitelistMiddleware(app)
    asyncio.run(middleware(scope, receive, send))
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    return handler_ran, status


class TestPeerWhitelistMiddleware:
    def test_local_peer_reaches_everything(self):
        for host in ("127.0.0.1", "::1"):
            for path in ["/generate", "/health", *LOCAL_ONLY_PATHS]:
                handler_ran, status = _drive(path, (host, 40000))
                assert handler_ran, f"{host} {path}: handler must run"
                assert status == 200, f"{host} {path}: got {status}"

    def test_remote_peer_passes_only_the_whitelisted_paths(self):
        peer = ("10.208.10.188", 40000)
        for path in ("/generate", "/health", "/server_info", "/model_info"):
            handler_ran, status = _drive(path, peer)
            assert handler_ran, f"{path}: handler must run"
            assert status == 200, f"{path}: got {status}"
        for path in LOCAL_ONLY_PATHS:
            handler_ran, status = _drive(path, peer)
            assert not handler_ran, f"{path}: handler must not run"
            assert status == 403, f"{path}: got {status}"

    def test_remote_peer_path_deformations_stay_denied(self):
        peer = ("10.208.10.188", 40000)
        for path in PATH_DEFORMATIONS:
            handler_ran, status = _drive(path, peer)
            assert not handler_ran, f"{path}: handler must not run"
            assert status == 403, f"{path}: got {status}"

    def test_scope_without_client_address_is_treated_as_remote(self):
        # An address we cannot verify gets the untrusted treatment: the
        # whitelist applies even though the peer is unknown.
        handler_ran, status = _drive("/rlinf/run_task_method", None)
        assert not handler_ran
        assert status == 403
        handler_ran, status = _drive("/generate", None)
        assert handler_ran
        assert status == 200

    def test_non_http_scopes_pass_through(self):
        handler_ran = False

        async def app(scope, receive, send):
            nonlocal handler_ran
            handler_ran = True

        async def receive():
            raise AssertionError("lifespan does not read the request body")

        async def send(message):
            raise AssertionError("middleware must not answer lifespan itself")

        scope = {"type": "lifespan", "asgi": {"version": "3.0"}}
        asyncio.run(PeerWhitelistMiddleware(app)(scope, receive, send))
        assert handler_ran


class TestControlPlaneUrl:
    def _unstarted_process(self, bind_host, advertise_host):
        from rlinf.workers.rollout.sglang.process import SGLangServerProcess

        process = SGLangServerProcess(
            server_args_kwargs={},
            acquire_free_port=lambda *_: 0,
            bind_host=bind_host,
            advertise_host=advertise_host,
        )
        # The URL accessors only read the port; set it directly so the
        # test does not spawn a real sglang server in start().
        process._server_port = 31234
        return process

    def test_control_url_stays_loopback_when_advertising_node_ip(self):
        # session_server.enable path: bind 0.0.0.0, advertise the node IP.
        # A worker connecting to its own node IP presents that IP as the
        # TCP peer, which the whitelist rejects on the /rlinf/* control
        # routes, so the control-plane client must stay on loopback.
        process = self._unstarted_process("0.0.0.0", "10.208.10.188")
        assert process.get_control_url() == "http://127.0.0.1:31234"
        assert process.get_server_url() == "http://10.208.10.188:31234"

    def test_control_url_is_loopback_without_session_server(self):
        # Legacy path: both addresses are loopback, so control-plane
        # clients behave exactly as before the whitelist existed.
        process = self._unstarted_process("127.0.0.1", "127.0.0.1")
        assert process.get_control_url() == "http://127.0.0.1:31234"
        assert process.get_server_url() == "http://127.0.0.1:31234"


if __name__ == "__main__":
    pytest.main([__file__])
