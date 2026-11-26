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

"""Default-deny path whitelist keyed on the TCP peer address.

The RLinf control routes mounted by
:mod:`rlinf.hybrid_engines.sglang.common.http_routes` deserialize pickled
request bodies, so while the sglang server binds only loopback the trust
boundary is "same machine". The multi-node rollout data plane needs the
router on another node to reach the server, so the server then binds a
network address — and this middleware keeps the trust boundary from
widening with it: requests whose TCP peer is a loopback address pass,
requests from any other address must hit a whitelisted path exactly or get
a 403
before any handler runs (in particular, before a pickled body is
deserialized).

Standard-library only: installed alongside the control routes inside the
sglang server subprocess, and unit-tested without FastAPI.
"""

from collections.abc import Collection
from typing import Any, Callable, MutableMapping

# Paths a non-local peer may request. Exact matches only: default-deny
# resists path obfuscation ("/rlinf%2F...", doubled slashes, case changes).
# The four entries are everything an sglang router calls on a worker under
# `policy: manual` with `disable_health_check: true`: the registration
# probes (GET /server_info, GET /model_info, GET /health, once per worker
# at registration) and the forwarded POST /generate. Any other router
# configuration reaches further endpoints and is rejected here: PowerOfTwo
# policies poll GET /v1/loads every 30 s, and enabled health checks issue
# periodic /health traffic. Config validation pins these two settings
# whenever the session server is enabled.
WHITELISTED_PATHS = frozenset({"/generate", "/health", "/server_info", "/model_info"})
EMBODIED_WHITELISTED_PATHS = frozenset(
    {
        "/v1/actions/generations",
        "/rollout/generate",
        "/health",
        "/server_info",
        "/model_info",
    }
)

# Loopback values of ``scope["client"][0]``.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})

_FORBIDDEN_BODY = (
    "403 Forbidden: non-local requests are only allowed on the "
    "router-facing paths (/generate, /health, /server_info, /model_info)."
).encode()


class PeerWhitelistMiddleware:
    """ASGI middleware enforcing router-facing path allowlists for remote peers.

    HTTP requests whose TCP peer is loopback pass through untouched. For
    any other peer (including a scope with no client address — treat
    unverifiable as untrusted) only the allowed paths pass; everything
    else is answered with 403 before the app, so no handler and no request
    body deserialization runs. Non-HTTP scopes (lifespan, websockets) pass
    through.
    """

    def __init__(self, app: Any, allowed_paths: Collection[str] | None = None) -> None:
        self.app = app
        self._allowed_paths = (
            frozenset(allowed_paths) if allowed_paths is not None else WHITELISTED_PATHS
        )
        if allowed_paths is None:
            self._forbidden_body = _FORBIDDEN_BODY
        else:
            paths = ", ".join(sorted(self._allowed_paths))
            self._forbidden_body = (
                "403 Forbidden: non-local requests are only allowed on the "
                f"router-facing paths ({paths})."
            ).encode()

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Any],
        send: Callable[[MutableMapping[str, Any]], Any],
    ) -> None:
        if scope["type"] == "http":
            client = scope.get("client")
            peer = client[0] if client else None
            if (
                peer not in _LOOPBACK_HOSTS
                and scope.get("path", "") not in self._allowed_paths
            ):
                await self._send_forbidden(send, self._forbidden_body)
                return
        await self.app(scope, receive, send)

    @staticmethod
    async def _send_forbidden(
        send: Callable[[MutableMapping[str, Any]], Any],
        body: bytes = _FORBIDDEN_BODY,
    ) -> None:
        """Answer the request with a plain-text 403."""
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
