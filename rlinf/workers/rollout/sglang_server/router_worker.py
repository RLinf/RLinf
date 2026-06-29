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


from __future__ import annotations

import dataclasses
import os
import shlex
import signal
import subprocess
import sys
import time
from typing import Optional

import ray.util
import requests
from omegaconf import DictConfig, OmegaConf
from sglang_router.router_args import RouterArgs

from rlinf.scheduler import Worker
from rlinf.utils import internal_http

# RouterArgs fields the launch_router CLI does NOT accept and that we
# therefore must not forward as flags.
_ROUTER_ARGS_NOT_ON_CLI = {
    # These are derived inside RouterArgs.from_cli_args() from --prefill /
    # --decode / --selector flags, not exposed as direct flags.
    "prefill_urls",
    "decode_urls",
    "control_plane_api_keys",
}

# RouterArgs flag names that map to a CLI name other than
# `--<field>.replace('_', '-')`.
_FIELD_NAME_TO_FLAG = {
    "server_cert_path": "tls-cert-path",
    "server_key_path": "tls-key-path",
}

_VALID_ROUTER_FIELDS = {f.name for f in dataclasses.fields(RouterArgs)}


def _flag_for_field(field_name: str) -> str:
    """Map a RouterArgs field name to the corresponding CLI flag."""
    cli_name = _FIELD_NAME_TO_FLAG.get(field_name, field_name.replace("_", "-"))
    return f"--{cli_name}"


def _router_cfg_to_cli(router_cfg: dict) -> list[str]:
    """Turn a ``{field_name: value}`` dict into ``launch_router`` CLI args.

    The dict's keys MUST be valid :class:`RouterArgs` field names. Bools
    map to flag-only args; lists/tuples become ``--flag a b c``; ``None``
    is dropped. Unknown keys raise.
    """
    args: list[str] = []
    for key, value in router_cfg.items():
        if key not in _VALID_ROUTER_FIELDS:
            raise ValueError(
                f"Unknown router config key {key!r}. Expected one of "
                f"{sorted(_VALID_ROUTER_FIELDS)}."
            )
        if key in _ROUTER_ARGS_NOT_ON_CLI:
            raise ValueError(
                f"Router config key {key!r} cannot be set via the launch_router "
                f"CLI; configure it through a dedicated method (e.g. "
                f"register_server) or by extending this helper."
            )
        if value is None:
            continue
        flag = _flag_for_field(key)
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, (list, tuple)):
            args.append(flag)
            args.extend(str(v) for v in value)
        else:
            args.extend([flag, str(value)])
    return args


class SGLangRouterWorker(Worker):
    """Worker that owns a single ``sglang_router.launch_router`` subprocess.

    The router boots with no workers attached. Use :meth:`register_server`
    to add backends dynamically, and :meth:`unregister_server` to remove
    them.

    Args:
        config: Full RLinf ``DictConfig``. Kept for parity with other
            workers / future use; the router itself is configured entirely
            from ``router_cfg``.
        router_cfg: The sub-config block whose keys are forwarded to
            ``launch_router`` as ``--<field>`` CLI flags (underscores
            replaced by dashes); keys MUST match :class:`RouterArgs` field
            names. Typically ``config.rollout.router`` when used inside a
            rollout, but any compatible block works (the router isn't tied
            to the rollout pipeline). The ``host`` and ``port`` keys
            control the router's bind address; if ``port`` is missing or
            ``None``, a free port is grabbed via the worker's PortLock at
            :meth:`init_router` time. ``host`` defaults to ``0.0.0.0``.
    """

    def __init__(
        self,
        config: DictConfig,
        router_cfg: DictConfig,
    ):
        Worker.__init__(self)
        self._cfg = config
        self._router_cfg = router_cfg

        self._proc: Optional[subprocess.Popen] = None
        self._router_url: Optional[str] = None
        self._advertise_host: Optional[str] = None
        self._port: Optional[int] = None
        # Tracks server URL -> worker_id (uuid) so we can DELETE by id later.
        self._registered: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def init_router(self) -> None:
        """Spawn the router subprocess with NO workers attached.

        Attach workers afterwards with :meth:`register_server`. On failure
        the subprocess is torn down via ``shutdown`` before ``RuntimeError``
        is re-raised, so the caller can retry or fail fast without leaking
        a zombie router process.

        Raises:
            RuntimeError: if the router subprocess exits before becoming
                healthy or fails the ``/health`` probe within the timeout.
        """
        assert self._proc is None, "router subprocess already started."

        router_cfg = OmegaConf.to_container(self._router_cfg, resolve=True)
        self._port = router_cfg.pop("port", self.acquire_free_port())

        cmd = [
            sys.executable,
            "-m",
            "sglang_router.launch_router",
            "--port",
            str(self._port),
            *_router_cfg_to_cli(router_cfg),
        ]

        self.log_info(
            "Launching sglang router subprocess: "
            + " ".join(shlex.quote(c) for c in cmd)
        )

        # start_new_session=True makes the child a session leader, so a
        # SIGTERM to it doesn't propagate to the Ray actor (and vice
        # versa). stdout/stderr inherit the actor's so router logs land
        # in the same place as the worker's own logs.
        self._proc = subprocess.Popen(
            cmd,
            env=internal_http.direct_request_env(),
            start_new_session=True,
        )

        if self._advertise_host is None:
            self._advertise_host = ray.util.get_node_ip_address()
        self._router_url = f"http://{self._advertise_host}:{self._port}"

        try:
            self._wait_for_router_health(self._port)
        except RuntimeError as e:
            self.log_error(f"sglang router failed to become healthy: {e!r}")
            self.shutdown()
            raise
        self.log_info(f"sglang router ready at {self._router_url}")

    def _wait_for_router_health(self, port: int, timeout: float = 300.0) -> None:
        """Block until ``GET /health`` on the router returns 200."""
        deadline = time.perf_counter() + timeout
        url = f"http://{self._advertise_host}:{port}/health"
        last_err: Optional[BaseException] = None
        while time.perf_counter() < deadline:
            assert self._proc is not None
            ret = self._proc.poll()
            if ret is not None:
                raise RuntimeError(
                    f"sglang router subprocess exited with code {ret} before "
                    f"becoming healthy."
                )
            try:
                if internal_http.get(url, timeout=5).status_code == 200:
                    return
            except requests.exceptions.RequestException as e:
                last_err = e
            time.sleep(0.5)
        raise RuntimeError(
            f"router at {url} not healthy within {timeout:.0f}s "
            f"(last error: {last_err!r})."
        )

    def get_router_url(self) -> str:
        """Return the advertised ``http://host:port`` URL of the router."""
        assert self._router_url is not None, "init_router() has not been called."
        return self._router_url

    def is_healthy(self) -> bool:
        if (
            self._router_url is None
            or self._proc is None
            or self._proc.poll() is not None
        ):
            return False
        try:
            return (
                internal_http.get(f"{self._router_url}/health", timeout=2).status_code
                == 200
            )
        except requests.exceptions.RequestException:
            return False

    # ------------------------------------------------------------------
    # Dynamic worker registration
    # ------------------------------------------------------------------
    def register_server(
        self,
        server_url: str,
        worker_type: str = "regular",
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> dict:
        """Attach an sglang server to the running router and wait until ready.

        ``POST /workers`` returns 202 immediately and the router runs the
        registration workflow asynchronously (probe metadata, activate,
        load tokenizer, …). This method blocks until ``GET /workers/<id>``
        reports ``is_healthy=true``, so a follow-up ``/generate`` won't
        race against an unactivated worker (which would return 503).

        Equivalent to::

            curl -X POST http://<router>/workers \\
                 -H "Content-Type: application/json" \\
                 -d '{"url":"<server_url>","worker_type":"<worker_type>"}'

        Args:
            server_url: ``http://host:port`` of the sglang server to add.
            worker_type: ``"regular"`` (default), ``"prefill"`` or
                ``"decode"`` for PD-disaggregated setups.
            timeout: Total seconds to wait for the worker to reach
                ``is_healthy=true``. Failed registrations raise sooner.
            poll_interval: Seconds between ``GET /workers/<id>`` polls.

        Returns:
            The final worker-status dict from ``GET /workers/<id>``.
        """
        assert self._router_url is not None, "init_router() has not been called."
        self.log_info(f"Registering worker {server_url!r} ({worker_type}) on router.")
        resp = internal_http.post(
            f"{self._router_url}/workers",
            json={"url": server_url, "worker_type": worker_type},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        worker_id = data.get("worker_id")
        if not worker_id:
            raise RuntimeError(
                f"router did not return a worker_id for {server_url!r}: {data!r}"
            )
        self._registered[server_url] = worker_id

        deadline = time.perf_counter() + timeout
        status: dict = {}
        while time.perf_counter() < deadline:
            time.sleep(poll_interval)
            r = internal_http.get(f"{self._router_url}/workers/{worker_id}", timeout=10)
            if r.status_code == 404:
                continue  # router hasn't persisted the row yet — keep polling
            r.raise_for_status()
            status = r.json()
            if status.get("is_healthy"):
                self.log_info(
                    f"Worker {server_url!r} active (id={worker_id}, "
                    f"model_id={status.get('model_id')!r})."
                )
                return status
            job = status.get("job_status") or {}
            if job.get("status") == "failed":
                raise RuntimeError(f"Router refused worker {server_url!r}: {job!r}")
        raise RuntimeError(
            f"Worker {server_url!r} (id={worker_id}) did not become healthy "
            f"within {timeout:.0f}s; last status={status!r}."
        )

    def unregister_server(self, server_url: str, timeout: float = 60.0) -> dict:
        """Remove a previously-registered server from the router.

        Issues ``DELETE /workers/<id>`` using the worker_id captured at
        register time. Quietly returns ``{}`` if the URL isn't tracked
        (already removed / never registered through this worker).
        """
        assert self._router_url is not None, "init_router() has not been called."
        worker_id = self._registered.pop(server_url, None)
        if worker_id is None:
            return {}
        self.log_info(
            f"Unregistering worker {server_url!r} (id={worker_id}) from router."
        )
        resp = internal_http.delete(
            f"{self._router_url}/workers/{worker_id}", timeout=timeout
        )
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # ------------------------------------------------------------------
    # Client-side helpers — call the router over HTTP.
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: Optional[str] = None,
        input_ids: Optional[list] = None,
        sampling_params: Optional[dict] = None,
        timeout: float = 600.0,
    ) -> dict:
        """POST a single request to the router's ``/generate`` endpoint."""
        assert self._router_url is not None, "init_router() has not been called."
        body: dict = {}
        if prompt is not None:
            body["text"] = prompt
        if input_ids is not None:
            body["input_ids"] = input_ids
        if sampling_params is not None:
            body["sampling_params"] = sampling_params
        resp = internal_http.post(
            f"{self._router_url}/generate", json=body, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()

    def shutdown(self) -> None:
        """SIGTERM the router subprocess (and its process group)."""
        proc = self._proc
        if proc is None:
            return
        self.log_info(f"Shutting down sglang router pid={proc.pid}.")
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                proc.terminate()
            except Exception:  # pragma: no cover — best-effort
                pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait(timeout=5)
        self._proc = None
        self._router_url = None
