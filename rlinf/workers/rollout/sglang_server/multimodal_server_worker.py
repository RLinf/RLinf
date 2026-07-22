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

"""Multimodal sglang server worker.

Subclass of :class:`SGLangServerWorker`. The base launches a standard SRT
server via ``ServerArgs(**)`` + ``sglang.srt...launch_server`` (an in-process
Python API) — that path rejects multimodal/diffusion fields (``pipeline``,
``pipeline_config_path``, ``dit_cpu_offload``, ``cfg_parallel_size``,
``sp_size``) because they are not ``ServerArgs`` dataclass fields; they only
exist on the multimodal ``sglang serve`` CLI (argparse).

This subclass overrides only the launch-mechanism-specific surface
(:meth:`init_server` / :meth:`is_healthy` / :meth:`shutdown`) to spawn the
*multimodal* ``sglang serve`` CLI command instead (the same command the
embodied eval always used), while inheriting ``__init__``,
:meth:`get_server_url`, port acquisition, and the health-poll helper from
the base. :func:`launch_sglang_router_and_server` picks this class (vs the
base) when ``router_server_args.multimodal`` is true.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from typing import Any

import ray.util
import requests

from .server_worker import SGLangServerWorker, _wait_for_http_health


class SGLangMultimodalServerWorker(SGLangServerWorker):
    """Worker that owns one multimodal ``sglang serve`` subprocess."""

    # ------------------------------------------------------------------
    # Lifecycle (overrides — spawn the multimodal `sglang serve` CLI)
    # ------------------------------------------------------------------
    def init_server(self) -> None:
        """Spawn the multimodal ``sglang serve`` subprocess and wait /health."""
        assert getattr(self, "_serve_proc", None) is None, (
            "multimodal sglang server already initialized."
        )

        # Now the router is not supported for the multimodal sglang server.
        if self._cfg.rollout.sglang.get("launch_router", False):
            raise RuntimeError(
                "launch_router is not supported for the multimodal sglang "
                "server: the sglang router only forwards fixed endpoints "
                "(/generate, /v1/chat/completions, ...), not the dreamzero "
                "action endpoint /v1/actions/generations. Set "
                "rollout.sglang.launch_router: false (rollout workers hit "
                "their rank-assigned server URL directly)."
            )

        # Two free ports: HTTP listener + torch.distributed bootstrap.
        http_port = self.acquire_free_port()
        master_port = self.acquire_free_port()

        # Localhost calls must never tunnel through an upstream proxy.
        _local_hosts = "127.0.0.1,localhost,::1"
        _no_proxy = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
        if not any(h in _no_proxy for h in ("127.0.0.1", "localhost")):
            os.environ["NO_PROXY"] = (
                f"{_no_proxy},{_local_hosts}".strip(",") if _no_proxy else _local_hosts
            )

        # tmpdir must exist before _build_serve_command: the DreamZero
        # model-specific args write the pipeline-config override into it.
        self._obs_tmpdir = tempfile.mkdtemp(prefix="sglang_mm_serve_")
        self._serve_log_path = os.path.join(
            self._obs_tmpdir, f"sglang_mm_serve_rank{self._rank}.log"
        )
        self._serve_log_fh = open(self._serve_log_path, "ab")

        cmd = self._build_serve_command(http_port=http_port, master_port=master_port)

        env = os.environ.copy()
        env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        self.log_info(f"multimodal sglang serve: launching: {' '.join(cmd)}")
        self.log_info(f"multimodal sglang serve: log -> {self._serve_log_path}")

        self._serve_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._serve_log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group so SIGTERM via killpg(pid) reaches the serve tree, not the actor
        )
        # Set the base attrs get_server_url() reads.
        self._server_port = http_port
        if self._advertise_host is None:
            self._advertise_host = ray.util.get_node_ip_address()

        spawn_timeout = float(self._cfg.rollout.sglang.get("spawn_timeout", 1800.0))
        try:
            # Poll localhost for /health: the serve subprocess is local to
            # this actor, and an upstream HTTP(S)_PROXY in the launch env
            # would otherwise tunnel the node-IP poll through a proxy that
            # cannot reach the internal node (NO_PROXY only covers
            # 127.0.0.1/localhost). get_server_url() still advertises the
            # node IP for cross-node clients (the action client uses a
            # no-proxy opener, so it reaches the node IP directly).
            _wait_for_http_health("127.0.0.1", http_port, timeout=spawn_timeout)
        except RuntimeError as e:
            self.log_error(f"multimodal sglang server failed to become healthy: {e!r}")
            self.shutdown()
            raise
        self.log_info(f"multimodal sglang server ready at {self.get_server_url()}")

    def is_healthy(self) -> bool:
        if (
            getattr(self, "_serve_proc", None) is None
            or self._serve_proc.poll() is not None
        ):
            return False
        try:
            url = f"http://127.0.0.1:{self._server_port}/health"
            return (
                requests.get(
                    url, timeout=2, proxies={"http": None, "https": None}
                ).status_code
                == 200
            )
        except requests.exceptions.RequestException:
            return False

    def shutdown(self) -> None:
        """Terminate the serve subprocess (and its process group)."""
        proc = getattr(self, "_serve_proc", None)
        if proc is None:
            return
        self.log_info(f"Shutting down multimodal sglang server pid={proc.pid}.")
        # start_new_session=True (Popen) makes proc.pid == its pgid, so
        # killpg(pid) hits the serve tree, not the Ray actor's group.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        if proc.poll() is None:  # pragma: no cover — best-effort kill
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if getattr(self, "_serve_log_fh", None) is not None:
            try:
                self._serve_log_fh.close()
            except Exception:
                pass
            self._serve_log_fh = None
        self._serve_proc = None
        self._server_port = None

    def _get(self, key: str, default: Any = None) -> Any:
        cfg = self._sglang_cfg
        return cfg.get(key, default) if cfg is not None else default

    def _eval_batch_size(self) -> int:
        try:
            eval_env_cfg = self._cfg.env.get("eval", None)
            total = int(eval_env_cfg.total_num_envs) if eval_env_cfg else 0
        except Exception:
            total = 0
        try:
            stages = int(self._cfg.rollout.pipeline_stage_num)
        except Exception:
            stages = 1
        return total // stages if stages else total

    def _build_serve_command(self, *, http_port: int, master_port: int) -> list[str]:
        """Build the ``sglang serve`` CLI from the ``server`` block fields.

        Explicit allow-list: only known serve flags are emitted (renamed where
        the config key differs from the CLI flag). cfg_scale/num_inference_steps/
        compile_components are NOT here — they go into the DreamZero
        pipeline-config JSON (see :meth:`_model_specific_serve_args`).
        """
        sglang_cfg = self._sglang_cfg
        model_cfg = self._cfg.rollout.model

        # Resolve checkpoint path: server.model_path, else rollout.model_path,
        # else rollout.model.model_path.
        rollout_model_path = self._cfg.rollout.get("model_path", None)
        model_path = (
            self._get("model_path", None) or rollout_model_path or model_cfg.model_path
        )

        sglang_bin = str(self._cfg.rollout.sglang.get("sglang_bin", "sglang"))
        if not shutil.which(sglang_bin):
            sglang_bin = "sglang"  # fall back to `python -m sglang` if console absent

        cmd = [
            sglang_bin,
            "serve",
            "--model-path",
            model_path,
            "--host",
            self._bind_host,
            "--port",
            str(http_port),
            "--master-port",
            str(master_port),
        ]
        if v := self._get("backend", None):
            cmd += ["--backend", str(v)]
        if v := self._get("pipeline", None):
            cmd += ["--pipeline", str(v)]
        if v := self._get("pipeline_config_path", None):
            cmd += ["--pipeline-config-path", str(v)]
        if (v := int(self._get("num_gpus", 0) or 0)) > 0:
            cmd += ["--num-gpus", str(v)]
        if (v := int(self._get("tp_size", 0) or 0)) > 0:
            cmd += ["--tp-size", str(v)]
        if (v := int(self._get("ulysses_degree", 0) or 0)) > 0:
            cmd += ["--ulysses-degree", str(v)]
        if (v := int(self._get("ring_degree", 0) or 0)) > 0:
            cmd += ["--ring-degree", str(v)]
        if (v := int(self._get("data_parallel_size", 0) or 0)) > 0:
            cmd += ["--data-parallel-size", str(v)]
        if isinstance(self._get("enable_cfg_parallel", None), bool):
            cmd += [
                "--enable-cfg-parallel",
                "true" if self._get("enable_cfg_parallel") else "false",
            ]
        if (v := int(self._get("cfg_parallel_size", 0) or 0)) > 0:
            cmd += ["--cfg-parallel-size", str(v)]
        if v := self._get("attention_backend", None):
            cmd += ["--attention-backend", str(v)]
        if (v := self._get("scheduler_port", None)) is not None:
            cmd += ["--scheduler-port", str(int(v))]
        if isinstance(self._get("dit_cpu_offload", None), bool):
            cmd += [
                "--dit-cpu-offload",
                "true" if self._get("dit_cpu_offload") else "false",
            ]
        if v := self._get("performance_mode", None):
            cmd += ["--performance-mode", str(v)]
        # Model-specific flags (DreamZero pipeline selection + config file).
        cmd += self._model_specific_serve_args(
            sglang_cfg=sglang_cfg, model_cfg=model_cfg
        )
        return cmd

    def _model_specific_serve_args(
        self,
        sglang_cfg: Any,
        model_cfg: Any,
    ) -> list[str]:
        """Extra ``sglang serve`` flags required by the embodied model type.

        For DreamZero: default backend/pipeline/cfg-parallel/sp-degree when the
        block doesn't set them, and write the DreamZeroPipeline override config
        (cfg_scale/num_inference_steps/compile_components go into this JSON,
        not the CLI).
        """
        args: list[str] = []
        model_type = str(getattr(model_cfg, "model_type", "") or "").lower()
        if model_type == "dreamzero":
            from rlinf.workers.rollout.sglang.action_policies.dreamzero import (
                DreamZeroActionPolicy,
            )

            if self._get("backend", None) is None:
                args += ["--backend", "sglang"]
            if self._get("pipeline", None) is None:
                args += ["--pipeline", "DreamZeroPipeline"]
            if self._get("pipeline_config_path", None) is None:
                pipeline_config_path = DreamZeroActionPolicy._write_pipeline_config(
                    sglang_cfg=sglang_cfg,
                    model_cfg=model_cfg,
                    tmpdir=self._obs_tmpdir,
                    rank=self._rank,
                    eval_batch_size=self._eval_batch_size(),
                )
                args += ["--pipeline-config-path", pipeline_config_path]
            if self._get("cfg_parallel_size", None) is None:
                args += ["--cfg-parallel-size", "1"]
            if (v := int(self._get("sp_degree", 1) or 1)) > 0:
                args += ["--sp-degree", str(v)]
        return args
