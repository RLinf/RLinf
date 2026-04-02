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

"""Websocket client backend for the official LingBot-VA server."""

from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.utils.logging import get_logger

logger = get_logger()


class LingbotVANativeBackend:
    """Thin wrapper around the official stateful single-session websocket server."""

    def __init__(self, cfg: Any, torch_dtype: torch.dtype) -> None:
        self.cfg = cfg
        self.torch_dtype = torch_dtype
        self.repo_path = Path(getattr(cfg.lingbotva, "repo_path"))
        self.model_path = Path(cfg.model_path)
        self.save_root = str(
            getattr(cfg.lingbotva, "save_root", "/tmp/lingbotva_rlinf")
        )
        self._validate_attn_mode(self.model_path)
        self._server_process: subprocess.Popen[str] | None = None
        self._server_log_path: str | None = None
        self._owns_server_process = False
        self._client = self._build_client()
        atexit.register(self.close)

    @staticmethod
    def _validate_attn_mode(model_path: Path) -> None:
        config_path = model_path / "transformer" / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"LingBot-VA transformer config not found at {config_path}."
            )
        config_data = json.loads(config_path.read_text())
        attn_mode = config_data.get("attn_mode")
        if attn_mode not in {"torch", "flashattn"}:
            raise ValueError(
                "LingBot-VA inference requires transformer/config.json attn_mode "
                f'to be "torch" or "flashattn", but got {attn_mode!r}.'
            )

    @staticmethod
    def _dtype_to_arg(torch_dtype: torch.dtype) -> str:
        if torch_dtype == torch.bfloat16:
            return "bf16"
        if torch_dtype == torch.float16:
            return "fp16"
        if torch_dtype == torch.float32:
            return "fp32"
        raise ValueError(
            f"Unsupported torch dtype for LingBot-VA backend: {torch_dtype!r}."
        )

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    def _import_client_cls(self):
        sys.path.insert(0, str(self.repo_path))
        sys.path.insert(0, str(self.repo_path / "wan_va"))
        from utils.Simple_Remote_Infer.deploy.websocket_client_policy import (
            WebsocketClientPolicy,
        )

        return WebsocketClientPolicy

    @staticmethod
    def _can_connect(host: str, port: int, timeout: float = 1.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _launch_server_process(self, host: str, port: int) -> None:
        launcher_path = Path(__file__).with_name("server_launcher.py")
        if not launcher_path.exists():
            repo_path = os.environ.get("REPO_PATH")
            if repo_path:
                candidate = (
                    Path(repo_path)
                    / "rlinf"
                    / "models"
                    / "embodiment"
                    / "lingbotva"
                    / "server_launcher.py"
                )
                if candidate.exists():
                    launcher_path = candidate
        if not launcher_path.exists():
            raise FileNotFoundError(
                f"LingBot-VA server launcher not found near {__file__}."
            )
        master_port = self._find_free_port()
        save_root = Path(self.save_root)
        save_root.mkdir(parents=True, exist_ok=True)
        self._server_log_path = str(save_root / f"lingbotva_server_{port}.log")
        env = os.environ.copy()
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(master_port)
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["LOCAL_RANK"] = "0"
        server_cuda_visible_devices = getattr(
            self.cfg.lingbotva, "server_cuda_visible_devices", None
        )
        if server_cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(server_cuda_visible_devices)
        server_threads = str(
            getattr(self.cfg.lingbotva, "server_torch_num_threads", 32)
        )
        env["LINGBOTVA_TORCH_NUM_THREADS"] = server_threads
        env["OMP_NUM_THREADS"] = server_threads
        env["MKL_NUM_THREADS"] = server_threads
        env["OPENBLAS_NUM_THREADS"] = server_threads
        env["NUMEXPR_NUM_THREADS"] = server_threads
        pythonpath_parts = [str(self.repo_path), str(self.repo_path / "wan_va")]
        existing = env.get("PYTHONPATH", "")
        if existing:
            pythonpath_parts.append(existing)
        env["PYTHONPATH"] = ":".join(pythonpath_parts)
        cmd = [
            sys.executable,
            str(launcher_path),
            "--repo-path",
            str(self.repo_path),
            "--config-name",
            "robotwin",
            "--model-path",
            str(self.model_path),
            "--save-root",
            self.save_root,
            "--host",
            host,
            "--port",
            str(port),
            "--enable-offload",
            "true"
            if bool(getattr(self.cfg.lingbotva, "enable_offload", True))
            else "false",
            "--param-dtype",
            self._dtype_to_arg(self.torch_dtype),
        ]
        log_file = open(self._server_log_path, "w", encoding="utf-8")
        self._server_process = subprocess.Popen(
            cmd,
            cwd=str(self.repo_path),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._owns_server_process = True
        logger.info(
            "Launched LingBot-VA official server subprocess on ws://%s:%s", host, port
        )

    def _build_client(self):
        # LingBot-VA official websocket inference keeps prompt/cache state inside a
        # single server session, so RLinf intentionally reuses one client/session
        # per rollout worker and guards batch size at higher levels.
        host = "127.0.0.1"
        reuse_server = bool(getattr(self.cfg.lingbotva, "reuse_server", False))
        fixed_port = getattr(self.cfg.lingbotva, "fixed_server_port", None)
        port = int(fixed_port) if fixed_port is not None else self._find_free_port()
        client_cls = self._import_client_cls()

        if reuse_server and self._can_connect(host, port):
            logger.info("Reusing existing LingBot-VA server at ws://%s:%s", host, port)
            return client_cls(host=host, port=port)

        self._launch_server_process(host, port)
        return client_cls(host=host, port=port)

    def reset(self, prompt: str) -> None:
        self._client.infer({"reset": True, "prompt": prompt})

    def infer(self, obs: dict[str, Any], prompt: str) -> np.ndarray:
        result = self._client.infer({"obs": obs, "prompt": prompt})
        action = result.get("action")
        if action is None:
            raise RuntimeError("LingBot-VA server infer returned no action tensor.")
        return np.asarray(action)

    def compute_kv_cache(self, obs: list[dict[str, Any]], state: np.ndarray) -> None:
        self._client.infer({"obs": obs, "compute_kv_cache": True, "state": state})

    def close(self) -> None:
        if self._server_process is None or not self._owns_server_process:
            return
        if self._server_process.poll() is None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
                self._server_process.wait(timeout=10)
        self._server_process = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
