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

import inspect
import logging
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _to_plain(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


class SGLangRewardServer:
    """Lifecycle manager for a runner-owned SGLang reward HTTP server."""

    def __init__(self, reward_model_cfg: DictConfig):
        self.reward_model_cfg = reward_model_cfg
        self.host = str(reward_model_cfg.get("server_host", "127.0.0.1"))
        self.port = int(reward_model_cfg.get("server_port", 30000))
        self.api_base = f"http://{self.host}:{self.port}/v1"
        self.startup_timeout = float(
            reward_model_cfg.get("server_startup_timeout", 600.0)
        )
        self.readiness_interval = float(
            reward_model_cfg.get("server_readiness_interval", 2.0)
        )
        self.process: subprocess.Popen | None = None

    @property
    def server_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def models_url(self) -> str:
        return f"{self.api_base}/models"

    def build_command(self) -> list[str]:
        model_path = self.reward_model_cfg.get("model_path")
        if not model_path:
            raise ValueError(
                "reward.model.model_path must be set to launch SGLang reward server"
            )

        served_model_name = str(
            self.reward_model_cfg.get("served_model_name")
            or Path(str(model_path)).name
            or "history_vlm_reward"
        )
        args: dict[str, Any] = {
            "model-path": model_path,
            "served-model-name": served_model_name,
            "host": self.host,
            "port": self.port,
            "trust-remote-code": True,
            "enable-multimodal": True,
            "dtype": "bfloat16",
            "disable-cuda-graph": True,
            "attention-backend": "triton",
            "sampling-backend": "pytorch",
            "grammar-backend": "none",
        }
        overrides = self.reward_model_cfg.get("sglang_server_args", {})
        overrides = _to_plain(overrides) or {}
        if not isinstance(overrides, dict):
            raise TypeError(
                "reward.model.sglang_server_args must be a dict, "
                f"got {type(overrides).__name__}."
            )
        for key, value in overrides.items():
            args[str(key).replace("_", "-")] = value

        command = [sys.executable, "-m", "sglang.launch_server"]
        for key, value in args.items():
            if value is None or value is False:
                continue
            flag = f"--{key}"
            if value is True:
                command.append(flag)
            else:
                command.extend([flag, str(value)])
        return command

    def start(self) -> str:
        if self.process is not None:
            raise RuntimeError("SGLang reward server has already been started.")

        command = self.build_command()
        logger.info("Starting SGLang reward server: %s", " ".join(command))
        self.process = subprocess.Popen(command, start_new_session=True)
        try:
            self.wait_until_ready()
        except Exception:
            self.stop()
            raise
        return self.api_base

    def wait_until_ready(self) -> None:
        try:
            from sglang.utils import wait_for_server
        except ImportError:
            wait_for_server = None

        if wait_for_server is not None:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    "SGLang reward server exited before readiness check succeeded "
                    f"with code {self.process.returncode}."
                )
            self._call_sglang_wait_for_server(wait_for_server)
            logger.info("SGLang reward server is ready at %s", self.api_base)
            return

        deadline = time.monotonic() + self.startup_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    "SGLang reward server exited before readiness check succeeded "
                    f"with code {self.process.returncode}."
                )
            try:
                with urllib.request.urlopen(self.models_url, timeout=5) as response:
                    if 200 <= response.status < 300:
                        logger.info(
                            "SGLang reward server is ready at %s", self.api_base
                        )
                        return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
            time.sleep(self.readiness_interval)

        raise TimeoutError(
            "Timed out waiting for SGLang reward server readiness at "
            f"{self.models_url}: {last_error}"
        )

    def _call_sglang_wait_for_server(self, wait_for_server) -> None:
        kwargs: dict[str, Any] = {}
        try:
            parameters = inspect.signature(wait_for_server).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_kwargs or "timeout" in parameters:
            kwargs["timeout"] = int(self.startup_timeout)

        try:
            wait_for_server(self.server_url, **kwargs)
        except TypeError as exc:
            if "timeout" not in kwargs or "timeout" not in str(exc):
                raise
            wait_for_server(self.server_url)

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            logger.info("Stopping SGLang reward server on %s", self.api_base)
            from sglang.utils import terminate_process

            terminate_process(self.process)
        self.process = None


def should_launch_sglang_reward_server(cfg: DictConfig) -> bool:
    reward_cfg = cfg.get("reward", {})
    if not reward_cfg.get("use_reward_model", False):
        return False
    model_cfg = reward_cfg.get("model", {})
    if model_cfg.get("model_type") != "history_vlm":
        return False
    if str(model_cfg.get("inference_backend", "")).lower() != "sglang":
        return False
    return not bool(model_cfg.get("api_base", None))


def maybe_start_sglang_reward_server(cfg: DictConfig) -> SGLangRewardServer | None:
    """Start the runner-owned SGLang reward server and inject ``api_base``."""
    if not should_launch_sglang_reward_server(cfg):
        return None

    server = SGLangRewardServer(cfg.reward.model)
    api_base = server.start()
    cfg.reward.model.api_base = api_base
    if not cfg.reward.model.get("served_model_name", None):
        cfg.reward.model.served_model_name = (
            Path(str(cfg.reward.model.model_path)).name or "history_vlm_reward"
        )
    return server
