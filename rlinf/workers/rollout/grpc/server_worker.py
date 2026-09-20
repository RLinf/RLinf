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

"""Ray ownership of a policy server process, following the SGLang launcher."""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any

import ray.util
from omegaconf import OmegaConf

from rlinf.scheduler import Worker

from .policy_server import serve_policy


class PolicyServerWorker(Worker):
    """Own one server subprocess and stop it on startup failure or shutdown."""

    def __init__(self, config: Any, startup_timeout_s: float) -> None:
        super().__init__()
        self.config = config
        self.startup_timeout_s = startup_timeout_s
        self._process = None
        self._address = None

    def init_server(self) -> None:
        """Start on the allocated node; return only after the model is loaded."""
        if self._process is not None:
            raise RuntimeError("Policy server is already started")
        host = ray.util.get_node_ip_address()
        context = mp.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        self._process = context.Process(
            target=serve_policy,
            args=(
                OmegaConf.create(OmegaConf.to_container(self.config, resolve=True)),
                host,
                writer,
            ),
            daemon=True,
        )
        try:
            self._process.start()
            writer.close()
            deadline = time.monotonic() + self.startup_timeout_s
            while not reader.poll(0.1):
                if not self._process.is_alive():
                    raise RuntimeError("Policy server exited during startup")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Policy server startup timed out")
            success, result = reader.recv()
            if not success:
                raise RuntimeError(f"Policy server startup failed: {result}")
            self._address = f"{host}:{result}"
            self.log_info(f"Policy server ready at {self._address}")
        except BaseException:
            self.shutdown()
            raise
        finally:
            reader.close()
            writer.close()

    def get_server_address(self) -> str:
        """Return the reachable endpoint of the initialized service."""
        if self._address is None:
            raise RuntimeError("Policy server is not ready")
        return self._address

    def shutdown(self) -> None:
        """Terminate only the subprocess owned by this worker."""
        process = self._process
        if process is None:
            return
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            process.close()
        self._process = None
        self._address = None
