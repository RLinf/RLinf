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

"""Spawn isolated Bench2Dex scenes for the shared BaseVectorEnv scheduler."""

from __future__ import annotations

import multiprocessing
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Any, Callable

from rlinf.envs.venv import CloudpickleWrapper, EnvWorker, SubprocEnvWorker
from rlinf.utils.logging import get_logger

if TYPE_CHECKING:
    from .runtime import Bench2DexRuntime


@dataclass(frozen=True)
class _WorkerFailure:
    command: str
    traceback: str


def _worker(parent: Connection, remote: Connection, env_fn: CloudpickleWrapper) -> None:
    """Run the standard subprocess protocol with one scene/app owner."""
    parent.close()
    runtime = None
    command = "initialize"
    try:
        runtime = env_fn.data()
        while True:
            command, data = remote.recv()
            if command == "step":
                remote.send(runtime.step(data))
            elif command == "reset":
                remote.send(runtime.reset(**data))
            elif command == "getattr":
                remote.send(getattr(runtime, data, None))
            elif command == "setattr":
                setattr(runtime, data["key"], data["value"])
            elif command == "render":
                render = getattr(runtime, "render", None)
                remote.send(render(**data) if render is not None else None)
            elif command == "seed":
                runtime.reset(seed=data)
                remote.send(None)
            elif command == "close":
                owner, runtime = runtime, None
                owner.close()
                remote.send(None)
                return
            else:
                raise ValueError(f"Unexpected Bench2Dex worker command {command!r}")
    except EOFError:
        pass  # The parent closed its connection.
    except Exception:
        failure = _WorkerFailure(command, traceback.format_exc())
        try:
            remote.send(failure)
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        try:
            if runtime is not None:
                runtime.close()
        finally:
            remote.close()


class Bench2DexSubprocEnvWorker(SubprocEnvWorker):
    """Own one spawned simulator; reuse BaseVectorEnv's send-all/receive-all API.

    NumPy observations travel through the existing pipe protocol. Isaac imports
    and SimulationApp ownership stay inside the runtime factory in the child.
    """

    def __init__(self, env_fn: Callable[[], Bench2DexRuntime]) -> None:
        ctx = multiprocessing.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe()
        self.share_memory = False
        self.buffer = None
        self.process = ctx.Process(
            target=_worker,
            args=(self.parent_remote, self.child_remote, CloudpickleWrapper(env_fn)),
            daemon=True,
        )
        try:
            self.process.start()
        except BaseException:
            self.parent_remote.close()
            self.child_remote.close()
            raise
        self.child_remote.close()
        EnvWorker.__init__(self, env_fn)

    def recv(self) -> Any:
        """Receive a transition or raise the simulator's original traceback."""
        try:
            result = super().recv()
        except (EOFError, OSError) as exc:
            self.process.join(timeout=0.1)
            raise RuntimeError(
                f"Expected a Bench2Dex response; subprocess {self.process.pid} "
                f"closed its pipe (exit code {self.process.exitcode})"
            ) from exc
        if isinstance(result, _WorkerFailure):
            raise RuntimeError(
                f"Bench2Dex subprocess failed during {result.command}:\n"
                f"{result.traceback}"
            )
        return result

    def _request(self, command: str, data: Any) -> Any:
        self.parent_remote.send([command, data])
        return self.recv()

    def reset(self, **kwargs: Any) -> tuple[dict, dict]:
        """Reset this scene through the same protocol as batched resets."""
        self.send(None, **kwargs)
        return self.recv()

    def get_env_attr(self, key: str) -> Any:
        return self._request("getattr", key)

    def render(self, **kwargs: Any) -> Any:
        return self._request("render", kwargs)

    def seed(self, seed: int | None = None) -> None:
        self._request("seed", seed)

    def close_env(self) -> None:
        """Close this worker's child without waiting indefinitely on a failure."""
        try:
            if self.process.is_alive():
                try:
                    self.parent_remote.send(["close", None])
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self.process.join(timeout=10)
            if self.process.is_alive():
                get_logger().warning(
                    f"Bench2Dex subprocess {self.process.pid} did not close in "
                    "10 seconds; terminating this worker's child"
                )
                self.process.terminate()
                self.process.join(timeout=5)
            if self.process.is_alive():
                raise RuntimeError(
                    f"Expected Bench2Dex subprocess {self.process.pid} to exit "
                    "after termination; it is still alive"
                )
            self.process.join()
        finally:
            self.parent_remote.close()
