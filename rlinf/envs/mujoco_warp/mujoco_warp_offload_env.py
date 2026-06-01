# Copyright 2025 The RLinf Authors.
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

"""Subprocess-offloaded MuJoCo-Warp environment proxy.

Follows the same pattern as ``ManiskillOffloadEnv``: the actual env
(``CartPoleTask`` or ``CubePickTask``) lives in a spawned subprocess so
that GPU memory can be freed between rollout phases.
"""

from __future__ import annotations

import traceback
from typing import Any, Optional

import torch.multiprocessing as mp

from rlinf.envs.mujoco_warp.tasks.cartpole import CartPoleTask
from rlinf.envs.mujoco_warp.tasks.cubepick import CubePickTask

_TASK_CLASSES = {
    "cartpole": CartPoleTask,
    "cube_pick": CubePickTask,
}


def _mujoco_warp_worker_main(
    cfg,
    task_name: str,
    num_envs: int,
    seed_offset: int,
    total_num_processes: int,
    worker_info: Any,
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    state_buffer: Optional[bytes],
):
    """Entry point for the subprocess running the actual MuJoCo-Warp env."""
    try:
        env_cls = _TASK_CLASSES[task_name]
        env = env_cls(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )
        if state_buffer:
            env.load_state(state_buffer)

        result_queue.put({"status": "ready"})

        while True:
            command = command_queue.get()
            method_name = command.get("method")

            if method_name == "shutdown":
                break

            try:
                args = command.get("args", [])
                kwargs = command.get("kwargs", {})

                if method_name == "__setattr__":
                    attr_name, attr_value = args
                    setattr(env, attr_name, attr_value)
                    result_queue.put({"status": "success", "data": None})
                    continue

                if not hasattr(env, method_name):
                    result_queue.put(
                        {
                            "status": "error",
                            "error": f"Method '{method_name}' not found",
                        }
                    )
                    continue

                method = getattr(env, method_name)
                if not callable(method):
                    result_queue.put(
                        {
                            "status": "error",
                            "error": f"Method '{method_name}' is not callable",
                        }
                    )
                    continue

                result = method(*args, **kwargs)
                result_queue.put({"status": "success", "data": result})
            except Exception as err:
                result_queue.put(
                    {
                        "status": "error",
                        "error": str(err),
                        "traceback": traceback.format_exc(),
                    }
                )
    except Exception as err:
        result_queue.put(
            {
                "status": "error",
                "error": str(err),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        command_queue.close()
        result_queue.close()


class MuJoCoWarpOffloadEnv:
    """Proxy environment that runs MuJoCo-Warp env in a spawned subprocess.

    This class does not implement env APIs directly.  Missing method calls
    are forwarded to the subprocess via ``_rpc()``.

    Lifecycle:
    - ``offload()``: snapshots state (``get_state``) and shuts down the worker.
    - ``onload()``: starts worker, restores from ``state_buffer`` if available.
    - ``_rpc(...)``: auto-calls ``onload()`` when the process is offloaded.
    """

    _LOCAL_ATTRS = frozenset(
        {
            "cfg",
            "num_envs",
            "seed_offset",
            "total_num_processes",
            "worker_info",
            "task_name",
            "rpc_timeout_s",
            "context",
            "process",
            "command_queue",
            "result_queue",
            "state_buffer",
            "_has_valid_state_buffer",
        }
    )

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
        record_metrics: bool = True,
        rpc_timeout_s: int = 120,
    ):
        del record_metrics
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.rpc_timeout_s = rpc_timeout_s

        init_params = cfg.init_params if hasattr(cfg, "init_params") else {}
        self.task_name = (
            init_params.get("task_name", "cartpole") if init_params else "cartpole"
        )

        self.context: Optional[mp.context.BaseContext] = None
        self.process: Optional[mp.Process] = None
        self.command_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.state_buffer: Optional[bytes] = None
        self._has_valid_state_buffer = False
        self.onload()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_env(self) -> None:
        if self.process is not None and self.process.is_alive():
            return

        self.context = mp.get_context("spawn")
        self.command_queue = self.context.Queue()
        self.result_queue = self.context.Queue()
        self.process = self.context.Process(
            target=_mujoco_warp_worker_main,
            kwargs={
                "cfg": self.cfg,
                "task_name": self.task_name,
                "num_envs": self.num_envs,
                "seed_offset": self.seed_offset,
                "total_num_processes": self.total_num_processes,
                "worker_info": self.worker_info,
                "command_queue": self.command_queue,
                "result_queue": self.result_queue,
                "state_buffer": self.state_buffer,
            },
        )
        self.process.start()
        try:
            result = self.result_queue.get(timeout=self.rpc_timeout_s)
        except Exception as err:
            self._force_shutdown()
            raise RuntimeError(
                f"MuJoCo-Warp offload init timeout/failure: {err}"
            ) from err
        if result.get("status") != "ready":
            self._force_shutdown()
            raise RuntimeError(f"MuJoCo-Warp offload init failed: {result}")

    def onload(self, force: bool = False) -> None:
        if not force and self.process is not None and self.process.is_alive():
            return
        self.start_env()
        if (not self._has_valid_state_buffer) and getattr(
            self.cfg, "auto_reset", False
        ):
            self._rpc("reset")

    def offload(self, keep_state: bool = True) -> None:
        if self.process is None or not self.process.is_alive():
            return

        if keep_state:
            try:
                self.state_buffer = self._rpc("get_state")
                self._has_valid_state_buffer = True
            except Exception:
                self.state_buffer = None
                self._has_valid_state_buffer = False
        else:
            self.state_buffer = None
            self._has_valid_state_buffer = False

        try:
            self.command_queue.put({"method": "shutdown"})
        finally:
            self._force_shutdown()

    def stop_env(self) -> None:
        self.offload(keep_state=True)

    def close(self) -> None:
        if self.process is not None and self.process.is_alive():
            try:
                self._rpc("close", timeout_s=30)
            except Exception:
                pass
        self.offload(keep_state=False)

    # ------------------------------------------------------------------
    # State serialisation
    # ------------------------------------------------------------------

    def get_state(self) -> bytes:
        return self._rpc("get_state")

    def load_state(self, state: bytes) -> None:
        self._rpc("load_state", args=[state])

    # ------------------------------------------------------------------
    # RPC
    # ------------------------------------------------------------------

    def _rpc(self, method_name: str, args=None, kwargs=None, timeout_s=None):
        if self.process is None or not self.process.is_alive():
            self.onload()
        payload = {
            "method": method_name,
            "args": args or [],
            "kwargs": kwargs or {},
        }
        self.command_queue.put(payload)
        result = self.result_queue.get(timeout=timeout_s or self.rpc_timeout_s)
        if result.get("status") == "error":
            trace = result.get("traceback")
            if trace:
                raise RuntimeError(f"{result.get('error')}\n{trace}")
            raise RuntimeError(result.get("error", "Unknown offload RPC error"))
        return result.get("data")

    def _force_shutdown(self) -> None:
        if self.command_queue is not None:
            self.command_queue.close()
        if self.result_queue is not None:
            self.result_queue.close()
        if self.process is not None:
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
        self.process = None
        self.command_queue = None
        self.result_queue = None

    # ------------------------------------------------------------------
    # Attribute proxy
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        def method_proxy(*args, **kwargs):
            return self._rpc(name, args=args, kwargs=kwargs)

        return method_proxy

    def __setattr__(self, name: str, value):
        if name in self._LOCAL_ATTRS:
            super().__setattr__(name, value)
            return

        if name.startswith("_"):
            raise AttributeError(
                f"Cannot set private attribute '{name}' on offloaded environment"
            )
        self._rpc("__setattr__", args=[name, value])
