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

"""Stateful embodied simulator tool worker for SessionServer / AgentLoop."""

from __future__ import annotations

import asyncio
import base64
import io
import multiprocessing as mp
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from omegaconf import DictConfig, OmegaConf

from rlinf.agents.tool_call.schema import ToolChannelRequest, ToolChannelResponse
from rlinf.models.embodiment.qwen_vlm.protocol import (
    mirror_instruction,
    native_step_budget,
    reset_wait_steps,
)
from rlinf.scheduler import Channel
from rlinf.workers.agent.tool_worker import ToolWorker


def encode_rgb_to_png_b64(image: Any) -> str:
    """Encode an HxWx3 uint8 numpy array (or PIL Image) to a base64 PNG string."""
    from PIL import Image

    if isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        pil_img = Image.fromarray(image).convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class LiberoEpisodeBackend:
    """One stateful LIBERO simulator instance driven by relative VLM actions."""

    def __init__(
        self,
        spec: dict[str, Any],
        *,
        camera_height: int = 512,
        camera_width: int = 512,
        reset_gripper_open: bool = True,
        num_steps_wait: Optional[int] = None,
        seed: int = 7,
    ) -> None:
        self._spec = dict(spec)
        self._camera_height = int(camera_height)
        self._camera_width = int(camera_width)
        self._reset_gripper_open = bool(reset_gripper_open)
        self._num_steps_wait = num_steps_wait
        self._seed = int(seed)
        self._env: Any = None
        self._remaining_native_budget = 0
        self._done = False
        self._instruction = ""
        self._render_cache: dict[tuple[int, int, bool], Any] = {}

    def _noop_render(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        *,
        camera_name: Optional[str] = None,
        depth: bool = False,
        mode: str = "offscreen",
        device_id: int = -1,
        segmentation: bool = False,
    ) -> Any:
        """Cached render stub that skips offscreen rendering on intermediate steps."""
        del camera_name, mode, device_id, segmentation
        import numpy as np

        h, w = height or 1, width or 1
        key = (h, w, depth)
        result = self._render_cache.get(key)
        if result is None:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            result = (frame, np.zeros((h, w), dtype=np.float32)) if depth else frame
            self._render_cache[key] = result
        return result

    @contextmanager
    def _no_render(self):
        """Suppress camera renders inside the block; proprioception stays live."""
        from rlinf.envs.sim.libero.vlm_controller import _robosuite_env

        sim = _robosuite_env(self._env).sim
        render = sim.render
        sim.render = self._noop_render
        try:
            yield
        finally:
            sim.render = render

    def open(self) -> dict[str, Any]:
        """Create, reset, and settle the LIBERO episode; return Turn 0 obs."""
        import numpy as np

        from rlinf.envs.sim.libero.libero_env import (
            _init_libero,
            _repoint_libero_config,
        )
        from rlinf.envs.sim.libero.utils import (
            get_benchmark_overridden,
            get_libero_image,
            get_libero_wrist_image,
        )
        from rlinf.envs.sim.libero.vlm_controller import _robosuite_env, observe_state

        _init_libero()
        import libero.libero as l_base
        from libero.libero.envs import OffScreenRenderEnv

        _repoint_libero_config(l_base)
        bddl_root = l_base.get_libero_path("bddl_files")

        suite_name = str(self._spec["task_suite_name"])
        task_id = int(self._spec["task_id"])
        trial_id = int(self._spec["trial_id"])
        suite = get_benchmark_overridden(suite_name)()
        task = suite.get_task(task_id)
        bddl_path = os.path.join(bddl_root, task.problem_folder, task.bddl_file)

        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=self._camera_height,
            camera_widths=self._camera_width,
            hard_reset=False,
        )
        self._env.seed(self._seed)
        self._env.reset()
        init_states = suite.get_task_init_states(task_id)
        self._env.set_init_state(init_states[trial_id])

        wait_steps = reset_wait_steps(True, self._num_steps_wait)
        zero_action = np.zeros(7, dtype=np.float64)
        if self._reset_gripper_open:
            zero_action[-1] = -1.0
        with self._no_render():
            for _ in range(wait_steps):
                self._env.step(zero_action)
        raw_obs = _robosuite_env(self._env)._get_observations(force_update=True)

        source_suite = str(self._spec.get("source_suite", suite_name))
        self._remaining_native_budget = native_step_budget(source_suite)
        self._instruction = str(
            self._spec.get("instruction") or mirror_instruction(str(task.language))
        )
        self._done = False
        return {
            "state": observe_state(self._env),
            "image_data": [
                encode_rgb_to_png_b64(get_libero_image(raw_obs)),
                encode_rgb_to_png_b64(get_libero_wrist_image(raw_obs)),
            ],
            "instruction": self._instruction,
            "native_budget": self._remaining_native_budget,
            "done": False,
            "reward": 0.0,
        }

    def step(self, action: Optional[dict[str, dict[str, float]]]) -> dict[str, Any]:
        """Execute one macro action via ``vlm_controller.execute_action``."""
        from rlinf.envs.sim.libero.utils import (
            get_libero_image,
            get_libero_wrist_image,
        )
        from rlinf.envs.sim.libero.vlm_controller import _robosuite_env, execute_action

        if self._env is None or self._done:
            return {
                "state": None,
                "image_data": None,
                "done": True,
                "reward": 0.0,
                "success": False,
                "outcome": "already_done",
                "native_steps": 0,
                "remaining_native_budget": self._remaining_native_budget,
            }

        with self._no_render():
            result = execute_action(
                self._env,
                action,
                native_budget=self._remaining_native_budget,
            )
        self._remaining_native_budget = max(
            self._remaining_native_budget - int(result["native_steps"]), 0
        )
        done = bool(result["done"]) or self._remaining_native_budget <= 0
        self._done = done
        image_data = None
        if not done:
            raw_obs = _robosuite_env(self._env)._get_observations(force_update=True)
            image_data = [
                encode_rgb_to_png_b64(get_libero_image(raw_obs)),
                encode_rgb_to_png_b64(get_libero_wrist_image(raw_obs)),
            ]
        return {
            "state": result.get("state"),
            "image_data": image_data,
            "done": done,
            "reward": float(result["reward"]),
            "success": bool(result["success"]),
            "outcome": str(result["outcome"]),
            "native_steps": int(result["native_steps"]),
            "remaining_native_budget": self._remaining_native_budget,
        }

    def close(self) -> None:
        """Release the underlying simulator instance (idempotent)."""
        env = self._env
        self._env = None
        self._done = True
        self._render_cache.clear()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _libero_slot_worker_main(conn: Any, init_kwargs: dict[str, Any]) -> None:
    """Run one LIBERO backend on a single thread inside an isolated process.

    MuJoCo's OSMesa/EGL OpenGL contexts are thread-local and not thread-safe;
    isolating each active environment in a dedicated spawned process avoids
    cross-thread ``GLContext.free()`` segfaults and removes GIL contention.
    """
    backend: Optional[LiberoEpisodeBackend] = None
    try:
        while True:
            try:
                cmd, payload = conn.recv()
            except EOFError:
                break
            if cmd == "shutdown":
                break
            try:
                if cmd == "open":
                    if backend is not None:
                        backend.close()
                    backend = LiberoEpisodeBackend(payload, **init_kwargs)
                    conn.send(("ok", backend.open()))
                elif cmd == "step":
                    if backend is None:
                        raise RuntimeError("LIBERO slot received step before open.")
                    conn.send(("ok", backend.step(payload)))
                elif cmd == "close":
                    if backend is not None:
                        backend.close()
                        backend = None
                    conn.send(("ok", None))
                else:
                    conn.send(("err", f"unknown command: {cmd!r}"))
            except Exception as exc:
                conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        if backend is not None:
            backend.close()
        try:
            conn.close()
        except Exception:
            pass


class _SubprocessEnvSlot:
    """One reusable spawned worker process hosting a single LIBERO episode."""

    def __init__(self, init_kwargs: dict[str, Any]) -> None:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=_libero_slot_worker_main,
            args=(child_conn, dict(init_kwargs)),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()

    @property
    def alive(self) -> bool:
        return self._proc.is_alive()

    def call(self, cmd: str, payload: Any = None) -> Any:
        if not self._proc.is_alive():
            raise RuntimeError(
                f"LIBERO worker subprocess exited unexpectedly (exitcode={self._proc.exitcode})"
            )
        self._conn.send((cmd, payload))
        status, result = self._conn.recv()
        if status != "ok":
            raise RuntimeError(str(result))
        return result

    def shutdown(self) -> None:
        try:
            if self._proc.is_alive():
                self._conn.send(("shutdown", None))
                self._proc.join(timeout=5.0)
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)
        try:
            self._conn.close()
        except Exception:
            pass


class _SubprocessLiberoBackend:
    """Adapter exposing a reusable ``_SubprocessEnvSlot`` as a standard backend."""

    def __init__(
        self,
        spec: dict[str, Any],
        acquire_slot: Callable[[], _SubprocessEnvSlot],
        release_slot: Callable[[_SubprocessEnvSlot], None],
    ) -> None:
        self._spec = spec
        self._acquire_slot = acquire_slot
        self._release_slot = release_slot
        self._slot: Optional[_SubprocessEnvSlot] = None

    def open(self) -> dict[str, Any]:
        if self._slot is None:
            self._slot = self._acquire_slot()
        return self._slot.call("open", self._spec)

    def step(self, action: Optional[dict[str, dict[str, float]]]) -> dict[str, Any]:
        assert self._slot is not None
        return self._slot.call("step", action)

    def close(self) -> None:
        slot = self._slot
        self._slot = None
        if slot is None:
            return
        try:
            if slot.alive:
                slot.call("close", None)
        except Exception:
            slot.shutdown()
        finally:
            self._release_slot(slot)


@dataclass
class _ActiveEnvSession:
    """One live simulator session and the lock serializing its thread calls."""

    backend: Any
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class EmbodiedEnvToolWorker(ToolWorker):
    """Stateful ToolWorker serving embodied simulator sessions over channels.

    Each trajectory opens its own session with ``request_type="session_start"``,
    steps it with ``request_type="execute"`` and ``tool_name="embodied_env"``,
    and closes it with ``request_type="session_end"``. Simulator steps execute
    in a bounded thread pool so stepping one environment never stalls other
    concurrent trajectories.
    """

    TOOL_NAME = "embodied_env"

    def __init__(
        self,
        cfg: DictConfig,
        backend_factory: Optional[Callable[[dict[str, Any]], Any]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        tools_cfg = cfg.get("tools", None)
        embodied_cfg = (
            tools_cfg.get("embodied_env", None) if tools_cfg is not None else None
        )
        if embodied_cfg is None:
            embodied_cfg = OmegaConf.create({})
        self._camera_height = int(embodied_cfg.get("camera_height", 512))
        self._camera_width = int(embodied_cfg.get("camera_width", 512))
        self._reset_gripper_open = bool(embodied_cfg.get("reset_gripper_open", True))
        self._num_steps_wait = embodied_cfg.get("num_steps_wait", None)
        self._seed = int(embodied_cfg.get("seed", cfg.get("seed", 7)))
        self._max_concurrent_envs = int(embodied_cfg.get("max_concurrent_envs", 32))
        self._backend_factory = backend_factory

        self._sessions: dict[str, _ActiveEnvSession] = {}
        self._idle_slots: list[_SubprocessEnvSlot] = []
        self._slots_lock = threading.Lock()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._worker_route_key: Optional[str] = None
        self._request_processor_tasks: list[asyncio.Task] = []
        self._inflight_tasks: set[asyncio.Task] = set()

    def _acquire_libero_slot(self) -> _SubprocessEnvSlot:
        while True:
            with self._slots_lock:
                slot = self._idle_slots.pop() if self._idle_slots else None
            if slot is None:
                break
            if slot.alive:
                return slot
            slot.shutdown()
        return _SubprocessEnvSlot(
            {
                "camera_height": self._camera_height,
                "camera_width": self._camera_width,
                "reset_gripper_open": self._reset_gripper_open,
                "num_steps_wait": self._num_steps_wait,
                "seed": self._seed,
            }
        )

    def _release_libero_slot(self, slot: _SubprocessEnvSlot) -> None:
        keep = False
        if slot.alive:
            with self._slots_lock:
                if len(self._idle_slots) < self._max_concurrent_envs:
                    self._idle_slots.append(slot)
                    keep = True
        if not keep:
            slot.shutdown()

    def init_worker(self, input_channel: Channel, output_channel: Channel) -> None:
        self.input_channel = input_channel
        self.output_channel = output_channel

    def start_server(self) -> None:
        loop = asyncio.get_running_loop()
        self._worker_route_key = f"worker-{self._rank}"
        self._semaphore = asyncio.Semaphore(self._max_concurrent_envs)
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_concurrent_envs,
            thread_name_prefix="embodied-env",
        )
        self._request_processor_tasks = [
            loop.create_task(self._process_session_starts()),
            loop.create_task(self._process_routed_requests()),
        ]

    def stop_server(self) -> None:
        for task in self._request_processor_tasks:
            if not task.done():
                task.cancel()
        self._request_processor_tasks.clear()
        for task in list(self._inflight_tasks):
            if not task.done():
                task.cancel()
        for session_id in list(self._sessions):
            session = self._sessions.pop(session_id, None)
            if session is not None:
                try:
                    session.backend.close()
                except Exception:
                    pass
        with self._slots_lock:
            idle_slots = list(self._idle_slots)
            self._idle_slots.clear()
        for slot in idle_slots:
            slot.shutdown()
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def _create_backend(self, spec: dict[str, Any]) -> Any:
        if self._backend_factory is not None:
            return self._backend_factory(spec)
        embodiment = str(spec.get("embodiment", "libero")).lower()
        if embodiment == "libero":
            return _SubprocessLiberoBackend(
                spec,
                self._acquire_libero_slot,
                self._release_libero_slot,
            )
        if embodiment == "robodojo":
            from rlinf.envs.sim.robodojo.vlm_action import execute_action

            raise NotImplementedError(
                f"RoboDojo backend requires Isaac Sim / cuRobo ({execute_action.__module__})"
            )
        raise ValueError(f"Unsupported embodied_env embodiment: {embodiment!r}")

    async def _run_in_executor(self, fn: Callable[..., Any], *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    async def handle_request(
        self, request: ToolChannelRequest, *, slot_acquired: bool = False
    ) -> ToolChannelResponse:
        """Process one session_start / execute / session_end request."""
        if request.request_type == "session_start":
            return await self._handle_session_start(
                request, slot_acquired=slot_acquired
            )
        if request.request_type == "execute":
            return await self._handle_execute(request)
        if request.request_type == "session_end":
            return await self._handle_session_end(request)
        return ToolChannelResponse(
            success=False,
            result=f"unsupported request_type: {request.request_type!r}",
        )

    async def _handle_session_start(
        self, request: ToolChannelRequest, *, slot_acquired: bool = False
    ) -> ToolChannelResponse:
        if request.session_id in self._sessions:
            if slot_acquired and self._semaphore is not None:
                self._semaphore.release()
            return ToolChannelResponse(
                success=False,
                result=f"session {request.session_id!r} already exists",
            )
        if not slot_acquired and self._semaphore is not None:
            await self._semaphore.acquire()
        spec = dict(request.tool_args or {})
        backend: Any = None
        try:
            backend = self._create_backend(spec)
            initial_obs = await self._run_in_executor(backend.open)
            self._sessions[request.session_id] = _ActiveEnvSession(backend=backend)
            result = dict(initial_obs)
            if self._worker_route_key is not None:
                result["worker_route_key"] = self._worker_route_key
            return ToolChannelResponse(success=True, result=result)
        except Exception as exc:
            if backend is not None:
                try:
                    await self._run_in_executor(backend.close)
                except Exception:
                    pass
            if self._semaphore is not None:
                self._semaphore.release()
            self.log_error(
                f"EmbodiedEnvToolWorker session_start failed for "
                f"{request.session_id}: {exc!r}"
            )
            return ToolChannelResponse(success=False, result=str(exc))

    async def _handle_execute(self, request: ToolChannelRequest) -> ToolChannelResponse:
        if request.tool_name not in (None, self.TOOL_NAME):
            return ToolChannelResponse(
                success=False,
                result=f"unsupported tool_name: {request.tool_name!r}",
            )
        session = self._sessions.get(request.session_id)
        if session is None:
            return ToolChannelResponse(
                success=False,
                result=f"unknown embodied_env session {request.session_id!r}",
            )
        tool_args = request.tool_args or {}
        action = tool_args.get("action")
        try:
            async with session.lock:
                step_result = await self._run_in_executor(session.backend.step, action)
            return ToolChannelResponse(success=True, result=step_result)
        except Exception as exc:
            self.log_error(
                f"EmbodiedEnvToolWorker execute failed for "
                f"{request.session_id}: {exc!r}"
            )
            return ToolChannelResponse(success=False, result=str(exc))

    async def _handle_session_end(
        self, request: ToolChannelRequest
    ) -> ToolChannelResponse:
        session = self._sessions.pop(request.session_id, None)
        if session is None:
            return ToolChannelResponse(success=True, result={"closed": False})
        try:
            async with session.lock:
                await self._run_in_executor(session.backend.close)
        except Exception as exc:
            self.log_warning(
                f"EmbodiedEnvToolWorker session_end failed for "
                f"{request.session_id}: {exc!r}"
            )
        finally:
            if self._semaphore is not None:
                self._semaphore.release()
        return ToolChannelResponse(success=True, result={"closed": True})

    async def _dispatch_and_reply(
        self, request: ToolChannelRequest, *, slot_acquired: bool = False
    ) -> None:
        response = await self.handle_request(request, slot_acquired=slot_acquired)
        await self.output_channel.put(
            response, key=request.session_id, async_op=True
        ).async_wait()

    async def _process_session_starts(self) -> None:
        """Pull unrouted requests (session_start) only while this worker has capacity."""
        while True:
            assert self._semaphore is not None
            await self._semaphore.acquire()
            try:
                request: ToolChannelRequest = await self.input_channel.get(
                    async_op=True
                ).async_wait()
            except BaseException:
                self._semaphore.release()
                raise
            if request.request_type == "session_start":
                coro = self._dispatch_and_reply(request, slot_acquired=True)
            else:
                self._semaphore.release()
                coro = self._dispatch_and_reply(request, slot_acquired=False)
            task = asyncio.create_task(coro)
            self._inflight_tasks.add(task)
            task.add_done_callback(self._inflight_tasks.discard)

    async def _process_routed_requests(self) -> None:
        """Serve session-pinned execute/session_end requests for sessions on this rank."""
        assert self._worker_route_key is not None
        while True:
            request: ToolChannelRequest = await self.input_channel.get(
                key=self._worker_route_key, async_op=True
            ).async_wait()
            task = asyncio.create_task(self._dispatch_and_reply(request))
            self._inflight_tasks.add(task)
            task.add_done_callback(self._inflight_tasks.discard)
