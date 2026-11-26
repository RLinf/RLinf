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

"""RLinf environment for an isolated Isaac Lab-Arena runtime."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils._pytree import tree_map_only

from rlinf.scheduler import AcceleratorType

from .config import validate_arena_config
from .runtime import receive_message, send_message, start_watchdog

if TYPE_CHECKING:
    from rlinf.scheduler.manager import WorkerInfo


class IsaacLabArenaEnv(gym.Env):
    """Run one batch of DROID environments in a dedicated Kit process."""

    _request_timeout = 900.0
    _shutdown_timeout = 5.0

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: WorkerInfo,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = int(cfg.seed) + seed_offset
        self.is_start = True
        # auto_reset is part of the validated train/eval contract.
        self._evaluation = cfg.get("is_eval", not cfg.auto_reset)
        validate_arena_config(cfg, evaluation=self._evaluation)
        gpu = worker_info.accelerator_rank
        if worker_info.accelerator_type != AcceleratorType.NV_GPU or gpu < 0:
            raise ValueError(
                f"Place the Arena env worker on an NVIDIA RTX GPU, got {worker_info.accelerator_type} rank {gpu}"
            )
        self._python = Path(
            os.environ.get(
                "ISAACLAB_ARENA_PYTHON", Path(sys.prefix) / "arena-runtime/bin/python"
            )
        )
        if not self._python.is_file():
            raise FileNotFoundError(
                f"Arena interpreter {self._python} is missing. Install with requirements/install.sh embodied --model openpi --env isaaclab_arena or set ISAACLAB_ARENA_PYTHON."
            )
        self._settings = dict(
            OmegaConf.to_container(cfg.init_params, resolve=True),
            num_envs=num_envs,
            device=f"cuda:{int(os.environ.get('LOCAL_RANK', gpu))}",
            max_episode_steps=cfg.max_episode_steps,
            auto_reset=cfg.auto_reset,
            all_frames=bool(
                cfg.video_cfg.save_video or cfg.get("return_step_observations", False)
            ),
        )
        self._seed_stride = total_num_processes
        self._round = 0
        self._process: subprocess.Popen | None = None
        self._socket: socket.socket | None = None
        self._watchdog: subprocess.Popen | None = None
        self._request_lock = Lock()
        self._start(self.seed)

    def _start(self, seed: int) -> None:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["VIRTUAL_ENV"] = str(self._python.parent.parent)
        env["PATH"] = os.pathsep.join(
            (str(self._python.parent), env.get("PATH", os.defpath))
        )
        self._socket, child = socket.socketpair()
        try:
            with child:
                self._process = subprocess.Popen(
                    [
                        str(self._python),
                        "-I",
                        str(Path(__file__).with_name("runtime.py")),
                        str(child.fileno()),
                    ],
                    env=env,
                    pass_fds=(child.fileno(),),
                    start_new_session=True,
                )
            self._watchdog = start_watchdog(self._process.pid)
            self.metadata = {
                "render_fps": self._request(dict(self._settings, seed=seed))
            }
        except BaseException:
            self._close(graceful=False)
            raise

    @contextmanager
    def _exclusive_request(self):
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("An Arena simulator request is already in progress")
        try:
            yield
        finally:
            self._request_lock.release()

    def _request(self, message):
        try:
            if self._socket is None:
                raise RuntimeError("Arena simulator is closed")
            deadline = time.monotonic() + self._request_timeout
            send_message(self._socket, message, deadline)
            ok, result = receive_message(self._socket, deadline)
            if not ok:
                raise RuntimeError(f"Arena simulator failed:\n{result}")
        except BaseException as exc:
            # After an interrupted exchange neither frame boundaries nor the
            # command/response pairing can safely be recovered.
            self._close(graceful=False)
            if isinstance(exc, TimeoutError):
                raise TimeoutError(
                    f"Arena request exceeded its {self._request_timeout:g} second deadline"
                ) from exc
            if isinstance(exc, (EOFError, ConnectionError)):
                raise RuntimeError(
                    "Arena simulator disconnected during a request"
                ) from exc
            raise
        return tree_map_only(np.ndarray, torch.from_numpy, result)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        """Reset training placements or start a seeded evaluation round."""
        with self._exclusive_request():
            if self._socket is None:
                raise RuntimeError("Arena simulator is closed")
            if self._evaluation:
                seed = (
                    self.seed + self._round * self._seed_stride
                    if seed is None
                    else seed
                )
                # Arena binds the placement pool and single-use episode limit at construction.
                if self._round or seed != self.seed:
                    self._close(graceful=True)
                    self._start(seed)
            elif seed is not None:
                raise ValueError(
                    "Arena training seeds its placement pool at construction. Set env.train.seed "
                    "before launch and call reset() without a seed."
                )
            result = self._request(("reset", (seed, options)))
            self._round += 1
            return result

    def chunk_step(self, chunk_actions: np.ndarray | torch.Tensor) -> tuple:
        """Execute a policy chunk and return the requested observation frames."""
        shape = chunk_actions.shape
        if (
            len(shape) != 3
            or shape[0] != self.num_envs
            or shape[1] < 1
            or shape[2] != 8
        ):
            raise ValueError(
                f"Expected [{self.num_envs}, positive horizon, 8] DROID actions, got {shape}"
            )
        if not self._evaluation and shape[1] != 1:
            raise ValueError("Isaac Lab-Arena training requires num_action_chunks=1")
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = (
                chunk_actions.detach().to(device="cpu", dtype=torch.float32).numpy()
            )
        with self._exclusive_request():
            return self._request(
                ("chunk_step", np.asarray(chunk_actions, dtype=np.float32))
            )

    def step(self, actions: np.ndarray | torch.Tensor) -> tuple:
        """Advance each environment by one control step."""
        if actions.ndim != 2:
            raise ValueError(
                f"Expected [{self.num_envs}, 8] DROID actions, got {actions.shape}"
            )
        obs, reward, term, trunc, info = self.chunk_step(actions[:, None, :])
        return obs[0], reward[:, 0], term[:, 0], trunc[:, 0], info[0]

    def update_reset_state_ids(self) -> None:
        """Arena samples independent placements during reset."""

    def _close(self, *, graceful: bool) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        process = self._process
        if process is not None:
            try:
                if graceful:
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=self._shutdown_timeout)
            finally:
                # Kit helpers may still own the GPU after their leader exits.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                self._process = None
                if self._watchdog is not None:
                    self._watchdog.kill()
                    self._watchdog.wait()
                    self._watchdog = None

    def close(self) -> None:
        """Reap the Kit process and its entire private process group."""
        with self._exclusive_request():
            self._close(graceful=True)
