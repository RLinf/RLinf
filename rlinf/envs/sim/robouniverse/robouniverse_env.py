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

"""RLinf embodied environment adapter for RoboUniverse-10K tasks."""

from __future__ import annotations

import copy
import os
import shutil
import signal
import socket
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.sim.ipc import (
    receive_message,
    resolve_physical_cuda_device,
    send_message,
)
from rlinf.envs.sim.robouniverse.catalog import RoboUniverseCatalog
from rlinf.envs.sim.robouniverse.materializer import (
    MaterializedTask,
    TaskMaterializer,
)
from rlinf.utils.logging import get_logger

logger = get_logger()

# Protocol 4 works across Python 3.9 through 3.14 so RLinf's parent venv and an
# external Isaac Sim 5.1.0 Python 3.11 interpreter interoperate cleanly.
_PICKLE_PROTOCOL = 4


def _resolve_gpu_env(
    device: str, base_env: dict[str, str]
) -> tuple[int, dict[str, str]]:
    """Map a logical CUDA device string to a physical GPU index for Isaac Sim."""
    physical, env = resolve_physical_cuda_device(device, base_env)
    env["EMBODY_SIM_GPU"] = str(physical)
    return physical, env


class _SimWorkerProcess:
    """Own one isolated ``sim_worker.py`` subprocess and its IPC socket pair."""

    def __init__(
        self,
        *,
        sim_python: str,
        settings: dict[str, Any],
        env: dict[str, str],
        startup_timeout_s: float = 300.0,
    ) -> None:
        self._socket: socket.socket | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._closed = False

        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            os.set_inheritable(child_sock.fileno(), True)
            self._process = subprocess.Popen(
                [
                    sim_python,
                    str(Path(__file__).with_name("sim_worker.py")),
                    str(child_sock.fileno()),
                ],
                env=env,
                pass_fds=(child_sock.fileno(),),
                start_new_session=True,
            )
            child_sock.close()
            self._socket = parent_sock
            try:
                self.request(
                    {"command": "init", "settings": settings},
                    timeout=startup_timeout_s,
                )
            except EOFError as exc:
                code = self._process.wait(timeout=5.0)
                raise RuntimeError(
                    f"RoboUniverse simulator worker exited during init (code={code})."
                ) from exc
        except BaseException:
            child_sock.close()
            if self._socket is None:
                parent_sock.close()
            self.close()
            raise

    def request(
        self, message: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        """Send one synchronous request and return the worker's response."""
        if self._closed or self._socket is None:
            raise RuntimeError("RoboUniverse simulator worker is closed.")
        previous_timeout = self._socket.gettimeout()
        if timeout is not None:
            self._socket.settimeout(timeout)
        try:
            send_message(self._socket, message, protocol=_PICKLE_PROTOCOL)
            reply = receive_message(self._socket, peer_name="RoboUniverse simulator")
        finally:
            if timeout is not None and self._socket is not None:
                self._socket.settimeout(previous_timeout)
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"RoboUniverse simulator error:\n{reply['error']}")
        return reply

    def close(self) -> None:
        """Close the connection and reap the owned process group idempotently."""
        if self._closed:
            return
        self._closed = True
        if self._socket is not None:
            try:
                self._socket.settimeout(1.0)
                send_message(
                    self._socket, {"command": "close"}, protocol=_PICKLE_PROTOCOL
                )
            except OSError:
                pass
            self._socket.close()
            self._socket = None
        if self._process is not None:
            try:
                self._process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._process.wait()
            self._process = None


class RoboUniverseEnv:
    """Own RoboUniverse-10K simulator workers and expose RLinf's env contract.

    Each slot runs in an isolated subprocess (typically Isaac Sim 5.1.0's
    Python) so simulator dependencies remain decoupled from RLinf's policy
    environment. Tasks are sharded deterministically across workers and
    unpacked on demand into a bounded LRU cache.
    """

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        worker_info: Any = None,
    ) -> None:
        del worker_info
        self._closed = False
        self._workers: list[_SimWorkerProcess] = []
        self._executor: ThreadPoolExecutor | None = None
        self._owned_work_dir = False
        self._work_dir: Path | None = None

        self.cfg = cfg
        self._num_envs = int(num_envs)
        self._seed_offset = int(seed_offset)
        self._total_num_processes = int(total_num_processes)
        self._seed = int(cfg.get("seed", 0)) + self._seed_offset
        self.is_start = True
        self._first_reset = True

        if self._num_envs < 1:
            raise ValueError(f"Expected num_envs >= 1, found {self._num_envs}.")
        self._group_size = int(cfg.get("group_size", 1) or 1)
        if self._group_size < 1:
            raise ValueError(f"Expected group_size >= 1, found {self._group_size}.")
        if self._num_envs % self._group_size != 0:
            raise ValueError(
                f"Expected group_size ({self._group_size}) to divide "
                f"num_envs ({self._num_envs})."
            )
        self._max_episode_steps = int(cfg.max_episode_steps)
        if self._max_episode_steps <= 0:
            raise ValueError(
                f"Expected max_episode_steps > 0, found {self._max_episode_steps}."
            )
        self._auto_reset = bool(cfg.get("auto_reset", False))
        self._ignore_terminations = bool(cfg.get("ignore_terminations", False))
        if self._group_size > 1 and self._auto_reset:
            raise ValueError(
                "RoboUniverse with group_size > 1 requires auto_reset=false so "
                "all group members stay aligned on the same task."
            )

        init_params = OmegaConf.to_container(cfg.init_params, resolve=True)
        if not isinstance(init_params, dict):
            raise ValueError(
                f"Expected env.init_params mapping, found {type(init_params).__name__}."
            )

        allow_cpu_fixture = bool(init_params.get("allow_cpu_fixture", False))
        accept_isaac_eula = bool(init_params.get("accept_isaac_eula", False))
        if not accept_isaac_eula and not allow_cpu_fixture:
            raise ValueError(
                "RoboUniverse requires env.init_params.accept_isaac_eula=true "
                "after reviewing the NVIDIA Isaac Sim license."
            )

        data_root = init_params.get("data_root")
        if not data_root:
            raise ValueError(
                "Expected env.init_params.data_root pointing to the RoboUniverse-10K "
                "dataset directory containing catalog.csv."
            )
        self._catalog = RoboUniverseCatalog(data_root)
        selected = self._catalog.select(
            split=init_params.get("split", "heldout500"),
            robot=init_params.get("robot"),
            cases=init_params.get("cases"),
        )
        robots = {record.robot for record in selected}
        if len(robots) > 1:
            raise ValueError(
                "RoboUniverseEnv requires a single robot family per env group so "
                f"action and state dimensions stay uniform; found {sorted(robots)}. "
                "Set init_params.robot to 'franka' or 'agilex', or configure "
                "separate entries under env.<mode>.envs."
            )
        self._robot = next(iter(robots))
        self._tasks = RoboUniverseCatalog.shard(
            selected, self._seed_offset, self._total_num_processes
        )

        self._num_groups = self._num_envs // self._group_size
        self._cycle_tasks = bool(init_params.get("cycle_tasks", True))
        self._task_cursor = [
            (slot // self._group_size) % len(self._tasks)
            for slot in range(self._num_envs)
        ]

        work_dir_raw = init_params.get("work_dir")
        if work_dir_raw:
            self._owned_work_dir = False
            self._work_dir = Path(work_dir_raw).expanduser().resolve()
        else:
            self._owned_work_dir = True
            self._work_dir = Path(
                tempfile.mkdtemp(prefix=f"rlinf_robouniverse_r{self._seed_offset}_")
            )

        self._materializer = TaskMaterializer(
            data_root=data_root,
            work_dir=self._work_dir,
            tasks_dir=self._work_dir / "tasks" / f"rank_{self._seed_offset}",
            asset_cache_dir=init_params.get("asset_cache_dir"),
            max_cached_tasks=init_params.get("max_cached_tasks", 32),
            verify_sha256=bool(init_params.get("verify_sha256", True)),
            offline=bool(init_params.get("offline", False)),
        )

        sim_python = str(init_params.get("sim_python", "/isaac-sim/python.sh"))
        device = str(init_params.get("device", "cuda:0"))
        device_index, worker_env = _resolve_gpu_env(device, os.environ)
        if accept_isaac_eula:
            worker_env["OMNI_KIT_ACCEPT_EULA"] = "YES"

        worker_settings = {
            "device_index": device_index,
            "max_episode_steps": self._max_episode_steps,
            "clip_actions": bool(init_params.get("clip_actions", True)),
            "include_joint_velocity": bool(
                init_params.get("include_joint_velocity", False)
            ),
            "grade_on_done": bool(init_params.get("grade_on_done", True)),
            "keep_evidence": bool(init_params.get("keep_evidence", False)),
        }
        self._startup_timeout_s = float(init_params.get("startup_timeout_s", 300.0))

        self._slot_cases: list[str] = [""] * self._num_envs
        self._slot_obs: list[dict[str, Any] | None] = [None] * self._num_envs
        self._episode_counters: list[int] = [0] * self._num_envs
        self._action_dim: int | None = None
        self._state_dim: int | None = None
        self._executor = (
            ThreadPoolExecutor(max_workers=self._num_envs)
            if self._num_envs > 1
            else None
        )

        try:
            for slot in range(self._num_envs):
                slot_env = dict(worker_env)
                kit_cache_dir = (
                    self._work_dir
                    / "kit_cache"
                    / f"rank_{self._seed_offset}"
                    / f"slot_{slot}"
                )
                kit_cache_dir.mkdir(parents=True, exist_ok=True)
                slot_env.setdefault("OMNI_KIT_USER_CACHE", str(kit_cache_dir))
                self._workers.append(
                    _SimWorkerProcess(
                        sim_python=sim_python,
                        settings=worker_settings,
                        env=slot_env,
                        startup_timeout_s=self._startup_timeout_s,
                    )
                )
        except BaseException:
            self.close()
            raise

    @property
    def num_envs(self) -> int:
        """Number of simulator slots owned by this env adapter."""
        return self._num_envs

    @property
    def seed(self) -> int:
        """Effective seed used for video naming and logging."""
        return self._seed

    @property
    def action_dim(self) -> int | None:
        """Action dimension reported by the loaded task's ``env_spec.json``."""
        return self._action_dim

    @property
    def state_dim(self) -> int | None:
        """Proprioceptive state dimension of the loaded task."""
        return self._state_dim

    @property
    def current_cases(self) -> list[str]:
        """Currently loaded RoboUniverse case IDs per slot."""
        return list(self._slot_cases)

    def _active_cases(self) -> frozenset[str]:
        return frozenset(self._tasks[cursor].case for cursor in self._task_cursor)

    def _advance_group(self, group_id: int) -> None:
        first = group_id * self._group_size
        next_idx = self._task_cursor[first] + self._num_groups
        if next_idx >= len(self._tasks):
            next_idx = (
                next_idx % len(self._tasks)
                if self._cycle_tasks
                else self._task_cursor[first]
            )
        for slot in range(first, first + self._group_size):
            self._task_cursor[slot] = next_idx

    def _load_slot(self, slot: int, materialized: MaterializedTask) -> dict[str, Any]:
        self._episode_counters[slot] += 1
        run_dir = (
            self._work_dir
            / "runs"
            / f"rank_{self._seed_offset}"
            / f"slot_{slot}"
            / f"ep_{self._episode_counters[slot]:06d}_{materialized.record.case}"
        )
        return self._workers[slot].request(
            {
                "command": "load_task",
                "task": {
                    "case": materialized.record.case,
                    "workspace_dir": str(materialized.workspace_dir),
                    "runtime_dir": str(materialized.runtime_dir),
                    "instruction": materialized.instruction,
                    "env_spec": materialized.env_spec,
                    "run_dir": str(run_dir),
                },
            },
            timeout=self._startup_timeout_s,
        )

    @staticmethod
    def _collate_observations(slot_obs: list[dict[str, Any]]) -> dict[str, Any]:
        batched: dict[str, Any] = {
            "main_images": torch.from_numpy(
                np.stack([o["main_images"] for o in slot_obs], axis=0)
            ),
            "states": torch.from_numpy(
                np.stack([o["states"] for o in slot_obs], axis=0)
            ),
            "task_descriptions": [o["task_descriptions"] for o in slot_obs],
        }
        if all("wrist_images" in o for o in slot_obs):
            batched["wrist_images"] = torch.from_numpy(
                np.stack([o["wrist_images"] for o in slot_obs], axis=0)
            )
        return batched

    def reset(
        self,
        seed: int | None = None,
        env_ids: list[int] | np.ndarray | torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset all or selected simulator slots to their current task."""
        del seed
        if env_ids is None:
            slots = list(range(self._num_envs))
        else:
            if isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.detach().cpu().tolist()
            slots = [int(idx) for idx in env_ids]
            for idx in slots:
                if not 0 <= idx < self._num_envs:
                    raise ValueError(
                        f"Expected env_id in [0, {self._num_envs}), found {idx}."
                    )

        if not self._first_reset:
            groups = sorted({slot // self._group_size for slot in slots})
            for group_id in groups:
                self._advance_group(group_id)

        protected = self._active_cases()
        records = [self._tasks[self._task_cursor[slot]] for slot in slots]
        materialized = [
            self._materializer.materialize(record, protected_cases=protected)
            for record in records
        ]

        if self._executor is not None and len(slots) > 1:
            futures = [
                self._executor.submit(self._load_slot, slot, task)
                for slot, task in zip(slots, materialized)
            ]
            replies = [future.result() for future in futures]
        else:
            replies = [
                self._load_slot(slot, task) for slot, task in zip(slots, materialized)
            ]

        for slot, record, reply in zip(slots, records, replies):
            self._slot_cases[slot] = record.case
            self._slot_obs[slot] = reply["observation"]
            meta = reply["metadata"]
            if self._action_dim is None:
                self._action_dim = meta["action_dim"]
                self._state_dim = meta["state_dim"]
            elif meta["action_dim"] != self._action_dim:
                raise ValueError(
                    f"Action dim mismatch across tasks: expected {self._action_dim}, "
                    f"found {meta['action_dim']} in {record.case!r}."
                )

        self.is_start = False
        self._first_reset = False
        obs = self._collate_observations(self._slot_obs)
        return obs, {"cases": list(self._slot_cases)}

    def chunk_step(
        self, chunk_actions: np.ndarray | torch.Tensor
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        """Execute an action chunk ``[num_envs, horizon, action_dim]``."""
        if self._first_reset or self._action_dim is None:
            raise RuntimeError("Call reset() before chunk_step() on RoboUniverseEnv.")
        if isinstance(chunk_actions, torch.Tensor):
            actions = chunk_actions.detach().cpu().numpy()
        else:
            actions = np.asarray(chunk_actions)
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        if actions.ndim == 2 and self._num_envs == 1:
            actions = actions[None, :, :]
        if (
            actions.ndim != 3
            or actions.shape[0] != self._num_envs
            or actions.shape[2] != self._action_dim
        ):
            raise ValueError(
                "RoboUniverseEnv expects chunk_actions shaped "
                f"({self._num_envs}, horizon, {self._action_dim}), found {actions.shape}."
            )

        if self._executor is not None:
            futures = [
                self._executor.submit(
                    self._workers[slot].request,
                    {"command": "chunk_step", "actions": actions[slot]},
                )
                for slot in range(self._num_envs)
            ]
            replies = [future.result() for future in futures]
        else:
            replies = [
                self._workers[slot].request(
                    {"command": "chunk_step", "actions": actions[slot]}
                )
                for slot in range(self._num_envs)
            ]

        horizon = actions.shape[1]
        for slot, reply in enumerate(replies):
            self._slot_obs[slot] = reply["observations"][-1]
            if reply["infra_error_message"]:
                logger.warning(
                    "[robouniverse] slot=%d case=%s infrastructure error: %s",
                    slot,
                    reply["case"],
                    reply["infra_error_message"],
                )

        obs_list = [
            self._collate_observations(
                [
                    replies[slot]["observations"][step_idx]
                    for slot in range(self._num_envs)
                ]
            )
            for step_idx in range(horizon)
        ]
        rewards = torch.from_numpy(
            np.stack([reply["rewards"] for reply in replies], axis=0)
        )
        terminations = torch.from_numpy(
            np.stack([reply["terminations"] for reply in replies], axis=0)
        )
        truncations = torch.from_numpy(
            np.stack([reply["truncations"] for reply in replies], axis=0)
        )

        if self._ignore_terminations:
            truncations = truncations | terminations
            terminations = torch.zeros_like(terminations)

        past_terminations = terminations.any(dim=1)
        past_truncations = truncations.any(dim=1)
        if self._auto_reset or self._ignore_terminations:
            terminations = torch.zeros_like(terminations)
            terminations[:, -1] = past_terminations
            truncations = torch.zeros_like(truncations)
            truncations[:, -1] = past_truncations

        episode_metrics = {
            key: torch.tensor(
                [reply["episode"][key] for reply in replies], dtype=torch.float32
            )
            for key in replies[0]["episode"]
        }
        last_info: dict[str, Any] = {
            "episode": episode_metrics,
            "verdicts": [reply["verdict"] for reply in replies],
            "cases": [reply["case"] for reply in replies],
        }
        infos_list: list[dict[str, Any]] = [{} for _ in range(horizon - 1)] + [
            last_info
        ]

        dones = past_terminations | past_truncations
        if dones.any() and self._auto_reset:
            done_indices = torch.nonzero(dones, as_tuple=False).flatten().tolist()
            final_obs = copy.deepcopy(obs_list[-1])
            final_info = copy.deepcopy(infos_list[-1])
            reset_obs, _ = self.reset(env_ids=done_indices)
            obs_list[-1] = reset_obs
            infos_list[-1] = {
                "episode": final_info["episode"],
                "final_observation": final_obs,
                "final_info": final_info,
                "_final_observation": dones.clone(),
                "_final_info": dones.clone(),
                "cases": list(self._slot_cases),
            }

        return obs_list, rewards, terminations, truncations, infos_list

    def step(
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        """Execute one control step per slot and return a batched Gym tuple."""
        if isinstance(actions, torch.Tensor):
            arr = actions.detach().cpu().numpy()
        else:
            arr = np.asarray(actions)
        if arr.ndim == 1 and self._num_envs == 1:
            arr = arr[None, :]
        obs_list, rewards, terminations, truncations, infos_list = self.chunk_step(
            arr[:, None, :]
        )
        return (
            obs_list[-1],
            rewards[:, 0],
            terminations[:, 0],
            truncations[:, 0],
            infos_list[-1],
        )

    def update_reset_state_ids(self) -> None:
        """Task progression is owned by ``reset`` via the sharded catalog cursor."""

    def close(self) -> None:
        """Terminate all owned simulator subprocesses and release resources."""
        if self._closed:
            return
        self._closed = True
        for worker in self._workers:
            worker.close()
        self._workers.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._owned_work_dir and self._work_dir is not None:
            shutil.rmtree(self._work_dir, ignore_errors=True)

    def __del__(self) -> None:
        self.close()
