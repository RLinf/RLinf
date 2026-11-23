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


"""RLinf adapter for RMBench's simulator process."""

import json
import os
import signal
import subprocess
from multiprocessing.connection import Pipe
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils._pytree import tree_map_only


class RMBenchEnv:
    """Run serial RMBench episodes through the standard evaluation worker."""

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ) -> None:
        del seed_offset, worker_info
        self._process = None
        if num_envs != 1 or total_num_processes != 1:
            raise ValueError(
                "RMBench expects one environment and one worker to preserve seed "
                f"order, got {num_envs} environments and {total_num_processes} workers."
            )
        if cfg.auto_reset or cfg.ignore_terminations:
            raise ValueError(
                "RMBench expects auto_reset=false and ignore_terminations=false, "
                f"got {cfg.auto_reset} and {cfg.ignore_terminations}."
            )

        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = int(cfg.seed)
        self.is_start = True
        params = cfg.init_params
        rmbench_path = Path(params.rmbench_path).expanduser().resolve()
        for relative_path in ("assets/embodiments/aloha-agilex", "assets/objects"):
            if not (rmbench_path / relative_path).is_dir():
                raise FileNotFoundError(
                    f"Missing RMBench assets at {rmbench_path / relative_path}. "
                    "Run bash scripts/_download_assets.sh from the RMBench checkout."
                )

        settings = {
            "task_name": params.task_name,
            "instruction_type": params.instruction_type,
            "seed": self.seed,
            "episodes": cfg.rollout_epoch,
            "max_episode_steps": cfg.max_episode_steps,
            "record_video": cfg.video_cfg.save_video,
            "short_memory": OmegaConf.to_container(cfg.short_memory, resolve=True)
            if cfg.get("short_memory") is not None
            else None,
        }
        self._connection, child = Pipe()
        env = os.environ.copy()
        repo_path = str(Path(__file__).resolve().parents[4])
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (repo_path, env.get("PYTHONPATH")))
        )
        try:
            self._process = subprocess.Popen(
                [
                    str(Path(params.sim_python).expanduser().absolute()),
                    "-m",
                    "rlinf.envs.sim.rmbench.sim_worker",
                    str(child.fileno()),
                    json.dumps(settings),
                ],
                cwd=rmbench_path,
                env=env,
                pass_fds=(child.fileno(),),
                start_new_session=True,
            )
        except BaseException:
            self._connection.close()
            raise
        finally:
            child.close()

    def _request(self, command: str, payload: Any = None) -> Any:
        self._connection.send((command, payload))
        return self._connection.recv()

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset to the next scene accepted by the RMBench expert."""
        return tree_map_only(np.ndarray, torch.from_numpy, self._request("reset")), {}

    def chunk_step(self, chunk_actions: np.ndarray) -> tuple:
        """Execute a chunk of Aloha joint and gripper targets."""
        observations, reward, terminated, truncated, info = self._request(
            "chunk_step", chunk_actions[0]
        )
        return (
            tree_map_only(np.ndarray, torch.from_numpy, observations),
            torch.from_numpy(reward),
            torch.from_numpy(terminated),
            torch.from_numpy(truncated),
            [{} for _ in observations[:-1]]
            + [tree_map_only(np.ndarray, torch.from_numpy, info)],
        )

    def step(self, actions: np.ndarray) -> tuple:
        """Execute one joint target per environment."""
        observations, rewards, terminations, truncations, infos = self.chunk_step(
            actions[:, None, :]
        )
        return (
            observations[-1],
            rewards[:, 0],
            terminations[:, 0],
            truncations[:, 0],
            infos[-1],
        )

    def update_reset_state_ids(self) -> None:
        """RMBench advances through accepted seeds in the simulator."""

    def close(self) -> None:
        """Close the pipe and reap the simulator and its planner processes."""
        process = self._process
        if process is None:
            return
        self._process = None
        self._connection.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def __del__(self) -> None:
        self.close()
