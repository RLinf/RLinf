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

"""RLinf environment adapter for RoboLab's Isaac Sim tasks."""

from __future__ import annotations

import json
import os
import pickle
import socket
import struct
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

import numpy as np
import torch
from omegaconf import OmegaConf


def _receive(sock: socket.socket) -> dict:
    header = sock.recv(8, socket.MSG_WAITALL)
    if len(header) != 8:
        raise ConnectionError("RoboLab simulator closed the connection")
    size = struct.unpack("!Q", header)[0]
    parts = []
    while size:
        part = sock.recv(min(size, 1024 * 1024))
        if not part:
            raise ConnectionError("RoboLab simulator closed the connection")
        parts.append(part)
        size -= len(part)
    return pickle.loads(b"".join(parts))


def _send(sock: socket.socket, value) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


class RoboLabEnv:
    """Own a RoboLab simulator process and expose RLinf's evaluation protocol."""

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        del total_num_processes, worker_info
        self.cfg = cfg
        self.num_envs = num_envs
        self.is_start = True
        self._socket = None
        self._process = None
        self._closed = False

        if cfg.auto_reset or cfg.ignore_terminations:
            raise NotImplementedError(
                "RoboLab evaluation requires auto_reset=false and "
                "ignore_terminations=false; received "
                f"auto_reset={cfg.auto_reset}, "
                f"ignore_terminations={cfg.ignore_terminations}"
            )
        if cfg.max_episode_steps <= 0:
            raise ValueError(
                f"Expected max_episode_steps > 0, found {cfg.max_episode_steps}"
            )

        try:
            package = distribution("robolab")
        except PackageNotFoundError as exc:
            raise ModuleNotFoundError(
                "RoboLab is not installed. Run requirements/install.sh embodied "
                "--model openpi --env robolab in this Python environment."
            ) from exc
        direct_url = package.read_text("direct_url.json")
        if direct_url is None:
            raise FileNotFoundError(
                "RoboLab must be installed from a source checkout with its assets. "
                "Run requirements/install.sh embodied --model openpi --env robolab."
            )
        source_url = urlsplit(json.loads(direct_url)["url"])
        if source_url.scheme != "file":
            raise ValueError(
                f"RoboLab source must be local, found {source_url.geturl()}"
            )
        robolab_dir = Path(url2pathname(source_url.path)).resolve()
        if not (robolab_dir / "assets").is_dir():
            raise FileNotFoundError(
                f"RoboLab assets not found beside the installed package at {robolab_dir}. "
                "Install RoboLab from a source checkout with its assets."
            )

        settings = OmegaConf.to_container(cfg.init_params, resolve=True)
        output_dir = Path(settings["output_dir"])
        settings["output_dir"] = str(
            output_dir if output_dir.is_absolute() else Path.cwd() / output_dir
        )
        settings.update(
            num_envs=num_envs,
            seed=int(cfg.seed) + seed_offset,
            max_episode_steps=int(cfg.max_episode_steps),
        )
        env = os.environ.copy()
        env["OMNI_KIT_ACCEPT_EULA"] = "Y"
        device = settings.get("device", "cuda:0")
        if device.startswith("cuda"):
            ordinal = int(device.split(":")[1]) if ":" in device else 0
            visible = env.pop("CUDA_VISIBLE_DEVICES", None)
            physical = int(visible.split(",")[ordinal]) if visible else ordinal
            # Kit's Vulkan device indices are not remapped by CUDA_VISIBLE_DEVICES.
            settings["device"] = f"cuda:{physical}"
            settings["kit_args"] = (
                settings.get("kit_args", "") + f" --/renderer/activeGpu={physical}"
            )

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(1)
        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("sim_worker.py")),
                    str(listener.getsockname()[1]),
                ],
                cwd=robolab_dir,
                env=env,
            )
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                try:
                    self._socket, _ = listener.accept()
                    break
                except socket.timeout:
                    if self._process.poll() is not None:
                        raise RuntimeError(
                            f"RoboLab simulator exited with code {self._process.returncode}"
                        )
            if self._socket is None:
                raise TimeoutError("RoboLab simulator did not connect in 300 seconds")
            self._request({"command": "init", "settings": settings})
        except BaseException:
            self.close()
            raise
        finally:
            listener.close()

    def _request(self, message):
        _send(self._socket, message)
        result = _receive(self._socket)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"RoboLab simulator error:\n{result['error']}")
        return result

    @staticmethod
    def _observation(data):
        return {
            "main_images": torch.from_numpy(np.array(data["main_images"], copy=True)),
            "wrist_images": torch.from_numpy(np.array(data["wrist_images"], copy=True)),
            "states": torch.from_numpy(np.array(data["states"], copy=True)),
            "task_descriptions": data["task_descriptions"],
            "active_mask": torch.from_numpy(np.array(data["active_mask"], copy=True)),
        }

    def reset(self, seed=None, env_ids=None):
        """Reset the full evaluation batch, optionally reseeding the simulator."""
        if env_ids is not None:
            raise ValueError("RoboLab evaluation resets all environments together")
        result = self._request({"command": "reset", "seed": seed})
        return self._observation(result), {}

    def chunk_step(self, chunk_actions):
        """Execute an action chunk and return observations and episode metrics."""
        actions = np.asarray(chunk_actions, dtype=np.float32)
        if (
            actions.ndim != 3
            or actions.shape[0] != self.num_envs
            or actions.shape[2] != 8
        ):
            raise ValueError(
                f"RoboLab expects actions [num_envs, horizon, 8], found {actions.shape}"
            )
        result = self._request(
            {
                "command": "chunk_step",
                "action_shape": actions.shape,
                "action_bytes": actions.tobytes(),
            }
        )

        observation = self._observation(result["observation"])
        return (
            [observation],
            torch.from_numpy(result["rewards"]),
            torch.from_numpy(result["terminations"]),
            torch.from_numpy(result["truncations"]),
            [
                {
                    "episode": {
                        key: torch.from_numpy(value)
                        for key, value in result["episode"].items()
                    }
                }
            ],
        )

    def update_reset_state_ids(self):
        """RoboLab owns its reset state selection."""

    def close(self):
        """Close the simulator connection and reap the owned process."""
        if self._closed:
            return
        self._closed = True
        if self._socket is not None:
            try:
                _send(self._socket, {"command": "close"})
            except OSError:
                pass
            self._socket.close()
        if self._process is not None:
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()

    def __del__(self):
        self.close()
