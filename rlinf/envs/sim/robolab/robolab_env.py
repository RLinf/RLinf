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
import signal
import socket
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.sim.robolab.protocol import receive, send


class RoboLabEnv:
    """Own a RoboLab simulator process and expose RLinf's evaluation protocol.

    ``cfg.max_episode_steps=None`` uses only the native task's episode limit.
    A positive value additionally limits the number of executed policy steps.
    """

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
        if cfg.max_episode_steps is not None and cfg.max_episode_steps <= 0:
            raise ValueError(
                f"Expected max_episode_steps > 0 or None, found {cfg.max_episode_steps}"
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
        self.seed = int(cfg.seed) + seed_offset
        settings.update(
            num_envs=num_envs,
            seed=self.seed,
            max_episode_steps=cfg.max_episode_steps,
        )
        env = os.environ.copy()
        env["OMNI_KIT_ACCEPT_EULA"] = "Y"
        device = settings.get("device", "cuda:0")
        if device.startswith("cuda"):
            ordinal = int(device.split(":")[1]) if ":" in device else 0
            if ordinal < 0:
                raise ValueError(f"Expected nonnegative CUDA ordinal, found {ordinal}")
            visible = env.get("CUDA_VISIBLE_DEVICES")
            physical = ordinal
            if visible:
                devices = [entry.strip() for entry in visible.split(",")]
                if all(entry.isdigit() for entry in devices):
                    if ordinal >= len(devices):
                        raise ValueError(
                            f"Device {device} is outside CUDA_VISIBLE_DEVICES={visible}"
                        )
                    # Kit's Vulkan device indices are not remapped by
                    # CUDA_VISIBLE_DEVICES, so unmask and select the GPU directly.
                    physical = int(devices[ordinal])
                    env.pop("CUDA_VISIBLE_DEVICES")
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
                start_new_session=True,
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
            metadata = self._request({"command": "init", "settings": settings})
            self.task_descriptions = metadata["task_descriptions"]
            self.max_episode_length = metadata["max_episode_length"]
            self.action_bounds = metadata["action_bounds"]
        except BaseException:
            self.close()
            raise
        finally:
            listener.close()

    def _request(self, message):
        send(self._socket, message)
        result = receive(self._socket)
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
        result = self._request({"command": "chunk_step", "actions": actions})

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

    def step(self, actions: np.ndarray | torch.Tensor) -> tuple:
        """Execute one action per environment, returning a batched Gym tuple."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        observations, rewards, terminations, truncations, infos = self.chunk_step(
            np.asarray(actions)[:, None, :]
        )
        return (
            observations[-1],
            rewards[:, 0],
            terminations[:, 0],
            truncations[:, 0],
            infos[-1],
        )

    def render_camera(self, camera_name: str, env_id: int = 0) -> np.ndarray:
        """Return the latest native RGB frame without advancing the simulation."""
        if camera_name not in {"over_shoulder_left_camera", "wrist_cam"}:
            raise ValueError(f"Unknown RoboLab camera: {camera_name!r}")
        if not 0 <= env_id < self.num_envs:
            raise ValueError(f"Expected env_id in [0, {self.num_envs}), found {env_id}")
        return self._request(
            {"command": "render_camera", "camera_name": camera_name, "env_id": env_id}
        )

    def get_camera_meta(self, camera_name: str, env_id: int = 0) -> dict[str, Any]:
        """Return the camera name and native image dimensions."""
        image = self.render_camera(camera_name, env_id)
        return {
            "camera_name": camera_name,
            "height": image.shape[0],
            "width": image.shape[1],
        }

    def get_status(self) -> list[dict[str, Any]]:
        """Return per-env native steps/success and effective termination flags.

        Success remains None until the native evaluator records a result, even
        when RLinf's configured policy-step limit truncates the episode.
        """
        return self._request({"command": "get_status"})

    def update_reset_state_ids(self):
        """RoboLab owns its reset state selection."""

    def close(self):
        """Close the simulator connection and reap the owned process."""
        if self._closed:
            return
        self._closed = True
        if self._socket is not None:
            try:
                self._socket.settimeout(1)
                send(self._socket, {"command": "close"})
            except OSError:
                pass
            self._socket.close()
        if self._process is not None:
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._process.wait()

    def __del__(self):
        self.close()
