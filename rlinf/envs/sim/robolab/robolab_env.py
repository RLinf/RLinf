# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLinf adapter for a RoboLab simulator owned by a Ray actor."""

from __future__ import annotations

import json
import os
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname
from uuid import uuid4

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from rlinf.envs.sim.robolab.robolab_process import RoboLabProcess
from rlinf.envs.utils import valid_action_mask_from_counts


def _resolve_robolab_dir() -> Path:
    """Return the installed RoboLab source checkout containing its assets."""
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
        raise ValueError(f"RoboLab source must be local, found {source_url.geturl()}")
    robolab_dir = Path(url2pathname(source_url.path)).resolve()
    if not (robolab_dir / "assets").is_dir():
        raise FileNotFoundError(
            f"RoboLab assets not found beside the installed package at {robolab_dir}. "
            "Install RoboLab from a source checkout with its assets."
        )
    return robolab_dir


def _actor_environment(settings: dict) -> dict[str, str]:
    """Prepare actor environment variables and physical CUDA/Vulkan indices."""
    env_vars = {
        key: os.environ[key]
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "OMNI_KIT_ACCEPT_EULA",
            "ISAAC_PATH",
            "EXP_PATH",
            "CARB_APP_PATH",
        )
        if key in os.environ
    }
    env_vars["OMNI_KIT_ACCEPT_EULA"] = "Y"
    env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

    device = settings.get("device", "cuda:0")
    if not device.startswith("cuda"):
        return env_vars
    ordinal = int(device.split(":")[1]) if ":" in device else 0
    if ordinal < 0:
        raise ValueError(f"Expected nonnegative CUDA ordinal, found {ordinal}")
    visible = env_vars.get("CUDA_VISIBLE_DEVICES")
    physical = ordinal
    if visible:
        devices = [entry.strip() for entry in visible.split(",")]
        if all(entry.isdigit() for entry in devices):
            if ordinal >= len(devices):
                raise ValueError(
                    f"Device {device} is outside CUDA_VISIBLE_DEVICES={visible}"
                )
            # Vulkan device indices are not remapped by CUDA_VISIBLE_DEVICES.
            physical = int(devices[ordinal])
            env_vars.pop("CUDA_VISIBLE_DEVICES")
    settings["device"] = f"cuda:{physical}"
    settings["kit_args"] = (
        settings.get("kit_args", "") + f" --/renderer/activeGpu={physical}"
    )
    return env_vars


class RoboLabEnv:
    """Expose a RoboLab Ray actor through RLinf's rollout environment contract."""

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        del total_num_processes, worker_info
        self.cfg = cfg
        self.num_envs = num_envs
        self.is_start = True
        self._process = None
        self._closed = False
        self.auto_reset = bool(cfg.auto_reset)
        self.ignore_terminations = bool(cfg.ignore_terminations)

        if self.ignore_terminations:
            raise NotImplementedError(
                "RoboLab does not support ignore_terminations=true because its "
                "native evaluator freezes successful environments."
            )
        if cfg.max_episode_steps is not None and cfg.max_episode_steps <= 0:
            raise ValueError(
                f"Expected max_episode_steps > 0 or None, found {cfg.max_episode_steps}"
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
            auto_reset=self.auto_reset,
        )
        self._start_process(_resolve_robolab_dir(), settings)

    def _start_process(self, robolab_dir: Path, settings: dict) -> None:
        env_vars = _actor_environment(settings)
        self._process = RoboLabProcess.options(
            name=f"RoboLabProcess-{uuid4().hex}",
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(), soft=False
            ),
            runtime_env={"env_vars": env_vars},
        ).remote(str(robolab_dir), settings)
        try:
            metadata = ray.get(self._process.ready.remote(), timeout=300)
            self.task_descriptions = metadata["task_descriptions"]
            self.max_episode_length = metadata["max_episode_length"]
            self.action_bounds = metadata["action_bounds"]
        except BaseException:
            self._kill_process()
            self._closed = True
            raise

    def _kill_process(self) -> None:
        process, self._process = getattr(self, "_process", None), None
        if process is not None and ray.is_initialized():
            try:
                ray.kill(process, no_restart=True)
            except ray.exceptions.RayActorError:
                pass

    @staticmethod
    def _observation(data: dict) -> dict:
        return {
            "main_images": torch.from_numpy(np.array(data["main_images"], copy=True)),
            "wrist_images": torch.from_numpy(np.array(data["wrist_images"], copy=True)),
            "states": torch.from_numpy(np.array(data["states"], copy=True)),
            "task_descriptions": data["task_descriptions"],
            "active_mask": torch.from_numpy(np.array(data["active_mask"], copy=True)),
        }

    def reset(self, seed=None, env_ids=None):
        """Reset all or selected slots, optionally reseeding the simulator."""
        result = ray.get(self._process.reset.remote(seed=seed, env_ids=env_ids))
        self.is_start = False
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
        result = ray.get(self._process.chunk_step.remote(actions))
        observation = self._observation(result["observation"])
        episode = {
            key: torch.from_numpy(value) for key, value in result["episode"].items()
        }
        info = {"episode": episode}
        executed_steps = result.get("executed_steps")
        if executed_steps is not None:
            info["valid_action_mask"] = valid_action_mask_from_counts(
                executed_steps, actions.shape[1]
            )
        final_mask = result.get("final_mask")
        if final_mask is not None and np.asarray(final_mask).any():
            final_mask = torch.from_numpy(np.asarray(final_mask, dtype=bool).copy())
            info.update(
                final_observation=self._observation(result["final_observation"]),
                final_info={"episode": episode},
                _final_observation=final_mask,
                _final_info=final_mask,
                _elapsed_steps=final_mask,
            )
        return (
            [observation],
            torch.from_numpy(result["rewards"]),
            torch.from_numpy(result["terminations"]),
            torch.from_numpy(result["truncations"]),
            [info],
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
        return ray.get(self._process.render_camera.remote(camera_name, env_id))

    def get_camera_meta(self, camera_name: str, env_id: int = 0) -> dict[str, Any]:
        """Return the camera name and native image dimensions."""
        image = self.render_camera(camera_name, env_id)
        return {
            "camera_name": camera_name,
            "height": image.shape[0],
            "width": image.shape[1],
        }

    def get_status(self) -> list[dict[str, Any]]:
        """Return per-env native steps/success and effective termination flags."""
        return ray.get(self._process.get_status.remote())

    def update_reset_state_ids(self):
        """RoboLab owns its reset state selection."""

    def close(self):
        """Close and reap the owned RoboLab Ray actor."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        process, self._process = getattr(self, "_process", None), None
        if process is not None and ray.is_initialized():
            try:
                ray.get(process.close.remote(), timeout=30)
            except (
                ray.exceptions.GetTimeoutError,
                ray.exceptions.RayActorError,
            ):
                pass
            finally:
                try:
                    ray.kill(process, no_restart=True)
                except ray.exceptions.RayActorError:
                    pass

    def __del__(self):
        self.close()
