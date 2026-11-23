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

"""Dream observation capture for RoboTwin."""

import base64
import io
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image, ImageOps

from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
from rlinf.scheduler import WorkerInfo


def _encode_frames(observation: dict[str, Any]) -> dict[str, str]:
    main, wrists = observation["main_images"], observation["wrist_images"]
    frames = {}
    for role, image in zip(
        ("overhead", "wrist_l", "wrist_r"),
        (main[0], wrists[0, 0], wrists[0, 1]),
        strict=True,
    ):
        buffer = io.BytesIO()
        ImageOps.pad(
            Image.fromarray(image.cpu().numpy()),
            (320, 240),
            method=Image.Resampling.BILINEAR,
            color=(0, 0, 0),
        ).save(buffer, format="JPEG", quality=85)
        frames[f"video.{role}"] = base64.b64encode(buffer.getvalue()).decode("ascii")
    return frames


class DreamRobotwinEnv(RoboTwinEnv):
    """Capture RGB, measured joints and simulator time for Dream."""

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: WorkerInfo,
        record_metrics: bool = True,
    ) -> None:
        self._observed_task = None
        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )

    def _detach_capture(self) -> None:
        if self._observed_task is not None:
            self._observed_task.get_obs = self._original_get_obs
            self._observed_task.scene.step = self._original_step
        self._observed_task = None

    def _attach_capture(self) -> None:
        task = self.venv.envs[0].task
        scene = task.scene
        self._original_get_obs = task.get_obs
        self._original_step = scene.step
        self._observed_task = task
        self.physics_time = 0.0

        def step() -> None:
            self._original_step()
            self.physics_time += scene.get_timestep()

        # RoboTwin forwards this vector as state when it flattens each observation.
        def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raw = self._original_get_obs(*args, **kwargs)
            raw["joint_action"]["vector"] = np.append(
                self._measured_state().numpy().reshape(-1), self.physics_time
            )
            return raw

        scene.step, task.get_obs = step, capture

    def _extract_obs_image(self, raw_obs: list[dict[str, Any]]) -> dict[str, Any]:
        observation = super()._extract_obs_image(raw_obs)
        if self._observed_task is not None:
            state = observation["states"]
            observation["states"] = state[:, :-1].float()
            observation["dream_sim_time"] = state[:, -1]
        return observation

    def _measured_state(self) -> torch.Tensor:
        robot = self.venv.envs[0].task.robot
        values = []
        for side in ("left", "right"):
            entity = getattr(robot, f"{side}_entity")
            active = list(entity.get_active_joints())
            qpos = np.asarray(entity.get_qpos())
            values.extend(
                float(qpos[active.index(j)])
                for j in getattr(robot, f"{side}_arm_joints")
            )
            finger, ratio, offset = getattr(robot, f"{side}_gripper")[0]
            low, high = getattr(robot, f"{side}_gripper_scale")
            opening = (qpos[active.index(finger)] - offset) / ratio
            values.append(float(np.clip((opening - low) / (high - low), 0, 1)))
        return torch.tensor([values], dtype=torch.float32)

    def _attach_history(
        self,
        observations: list[dict[str, Any]],
        times: list[float],
        commanded: torch.Tensor | None = None,
        finished: bool = False,
    ) -> None:
        frames, states, timestamps = [], [], []
        for observation, timestamp in zip(observations, times, strict=True):
            if timestamps and timestamp <= timestamps[-1]:
                continue
            frames.append(_encode_frames(observation))
            states.append(observation["states"].cpu())
            timestamps.append(timestamp)
        last_obs = observations[-1]
        last_obs["history"] = [
            {
                "frames": [frames],
                "states": torch.stack(states, dim=1),
                "timestamps": torch.tensor([timestamps], dtype=torch.float64),
                "task_descriptions": last_obs["task_descriptions"],
                "finished": torch.tensor([finished]),
                "commanded": commanded,
                "active_arms": [self.active_arms],
            }
        ]

    def reset(self, *args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start SDK observation capture from the native reset observation."""
        self._detach_capture()
        observation, info = super().reset(*args, **kwargs)
        self._attach_capture()
        side = str(self.venv.envs[0].task.get_info().get("{a}", "")).lower()
        self.active_arms = (side,) if side in ("left", "right") else ("left", "right")
        observation["states"] = self._measured_state()
        self._attach_history([observation], [0.0])
        return observation, info

    def chunk_step(self, actions: torch.Tensor) -> tuple:
        """Execute native RoboTwin actions and attach measured observations."""
        remaining = self.cfg.max_episode_steps - int(self.elapsed_steps[0])
        actions = actions[:, :remaining]
        result = super().chunk_step(actions)
        observations, infos = result[0], result[-1]
        finished = bool(
            result[2].any()
            or result[3].any()
            or torch.as_tensor(infos[-1]["success"]).any()
        )
        self._attach_history(
            observations,
            [float(observation["dream_sim_time"][0]) for observation in observations],
            commanded=torch.as_tensor(actions[:, -1]).detach().cpu(),
            finished=finished,
        )
        return result

    def close(self) -> None:
        """Restore capture hooks and release the native environment."""
        self._detach_capture()
        self.offload()
