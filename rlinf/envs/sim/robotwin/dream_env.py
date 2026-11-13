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

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
from rlinf.scheduler import Channel, WorkerInfo
from rlinf.workers.env.env_worker import EnvWorker


class DreamObservationBuffer:
    """Collect one environment's control observations before worker transport."""

    def __init__(self) -> None:
        self.episode_id = 0
        self.samples: list[dict[str, Any]] = []
        self.last_sample: dict[str, Any] | None = None

    def reset(self) -> None:
        """Start an episode with an empty observation history."""
        self.episode_id += 1
        self.samples = []
        self.last_sample = None

    def append(
        self,
        observation: dict[str, Any],
        timestamp: float,
    ) -> None:
        """Store native RGB observations with measured state and time."""
        sample = {
            "frames": {
                "main_images": observation["main_images"].cpu(),
                "wrist_images": observation["wrist_images"].cpu(),
            },
            "state": observation["states"].detach().cpu().clone(),
            "timestamp": timestamp,
        }
        self.samples.append(sample)
        self.last_sample = sample

    def pack(
        self,
        task_descriptions: list[str],
        commanded: torch.Tensor | None,
        *,
        finished: bool,
        active_arms: tuple[str, ...] = ("left", "right"),
    ) -> dict[str, Any]:
        """Return a single-environment batch with an explicit control time axis."""
        samples = self.samples or [self.last_sample]
        return {
            "frames": [[sample["frames"] for sample in samples]],
            "states": torch.stack([sample["state"] for sample in samples], dim=1),
            "timestamps": torch.tensor(
                [[sample["timestamp"] for sample in samples]], dtype=torch.float64
            ),
            "task_descriptions": task_descriptions,
            "episode_id": torch.tensor([self.episode_id]),
            "finished": torch.tensor([finished]),
            "commanded": commanded,
            "active_arms": [active_arms],
        }


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
        self._capture_metadata = {}
        self._observed_task = None
        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )
        self.observations = DreamObservationBuffer()
        self.commanded = None

    def _detach_capture(self) -> None:
        if self._observed_task is not None:
            self._observed_task.get_obs = self._original_get_obs
            self._observed_task.scene.step = self._original_step
        self._observed_task = None
        self._capture_metadata = {}

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

        # Native joint_action contains drive targets; sample measured qpos at render time.
        def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raw = self._original_get_obs(*args, **kwargs)
            self._capture_metadata[id(raw["joint_action"]["vector"])] = (
                self.physics_time,
                self._measured_state(),
            )
            return raw

        scene.step, task.get_obs = step, capture

    def _extract_obs_image(self, raw_obs: list[dict[str, Any]]) -> dict[str, Any]:
        observation = super()._extract_obs_image(raw_obs)
        if self._observed_task is not None:
            timestamp, measured = self._capture_metadata[id(raw_obs[0]["state"])]
            observation["states"] = measured.to(observation["states"].device)
            observation["dream_sim_time"] = torch.tensor(
                [timestamp], dtype=torch.float64
            )
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

    def _pack(self, finished: bool = False) -> dict[str, Any]:
        return self.observations.pack(
            self.instructions,
            self.commanded,
            finished=finished,
            active_arms=tuple(self.active_arms),
        )

    def reset(self, *args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start SDK observation capture from the native reset observation."""
        self._detach_capture()
        observation, info = super().reset(*args, **kwargs)
        self._attach_capture()
        self.observations.reset()
        self.commanded = None
        self.instructions = observation["task_descriptions"]
        side = str(self.venv.envs[0].task.get_info().get("{a}", "")).lower()
        self.active_arms = [side] if side in ("left", "right") else ["left", "right"]
        observation["states"] = self._measured_state()
        self.observations.append(observation, 0.0)
        info["dream_observations"] = self._pack()
        return observation, info

    def chunk_step(self, actions: torch.Tensor) -> tuple:
        """Execute native RoboTwin actions and attach measured observations."""
        remaining = self.cfg.max_episode_steps - int(self.elapsed_steps[0])
        if remaining <= 0:
            raise RuntimeError("RoboTwin episode already reached its configured limit.")
        self._capture_metadata = {}
        result = super().chunk_step(actions[:, :remaining])
        observations, infos = result[0], result[-1]
        self.observations.samples.clear()
        for observation in observations:
            timestamp = float(observation["dream_sim_time"][0])
            if (
                not self.observations.samples
                or timestamp > self.observations.last_sample["timestamp"]
            ):
                self.observations.append(observation, timestamp)
        finished = bool(
            result[2].any()
            or result[3].any()
            or torch.as_tensor(infos[-1]["success"]).any()
        )
        self.commanded = torch.as_tensor(actions[:, :remaining][:, -1]).detach().cpu()
        infos[-1]["dream_observations"] = self._pack(finished)
        return result

    def close(self) -> None:
        """Restore capture hooks and release the native environment."""
        self._detach_capture()
        self.offload()


class DreamEnvWorker(EnvWorker):
    """Attach Dream observation capture before the standard RLinf wrappers."""

    def _setup_env_and_wrappers(
        self, env_cls: type, env_cfg: DictConfig, num_envs_per_stage: int
    ) -> list:
        if self._world_size != 1:
            raise ValueError("Dream requires exactly one environment worker.")
        return super()._setup_env_and_wrappers(
            DreamRobotwinEnv, env_cfg, num_envs_per_stage
        )

    def _build_rollout_input_data(self, env_batch: dict[str, Any]) -> dict[str, Any]:
        data = super()._build_rollout_input_data(env_batch)
        data["obs"] = {"history": [env_batch["env_infos"]["dream_observations"]]}
        return data

    def close(self) -> None:
        """Close the environments initialized by this worker."""
        for env in self.eval_env_list:
            env.close()
        self.eval_env_list = []

    def evaluate(
        self, input_channel: Channel, rollout_channel: Channel
    ) -> dict[str, torch.Tensor]:
        """Evaluate and release the environment on completion or failure."""
        try:
            return super().evaluate(input_channel, rollout_channel)
        finally:
            self.close()
