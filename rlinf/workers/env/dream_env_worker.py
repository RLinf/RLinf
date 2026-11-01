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

"""Dream observation transport through the standard environment worker."""

from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker


class DreamEnvWorker(EnvWorker):
    """Attach Dream observation capture before the standard RLinf wrappers."""

    def _setup_env_and_wrappers(
        self, env_cls: type, env_cfg: DictConfig, num_envs_per_stage: int
    ) -> list:
        if self._world_size != 1:
            raise ValueError("Dream requires exactly one environment worker.")
        if env_cfg.env_type == "real":
            from rlinf.envs.real.dream_env import DreamRealEnv

            env_cls = DreamRealEnv
        else:
            from rlinf.envs.sim.robotwin.dream_env import DreamRobotwinEnv

            env_cls = DreamRobotwinEnv
        return super()._setup_env_and_wrappers(env_cls, env_cfg, num_envs_per_stage)

    def _build_rollout_input_data(self, env_batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "obs": {"states": env_batch["obs"]["states"]},
            "dream_observations": env_batch["env_infos"]["dream_observations"],
        }

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
