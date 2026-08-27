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

"""The Wan world model as an environment: :class:`WorldModelEnv` plus :class:`WanBackend`."""

from __future__ import annotations

import os
from typing import Optional

from diffsynth.models.reward_model import ResnetRewModel, TaskEmbedResnetRewModel

from rlinf.envs.world_model.backend import WorldModelBackend
from rlinf.envs.world_model.wan_backend import WanBackend
from rlinf.envs.world_model.world_model_env import WorldModelEnv

__all__ = ["WanEnv"]


class WanEnv(WorldModelEnv):
    def _build_backend(self) -> WorldModelBackend:
        return WanBackend(self.cfg, self.device, self._get_runtime_device_str())

    def _load_reward_model(self):
        if self.cfg.reward_model.type == "ResnetRewModel":
            return ResnetRewModel(self.cfg.reward_model.from_pretrained)
        elif self.cfg.reward_model.type == "TaskEmbedResnetRewModel":
            return TaskEmbedResnetRewModel(
                checkpoint_path=self.cfg.reward_model.from_pretrained,
                task_suite_name=self.cfg.task_suite_name,
            )
        raise ValueError(f"Unknown reward model type: {self.cfg.reward_model.type}")

    def _reward_instructions(self) -> Optional[list[str]]:
        if self.cfg.reward_model.type != "TaskEmbedResnetRewModel":
            return None
        # One instruction per scored frame, so each env's description repeats over its chunk
        instructions = []
        for env_idx in range(self.num_envs):
            instructions.extend([self.task_descriptions[env_idx]] * self.chunk)
        return instructions


# PYTHONPATH="/mnt/project_rlinf/jzn/workspace/release/DiffSynth-Studio:$PYTHONPATH" python -m rlinf.envs.world_model.world_model_wan_env
if __name__ == "__main__":
    from pathlib import Path

    import numpy as np
    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import initialize_config_dir

    # # Set required environment variable
    os.environ.setdefault("EMBODIED_PATH", "examples/embodiment")

    repo_root = Path(__file__).resolve().parents[3]

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    config_dir = Path(
        os.environ.get("EMBODIED_CONFIG_DIR", repo_root / "examples/embodiment/config")
    ).resolve()
    config_name = "wan_libero_spatial_grpo_openvlaoft_quick"

    print(f"Loading config: {config_name} from {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg_ = compose(config_name=config_name)
        cfg = cfg_["env"]["train"]

    env = WanEnv(cfg, cfg.total_num_envs, seed_offset=0, total_num_processes=1)

    # Reset environment
    for i in range(20):
        obs, info = env.reset()

    print("\nAfter reset:")
    print(f"  obs keys: {list(obs.keys())}")

    print("\n" + "-" * 80)

    chunk_steps = cfg.chunk
    num_envs = cfg.total_num_envs

    chunk_traj = 1
    zeros_actions = np.zeros((num_envs, chunk_steps, 7))

    for i in range(chunk_traj):
        print(f"Chunk {i} of {chunk_traj}")
        print("-" * 100)
        o, r, te, tr, infos = env.chunk_step(
            zeros_actions[:, i * chunk_steps : (i + 1) * chunk_steps, :]
        )
