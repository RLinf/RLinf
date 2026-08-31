# Copyright 2025 The RLinf Authors.
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

"""The OpenSora world model as an env: :class:`WorldModelEnv` + :class:`OpenSoraBackend`."""

from __future__ import annotations

import os

from omegaconf import OmegaConf
from opensora.registry import MODELS, build_module

from rlinf.envs.world_model.backend import WorldModelBackend
from rlinf.envs.world_model.opensora_backend import OpenSoraBackend
from rlinf.envs.world_model.world_model_env import WorldModelEnv

__all__ = ["OpenSoraEnv"]


class OpenSoraEnv(WorldModelEnv):
    def _build_backend(self) -> WorldModelBackend:
        return OpenSoraBackend(self.cfg, self.device, self._get_runtime_device_str())

    def _load_reward_model(self):
        rm_cfg = OmegaConf.to_container(
            self.cfg.world_model_cfg.reward_model, resolve=True
        )
        return build_module(rm_cfg, MODELS)


# PYTHONPATH="/mnt/project_rlinf/jzn/workspace/opensora:$PYTHONPATH" python -m rlinf.envs.world_model.world_model_opensora_env
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
    config_name = "opensora_libero_spatial_grpo_openvlaoft_impl"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg_ = compose(config_name=config_name)
        cfg = cfg_["env"]["train"]

    num_envs = cfg.total_num_envs

    env = OpenSoraEnv(cfg, num_envs, seed_offset=0, total_num_processes=1)

    obs, info = env.reset()
    print("Reset OK. Keys:", list(obs.keys()))

    chunk_steps = cfg.world_model_cfg.chunk
    chunk_traj = 1
    zeros_actions = np.zeros((num_envs, chunk_steps, 7))

    for i in range(chunk_traj):
        print(f"Chunk {i} of {chunk_traj}")
        print("-" * 100)
        o, r, te, tr, infos = env.chunk_step(
            zeros_actions[:, i * chunk_steps : (i + 1) * chunk_steps, :]
        )
