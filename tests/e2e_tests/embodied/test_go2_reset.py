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

import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from rlinf.envs import get_env_cls


def _reset_observations(seed: int) -> torch.Tensor:
    config_dir = Path(__file__).resolve().parents[3] / "examples/embodiment/config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="isaaclab_go2_c_ppo",
            overrides=[f"actor.seed={seed}", "env.train.total_num_envs=4"],
        ).env.train
    cfg.init_params.asset_root = os.environ.get("GO2_ASSET_ROOT")
    env = get_env_cls(cfg.env_type, cfg)(cfg, cfg.total_num_envs)
    try:
        observations = []
        for _ in range(2):
            obs, _ = env.reset()
            observations.append(obs["states"].cpu())
            del obs
        return torch.stack(observations)
    finally:
        env.close()


def test_go2_stairs_seeded_resets():
    first = _reset_observations(0)
    replay = _reset_observations(0)
    other = _reset_observations(2)
    assert first.shape == (2, 4, 235)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(replay, first, rtol=0, atol=1e-5)

    # The stair task samples the forward command without observation noise.
    forward = first[:, :, 9]
    assert ((0.4 <= forward) & (forward <= 0.9)).all()
    assert not torch.equal(forward[0], forward[1])
    assert not torch.equal(forward, other[:, :, 9])
