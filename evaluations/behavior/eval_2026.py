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

import hydra
from omegaconf import DictConfig

from rlinf.envs.layout import EnvLayout
from rlinf.envs.sim.behavior.utils import resolve_step_limits


@hydra.main(
    version_base="1.1", config_path=".", config_name="behavior2026_openpi_pi05_eval"
)
def main(cfg: DictConfig) -> None:
    from evaluations.eval_embodied_agent import main as run_eval

    for env_cfg in EnvLayout.env_cfgs(cfg.env.eval, "env.eval").values():
        if env_cfg.env_type == "behavior2026":
            resolve_step_limits(env_cfg, cfg.rollout.model.num_action_chunks)
    run_eval(cfg)


if __name__ == "__main__":
    main()
