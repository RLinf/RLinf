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
import torch.multiprocessing as mp
from omegaconf import DictConfig

from examples.agent.train_agent import run_agent_training
from rlinf.agents.envscaler.agent_loop import EnvScalerAgentLoopWorker
from rlinf.utils.utils import output_redirector


@hydra.main(version_base="1.1", config_path="config", config_name="train")
@output_redirector
def main(cfg: DictConfig) -> None:
    """Train an EnvScaler policy with multi-turn GRPO."""
    run_agent_training(cfg, EnvScalerAgentLoopWorker)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
