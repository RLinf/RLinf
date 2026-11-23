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

import hydra
import torch.multiprocessing as mp

from examples.agent.train_agent import run_agent_training
from rlinf.agents.searchr1.search_tool_worker import SearchToolWorker
from rlinf.agents.searchr1.searchr1_agent_loop import Searchr1AgentLoopWorker
from rlinf.scheduler import NodePlacementStrategy
from rlinf.utils.utils import output_redirector
from rlinf.workers.agent.tool_worker import ToolWorkerInfo


def _launch_tools(cfg, cluster):
    return {
        SearchToolWorker.create_group(cfg).launch(
            cluster, name="search", placement_strategy=NodePlacementStrategy([0])
        ): ToolWorkerInfo(tool_names=["search"], has_session=False),
    }


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    run_agent_training(cfg, Searchr1AgentLoopWorker, _launch_tools)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
