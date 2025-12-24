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

from typing import Optional

import hydra
from node import (
    ComponentNode,
    EnvNode,
    EnvProfiler,
    EnvRolloutNode,
    MegatronNode,
    RolloutNode,
)
from placement import (
    ScheduleMode,
    ScheduleResult,
    SingleNodeScheduleResult,
)
from util import get_global_config, get_valid_gpu_num_list, init_global_config
from workflow import Workflow, traverse_st_cuts

from rlinf.scheduler import Cluster
from rlinf.utils.placement import ModelParallelComponentPlacement


class AutoPlacementWorker:
    def __init__(
        self,
        cfg,
        component_placement,
        graph: Optional[dict[str, list[str]]] = None,
    ):
        init_global_config(cfg, component_placement)
        self.config = get_global_config()
        self.components_config = self.config.components_config
        self._name_to_node_dict: dict[str, ComponentNode] = {}
        self._init_workflow(graph)

    def get_node(self, component_name: str) -> ComponentNode:
        if component_name in self._name_to_node_dict:
            return self._name_to_node_dict[component_name]

        if component_name == "rollout":
            node = RolloutNode()
        elif component_name in ["actor", "inference"]:
            valid_gpu_num_list: list[int] = get_valid_gpu_num_list(component_name)
            node = MegatronNode(
                role=component_name,
                valid_gpu_nums=valid_gpu_num_list,
            )
        elif component_name == "env":
            node = EnvNode(
                profiler=EnvProfiler(
                    self.config.profile_data.env_profile_data,
                    self.config.env_num,
                )
            )
        elif component_name == "env_rollout":
            node = EnvRolloutNode(
                profiler=EnvProfiler(
                    self.config.profile_data.rollout_profile_data,
                    self.config.env_num,
                ),
                model_parallel_size=self.components_config[
                    "env_rollout"
                ].model_parallel_size,
            )
        else:
            raise ValueError(f"{component_name=} is not supported")

        self._name_to_node_dict[component_name] = node
        return node

    def _init_workflow(self, graph: dict[str, list[str]]):
        # Create ComponentNode and conver graph(str) to graph(ComponentNode)
        workflow_graph: dict[ComponentNode, list[ComponentNode]] = {}
        for component_name, neighbors in graph.items():
            node = self.get_node(component_name)
            workflow_graph[node] = [self.get_node(neighbor) for neighbor in neighbors]

        # Compress strongly connected components
        workflow = Workflow(workflow_graph)
        self.workflow = workflow.compress_sccs()

    def _find_schedule(
        self, workflow: Workflow, gpu_num: int
    ) -> Optional[ScheduleResult]:
        key = (workflow, gpu_num)
        if key in self._result_cache:
            return self._result_cache[key]

        if workflow.is_node():
            cost = workflow.profile(gpu_num)
            if cost is None:
                return None

            # For reasoning task, the cost is the cost per group batch.
            if self.config.task_type == "reasoning":
                cost_per_group_batch = cost
                total_cost = cost * self.config.rollout_batch_size
            else:  # For embodiment task, the cost means total cost.
                cost_per_group_batch = cost / self.config.env_num
                total_cost = cost

            self._result_cache[key] = SingleNodeScheduleResult(
                total_gpu_num=gpu_num,
                node=workflow.nodes[0],
                cost_per_group_batch=cost_per_group_batch,
                total_cost=total_cost,
            )

            return self._result_cache[key]

        best_res = None

        cuts = traverse_st_cuts(workflow)
        for source_workflow, sink_workflow in cuts:
            source_res: ScheduleResult = self._find_schedule(source_workflow, gpu_num)
            sink_res: ScheduleResult = self._find_schedule(sink_workflow, gpu_num)
            collocated_res = ScheduleResult.merger_schedule_results(
                gpu_num, source_res, sink_res, is_collocated=True
            )

            best_res = ScheduleResult.find_best_schedule(best_res, collocated_res)

            # Pipeline schedule
            for source_gpu_num in range(1, gpu_num - 1):
                sink_gpu_num = gpu_num - source_gpu_num
                source_res: ScheduleResult = self._find_schedule(
                    source_workflow, source_gpu_num
                )
                sink_res: ScheduleResult = self._find_schedule(
                    sink_workflow, sink_gpu_num
                )

                disaggregated_res = ScheduleResult.merger_schedule_results(
                    gpu_num, source_res, sink_res, is_collocated=False
                )

                best_res = ScheduleResult.find_best_schedule(
                    best_res, disaggregated_res
                )

        self._result_cache[key] = best_res
        return best_res

    def run(self) -> ScheduleResult:
        self._result_cache: dict[tuple[Workflow, int], ScheduleResult] = {}
        return self._find_schedule(self.workflow, self.config.total_gpus)


def get_workflow_graph(cfg) -> dict[str, list[str]]:
    if cfg.runner.task_type == "reasoning":
        if cfg.algorithm.recompute_logprobs:
            return {
                "rollout": ["inference"],
                "inference": ["actor"],
                "actor": [],
            }
        else:
            return {
                "rollout": ["actor"],
                "actor": [],
            }
    elif cfg.runner.task_type == "embodiment":
        return {
            "env": ["env_rollout"],
            "env_rollout": ["actor"],
            "actor": [],
        }
    else:
        raise ValueError(f"{cfg.runner.task_type=} is not supported")


@hydra.main(version_base="1.1")
def main(cfg):
    cluster = Cluster(cfg.cluster.num_nodes)
    component_placement = ModelParallelComponentPlacement(cfg, cluster)

    workflow_graph: dict[str, list[str]] = get_workflow_graph(cfg)
    auto_placement_worker = AutoPlacementWorker(
        cfg, component_placement, workflow_graph
    )

    schedule_result: ScheduleResult = auto_placement_worker.run()

    if (
        schedule_result.mode == ScheduleMode.COLLOCATED
        and not schedule_result.is_hybrid()
    ):
        res = (
            ", ".join(
                [
                    node.role
                    for node in schedule_result.placement
                    if node.role != "inference"
                ]
            )
            + " : all"
        )
    else:
        res = schedule_result.placement_str

    print("=" * 50)
    print("Best placement for this task is:\n")
    print(res)


if __name__ == "__main__":
    main()
